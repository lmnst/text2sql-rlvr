"""Run model-generated SQL against a BIRD database without trusting it.

Three independent guards, because any one of them can be talked around:

1. :func:`text2sql_rlvr.sql.validate_read_only` rejects non-SELECT text early and
   gives a readable reason.
2. The connection is opened read-only, so the file cannot be written even in
   principle.
3. A sqlite authorizer denies every action except reads, so a statement that
   slips past the text-level check still cannot mutate or attach anything.

Timeouts use a progress handler rather than a watchdog thread: SQLite checks it
between VM instructions and unwinds the query itself, leaving the connection
reusable. That matters because reward computation reuses connections heavily.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from text2sql_rlvr.sql import validate_read_only

OK = "ok"
REJECTED = "rejected"
ERROR = "error"
TIMEOUT = "timeout"

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_ROWS = 10_000

# Authorizer action codes. Python only exposes these as attributes from 3.11 on,
# but the numeric values are fixed by the SQLite C API.
_SQLITE_READ = getattr(sqlite3, "SQLITE_READ", 20)
_SQLITE_SELECT = getattr(sqlite3, "SQLITE_SELECT", 21)
_SQLITE_FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
_SQLITE_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_ALLOWED_ACTIONS = frozenset({_SQLITE_READ, _SQLITE_SELECT, _SQLITE_FUNCTION, _SQLITE_RECURSIVE})

#: VM instructions between progress-handler callbacks. Small enough to keep the
#: timeout tight, large enough that the callback is not a measurable cost.
_PROGRESS_INTERVAL = 1_000


@dataclass(frozen=True)
class ExecResult:
    """Outcome of executing one statement."""

    status: str
    rows: tuple[tuple[Any, ...], ...] = ()
    columns: tuple[str, ...] = ()
    error: str | None = None
    elapsed_s: float = 0.0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def n_rows(self) -> int:
        return len(self.rows)


def _authorizer(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _decode(raw: bytes) -> str:
    # Several BIRD databases store TEXT that is not valid UTF-8. The default
    # text factory raises on those rows and would silently turn a correct query
    # into an execution error.
    return raw.decode("utf-8", errors="replace")


def open_read_only(db_path: str | Path, *, access: str = "immutable") -> sqlite3.Connection:
    """Open a read-only connection *without* the statement authorizer.

    Only for our own introspection queries, which need ``PRAGMA``. Never pass
    model output to a connection from here -- use :func:`connect` for that.

    ``access="immutable"`` promises SQLite the file will not change, which skips
    all locking and is the right choice for a fixed benchmark corpus. Use
    ``access="ro"`` if the file may be written by something else.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    if access not in {"immutable", "ro"}:
        raise ValueError(f"access must be 'immutable' or 'ro', got {access!r}")

    flag = "immutable=1" if access == "immutable" else "mode=ro"
    conn = sqlite3.connect(
        f"{path.as_uri()}?{flag}",
        uri=True,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.text_factory = _decode
    return conn


def connect(db_path: str | Path, *, access: str = "immutable") -> sqlite3.Connection:
    """Open a hardened read-only connection for running untrusted SQL."""
    conn = open_read_only(db_path, access=access)
    conn.set_authorizer(_authorizer)
    return conn


def execute_on(
    conn: sqlite3.Connection,
    sql: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
    validate: bool = True,
) -> ExecResult:
    """Execute one statement on an already-open hardened connection."""
    if validate:
        check = validate_read_only(sql)
        if not check.ok:
            return ExecResult(REJECTED, error=check.reason)
        sql = check.statement

    deadline = time.monotonic() + timeout_s
    timed_out = False

    def progress() -> int:
        nonlocal timed_out
        if time.monotonic() >= deadline:
            timed_out = True
            return 1
        return 0

    started = time.monotonic()
    cursor: sqlite3.Cursor | None = None
    conn.set_progress_handler(progress, _PROGRESS_INTERVAL)
    try:
        cursor = conn.execute(sql)
        columns = tuple(d[0] for d in cursor.description or ())
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = tuple(tuple(row) for row in fetched[:max_rows])
        return ExecResult(OK, rows, columns, None, time.monotonic() - started, truncated)
    except Exception as exc:  # noqa: BLE001 - any failure is just a failed reward
        status = TIMEOUT if timed_out else ERROR
        return ExecResult(status, error=f"{type(exc).__name__}: {exc}",
                          elapsed_s=time.monotonic() - started)
    finally:
        conn.set_progress_handler(None, 0)
        if cursor is not None:
            cursor.close()


def execute_sql(
    db_path: str | Path,
    sql: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
    validate: bool = True,
    access: str = "immutable",
) -> ExecResult:
    """Open a connection, run one statement, close it. Stateless but not cheap."""
    try:
        conn = connect(db_path, access=access)
    except Exception as exc:  # noqa: BLE001
        return ExecResult(ERROR, error=f"{type(exc).__name__}: {exc}")
    try:
        return execute_on(conn, sql, timeout_s=timeout_s, max_rows=max_rows, validate=validate)
    finally:
        conn.close()


@dataclass
class ExecutorStats:
    """Counters for one :class:`SqlExecutor`."""

    calls: int = 0
    cache_hits: int = 0
    ok: int = 0
    rejected: int = 0
    errors: int = 0
    timeouts: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass
class SqlExecutor:
    """Reusable executor with per-thread connections and a shared result cache.

    Reward computation, not the GPU, is the throughput bottleneck during RL:
    every rollout in every group needs an execution. Reopening a multi-gigabyte
    database per call, and re-running identical SQL that different rollouts
    happened to produce, are the two costs worth removing up front.

    SQLite releases the GIL while a statement runs, so threads give real
    parallelism here and avoid the pickling and spawn cost of processes.
    """

    timeout_s: float = DEFAULT_TIMEOUT_S
    max_rows: int = DEFAULT_MAX_ROWS
    access: str = "immutable"
    result_cache_size: int = 200_000
    connections_per_thread: int = 32

    stats: ExecutorStats = field(default_factory=ExecutorStats)
    _local: threading.local = field(default_factory=threading.local, repr=False)
    _all_connections: list[sqlite3.Connection] = field(default_factory=list, repr=False)
    _results: OrderedDict[tuple, ExecResult] = field(default_factory=OrderedDict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _connection(self, db_path: Path) -> sqlite3.Connection:
        cache: OrderedDict[Path, sqlite3.Connection] | None = getattr(self._local, "conns", None)
        if cache is None:
            cache = OrderedDict()
            self._local.conns = cache

        conn = cache.get(db_path)
        if conn is not None:
            cache.move_to_end(db_path)
            return conn

        conn = connect(db_path, access=self.access)
        cache[db_path] = conn
        with self._lock:
            self._all_connections.append(conn)
        while len(cache) > self.connections_per_thread:
            _, evicted = cache.popitem(last=False)
            evicted.close()
            with self._lock:
                if evicted in self._all_connections:
                    self._all_connections.remove(evicted)
        return conn

    def execute(self, db_path: str | Path, sql: str, *, use_cache: bool = True) -> ExecResult:
        path = Path(db_path).resolve()
        key = (str(path), sql, self.timeout_s, self.max_rows)

        if use_cache:
            with self._lock:
                hit = self._results.get(key)
                if hit is not None:
                    self._results.move_to_end(key)
                    self.stats.calls += 1
                    self.stats.cache_hits += 1
                    return hit

        try:
            conn = self._connection(path)
        except Exception as exc:  # noqa: BLE001
            result = ExecResult(ERROR, error=f"{type(exc).__name__}: {exc}")
        else:
            result = execute_on(conn, sql, timeout_s=self.timeout_s, max_rows=self.max_rows)

        with self._lock:
            self.stats.calls += 1
            if result.status == OK:
                self.stats.ok += 1
            elif result.status == REJECTED:
                self.stats.rejected += 1
            elif result.status == TIMEOUT:
                self.stats.timeouts += 1
            else:
                self.stats.errors += 1

            # Timeouts are wall-clock dependent, so caching them would freeze a
            # transient result into every later epoch.
            if use_cache and result.status != TIMEOUT:
                self._results[key] = result
                self._results.move_to_end(key)
                while len(self._results) > self.result_cache_size:
                    self._results.popitem(last=False)

        return result

    def clear_cache(self) -> None:
        with self._lock:
            self._results.clear()

    def close(self) -> None:
        with self._lock:
            connections = list(self._all_connections)
            self._all_connections.clear()
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._local = threading.local()

    def __enter__(self) -> SqlExecutor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
