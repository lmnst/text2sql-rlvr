"""Append-only experiment ledger.

Every number that leaves this project has to be traceable to one line here, so
the fields that make a result reproducible are filled in automatically rather
than remembered. See the "实验记录" section of AGENTS.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LEDGER = Path("results/runs.jsonl")

REQUIRED_FIELDS = ("stage", "split", "n_samples", "metrics")

_STAGES = ("baseline", "sft", "grpo", "ablation", "smoke")

_write_lock = threading.Lock()


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def git_state(
    repo: Path | None = None, ignore_paths: tuple[str | Path, ...] = ()
) -> tuple[str, bool]:
    """Return ``(sha, dirty)``. ``("", True)`` when git is unavailable.

    Unknown provenance is treated as dirty on purpose: a run we cannot pin to a
    commit must not be reportable.

    ``ignore_paths`` exists for the ledger itself. It is tracked and every run
    appends to it, so without this the second run of any session would report a
    dirty tree and the warning would stop meaning anything.
    """
    sha = _git("rev-parse", "HEAD", cwd=repo)
    if not sha:
        return "", True

    args = ["status", "--porcelain"]
    root = _git("rev-parse", "--show-toplevel", cwd=repo) if ignore_paths else ""
    if root:
        # Let git do the excluding. Parsing porcelain output by column offset is
        # brittle -- the status code is two columns wide and the first is often a
        # space, which any stripping of the captured output silently eats.
        excludes = []
        for candidate in ignore_paths:
            try:
                relative = Path(candidate).resolve().relative_to(Path(root).resolve())
            except ValueError:
                continue
            excludes.append(f":(exclude,top){relative.as_posix()}")
        if excludes:
            args += ["--", ":/", *excludes]

    return sha, bool(_git(*args, cwd=repo))


def file_sha256(path: str | Path | None) -> str:
    if not path:
        return ""
    file = Path(path)
    if not file.is_file():
        return ""
    return hashlib.sha256(file.read_bytes()).hexdigest()


def describe_hardware() -> str:
    cpu = platform.processor() or platform.machine()
    gpu = os.environ.get("TEXT2SQL_HARDWARE") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return f"{platform.system()} {platform.release()} / {cpu}" + (f" / gpu={gpu}" if gpu else "")


def append_run(record: dict[str, Any], path: str | Path = DEFAULT_LEDGER) -> dict[str, Any]:
    """Fill in provenance, validate, and append one line. Returns the stored record."""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"ledger record is missing required fields: {', '.join(missing)}")
    if record["stage"] not in _STAGES:
        raise ValueError(f"stage must be one of {_STAGES}, got {record['stage']!r}")

    sha, dirty = git_state(ignore_paths=(path,))
    entry: dict[str, Any] = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "hardware": describe_hardware(),
    }
    entry.update(record)
    entry.setdefault("config_sha256", file_sha256(entry.get("config_path")))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    with _write_lock:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return entry


def read_runs(path: str | Path = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    """Read the ledger back, skipping blank lines."""
    target = Path(path)
    if not target.is_file():
        return []
    runs = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            runs.append(json.loads(line))
    return runs
