"""A minimal execute-observe-revise loop for Text-to-SQL.

The generator so far answers in one shot. This module lets a model *act*
before committing: inspect a table it was not shown, run a candidate query
against the real (read-only) database, read the rows or the error, and try
again. One action per reply, a hard turn budget, and a plain-text protocol
that any chat model can follow without native tool-calling support:

    DESCRIBE <table>            -> the table's DDL and a few example rows
    ```sql ... ```              -> the query is executed; result or error comes back
    FINAL + ```sql ... ```      -> this query is the answer; the loop stops

Observations are appended as ``user`` turns, so a trajectory is an ordinary
chat transcript: it can be scored like any prediction (the ``sql`` field), or
used as SFT / preference data with the loss restricted to assistant turns.

No agent framework is used on purpose. The loop is ~100 lines over pieces
the project already has (sandbox executor, schema rendering, SQL extraction),
which keeps the train-time and inference-time contract in one place.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.schema import DatabaseSchema, fetch_sample_rows, format_schema
from text2sql_rlvr.rewards.sandbox import OK, REJECTED, TIMEOUT, ExecResult, SqlExecutor
from text2sql_rlvr.sql import extract_sql

AGENT_SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite queries. You work step by "
    "step: you may inspect tables and run queries against the real database "
    "before committing to an answer."
)

#: The instruction deliberately spells out the fence in words rather than
#: showing one. An earlier version wrote the literal opener followed by prose;
#: 34 of 788 replies copied that whole phrase as their opening fence, so the
#: extracted "statement" began with the word "code" and the validator rejected
#: it. Never put a fence opener next to text the model could read as a template.
AGENT_INSTRUCTION = (
    "Answer the question with one SQLite SELECT query. Take exactly one action per "
    "reply, and reply with the action only, no explanation:\n"
    "- DESCRIBE <table> shows that table's full definition and a few example rows.\n"
    "- A SQL query on its own, inside a markdown sql code block, is executed: you see "
    "its result or its error.\n"
    "- The word FINAL on its own line, followed by a SQL code block, commits that "
    "query as your answer.\n"
    "Run a query and check its result before FINAL unless you are certain."
)

FINAL, EXECUTE, DESCRIBE, NONE = "final", "execute", "describe", "none"
STOP_FINAL, STOP_BUDGET, STOP_NO_SQL, STOP_REQUEST_ERROR = (
    "final", "budget", "no_sql", "request_error",
)

_FINAL_RE = re.compile(r"^\s*FINAL\b", re.IGNORECASE | re.MULTILINE)
_DESCRIBE_RE = re.compile(
    r"^\s*DESCRIBE\s+[`\"\[]?([^`\"\]\n;]+?)[`\"\]]?\s*;?\s*$", re.IGNORECASE | re.MULTILINE
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

ChatFn = Callable[[list[dict[str, str]]], str]

#: Loop-level guardrails. They do not make an untrained model smarter, but
#: they stop it from spending its whole budget re-reading a table it was
#: already shown or re-running a query whose result it already has, which is
#: exactly what an untrained 1.7B did on the first smoke test (69 of 70
#: DESCRIBEs targeted a table already in the prompt; no episode used FINAL).
NUDGE_ALREADY_SHOWN = (
    "Table {table} is already shown in full above. Do not describe it again: run a "
    "query with a ```sql block, or answer with FINAL and a ```sql block."
)
NUDGE_REPEATED_QUERY = (
    "You already ran exactly this query and saw its result above. If it answers the "
    "question, reply FINAL followed by the same ```sql block; otherwise change the query."
)
HINT_RESULT_OK = (
    "If this result answers the question, reply FINAL followed by the same ```sql block."
)
HINT_LAST_TURN = "This is your last reply: answer with FINAL and a ```sql block."


@dataclass(frozen=True)
class AgentConfig:
    max_turns: int = 4
    max_rows_shown: int = 5
    sample_rows: int = 3
    max_cell_chars: int = 60


@dataclass(frozen=True)
class Action:
    kind: str
    sql: str = ""
    table: str = ""


def parse_action(text: str) -> Action:
    """Classify one assistant reply. FINAL wins, then a query, then DESCRIBE."""
    body = _THINK_RE.sub("", text or "")
    sql = extract_sql(body)
    if sql and _FINAL_RE.search(body):
        return Action(FINAL, sql=sql)
    if sql:
        return Action(EXECUTE, sql=sql)
    match = _DESCRIBE_RE.search(body)
    if match:
        return Action(DESCRIBE, table=match.group(1).strip())
    return Action(NONE)


# ------------------------------------------------------------------ prompt


def build_agent_messages(
    example: BirdExample,
    schema_text: str,
    other_tables: Sequence[str] = (),
) -> list[dict[str, str]]:
    """First two turns. ``other_tables`` are tables left out of ``schema_text``
    (by the selector, say) that the model may still DESCRIBE."""
    parts = [f"Database schema:\n\n{schema_text}"]
    if other_tables:
        parts.append(
            "Other tables in this database, not shown above (use DESCRIBE to see one): "
            + ", ".join(other_tables)
        )
    if example.evidence:
        parts.append(f"External knowledge: {example.evidence}")
    parts.append(f"Question: {example.question}")
    parts.append(AGENT_INSTRUCTION)
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# ------------------------------------------------------------------ observations


def _cell(value: Any, max_chars: int) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def format_result(result: ExecResult, *, max_rows: int, max_cell_chars: int) -> str:
    if result.status == REJECTED:
        return f"Query rejected: {result.error}"
    if result.status == TIMEOUT:
        return f"Query timed out after {result.elapsed_s:.1f}s. Simplify it."
    if result.status != OK:
        return f"Query error: {result.error}"
    n = len(result.rows)
    if n == 0:
        return "Query result: 0 rows."
    shown = result.rows[:max_rows]
    head = f"Query result: {n}{'+' if result.truncated else ''} row(s)"
    if n > len(shown):
        head += f", showing the first {len(shown)}"
    lines = [head + ".", " | ".join(result.columns)]
    lines += [" | ".join(_cell(v, max_cell_chars) for v in row) for row in shown]
    return "\n".join(lines)


def describe_table(
    schema: DatabaseSchema, table_name: str, db_path: str | Path, *, sample_rows: int
) -> str:
    table = schema.table(table_name)
    if table is None:
        names = ", ".join(t.name for t in schema.tables)
        return f"No table named {table_name!r}. Tables in this database: {names}"
    one = DatabaseSchema(schema.db_id, (table,))
    rows = fetch_sample_rows(db_path, one, sample_rows) if sample_rows > 0 else None
    return format_schema(one, style="ddl", sample_rows=rows)


# ------------------------------------------------------------------ episode


@dataclass
class Step:
    turn: int
    action: str
    sql: str = ""
    table: str = ""
    observation: str = ""
    exec_status: str | None = None
    n_rows: int | None = None
    latency_s: float = 0.0


@dataclass
class Episode:
    question_id: int
    db_id: str
    final_sql: str = ""
    stop_reason: str = STOP_BUDGET
    steps: list[Step] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    final_exec_status: str | None = None
    error: str | None = None

    @property
    def n_turns(self) -> int:
        return len(self.steps)

    def count(self, kind: str) -> int:
        return sum(1 for s in self.steps if s.action == kind)

    @property
    def final_verified(self) -> bool:
        """Whether the committed query is one this episode ran successfully.

        A model can watch a query succeed and then commit a different one. That
        answer carries no execution evidence, so trajectory filtering and error
        analysis both need to tell the two cases apart.
        """
        if not self.final_sql:
            return False
        target = " ".join(self.final_sql.split()).casefold()
        return any(
            step.exec_status == OK and " ".join(step.sql.split()).casefold() == target
            for step in self.steps
        )

    @property
    def first_exec_status(self) -> str | None:
        for step in self.steps:
            if step.exec_status is not None:
                return step.exec_status
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "sql": self.final_sql,
            "stop_reason": self.stop_reason,
            "n_turns": self.n_turns,
            "n_executions": self.count(EXECUTE),
            "n_describes": self.count(DESCRIBE),
            "n_exec_errors": sum(
                1 for s in self.steps if s.exec_status not in (None, OK)
            ),
            "first_exec_status": self.first_exec_status,
            "final_exec_status": self.final_exec_status,
            "final_verified": self.final_verified,
            "error": self.error,
            "steps": [vars(s) for s in self.steps],
            "messages": self.messages,
        }


def run_episode(
    example: BirdExample,
    schema: DatabaseSchema,
    schema_text: str,
    db_path: str | Path,
    executor: SqlExecutor,
    chat: ChatFn,
    *,
    other_tables: Sequence[str] = (),
    shown_tables: Sequence[str] = (),
    config: AgentConfig | None = None,
) -> Episode:
    """Drive one question to a final SQL. ``chat`` maps messages to a reply.

    ``shown_tables`` are the tables whose full definition is already in
    ``schema_text``; describing one of them again is answered with a nudge
    instead of the definition.
    """
    config = config or AgentConfig()
    episode = Episode(question_id=example.question_id, db_id=example.db_id)
    messages = build_agent_messages(example, schema_text, other_tables)
    executed: list[tuple[str, str]] = []  # (sql, status) in order
    seen_tables = {name.casefold() for name in shown_tables}

    for turn in range(1, config.max_turns + 1):
        started = time.monotonic()
        try:
            reply = chat(messages)
        except Exception as exc:  # noqa: BLE001 - recorded, episode ends
            episode.error = f"{type(exc).__name__}: {exc}"
            episode.stop_reason = STOP_REQUEST_ERROR
            break
        latency = round(time.monotonic() - started, 3)
        messages.append({"role": "assistant", "content": reply})
        action = parse_action(reply)
        step = Step(turn=turn, action=action.kind, sql=action.sql, table=action.table,
                    latency_s=latency)

        if action.kind == FINAL:
            episode.final_sql = action.sql
            episode.stop_reason = STOP_FINAL
            episode.steps.append(step)
            break

        if action.kind == EXECUTE:
            repeated = any(sql == action.sql for sql, _ in executed)
            result = executor.execute(db_path, action.sql)
            step.exec_status = result.status
            step.n_rows = len(result.rows) if result.status == OK else None
            executed.append((action.sql, result.status))
            if repeated:
                step.observation = NUDGE_REPEATED_QUERY
            else:
                step.observation = format_result(
                    result, max_rows=config.max_rows_shown,
                    max_cell_chars=config.max_cell_chars,
                )
                if result.status == OK:
                    step.observation += "\n" + HINT_RESULT_OK
        elif action.kind == DESCRIBE:
            table = schema.table(action.table)
            if table is not None and table.name.casefold() in seen_tables:
                step.observation = NUDGE_ALREADY_SHOWN.format(table=table.name)
            else:
                step.observation = describe_table(
                    schema, action.table, db_path, sample_rows=config.sample_rows
                )
                if table is not None:
                    seen_tables.add(table.name.casefold())
        else:
            step.observation = (
                "Your reply contained no action. Send DESCRIBE <table>, a ```sql block "
                "to run, or FINAL with a ```sql block."
            )
        episode.steps.append(step)
        if turn < config.max_turns:
            observation = step.observation
            if turn == config.max_turns - 1:
                observation += "\n" + HINT_LAST_TURN
            messages.append({"role": "user", "content": observation})

    if episode.stop_reason != STOP_FINAL and not episode.final_sql:
        ok = [sql for sql, status in executed if status == OK]
        if ok:
            episode.final_sql = ok[-1]
        elif executed:
            episode.final_sql = executed[-1][0]
        elif episode.stop_reason == STOP_BUDGET:
            episode.stop_reason = STOP_NO_SQL

    if episode.final_sql:
        episode.final_exec_status = executor.execute(db_path, episode.final_sql).status
    episode.messages = messages
    return episode
