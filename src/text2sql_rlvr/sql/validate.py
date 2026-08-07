"""Extract SQL from model output and reject anything that is not a read-only query.

This is a pre-filter that produces a readable rejection reason. It is not the
security boundary -- the sqlite authorizer in :mod:`text2sql_rlvr.rewards.sandbox`
is. Both are applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from text2sql_rlvr.sql.tokens import iter_code_words, leading_keyword, split_statements

#: Statements that only read. ``VALUES (1),(2)`` is a legal standalone SQLite statement.
READ_ONLY_LEADING = frozenset({"SELECT", "WITH", "VALUES"})

#: Words that indicate the statement mutates state, loads code or changes the session.
FORBIDDEN_WORDS = frozenset(
    {
        "ALTER",
        "ANALYZE",
        "ATTACH",
        "BEGIN",
        "COMMIT",
        "CREATE",
        "DELETE",
        "DETACH",
        "DROP",
        "INSERT",
        "LOAD_EXTENSION",
        "PRAGMA",
        "REINDEX",
        "RELEASE",
        "REPLACE",
        "ROLLBACK",
        "SAVEPOINT",
        "UPDATE",
        "VACUUM",
    }
)

#: Forbidden words that are also legal scalar functions when followed by ``(``.
FUNCTION_FORMS = frozenset({"REPLACE"})

_FENCE = r"```[ \t]*(?:sql|sqlite)?[ \t]*\r?\n?"
_FLAGS = re.DOTALL | re.IGNORECASE

_THINK_RE = re.compile(r"^.*</think>", _FLAGS)
_CLOSED_FENCE_RE = re.compile(_FENCE + r"(.*?)```", _FLAGS)
_OPEN_FENCE_RE = re.compile(_FENCE + r"(.*)\Z", _FLAGS)
_TAG_RE = re.compile(r"<(sql|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_BARE_START_RE = re.compile(r"(?im)^[ \t]*(SELECT|WITH)\b")


@dataclass(frozen=True)
class Validation:
    """Outcome of read-only validation."""

    ok: bool
    statement: str
    reason: str | None = None


def extract_sql(text: str) -> str:
    """Pull the final SQL query out of a raw model completion.

    Prefers the *last* fenced block: models routinely sketch a candidate query
    while reasoning and only commit to an answer at the end.
    """
    if not text:
        return ""

    body = _THINK_RE.sub("", text, count=1)

    closed = [m for m in _CLOSED_FENCE_RE.findall(body) if m.strip()]
    if closed:
        return _tidy(closed[-1])

    # A truncated generation can leave an unterminated fence.
    open_fence = _OPEN_FENCE_RE.search(body)
    if open_fence and open_fence.group(1).strip():
        return _tidy(open_fence.group(1))

    tagged = [m[1] for m in _TAG_RE.findall(body) if m[1].strip()]
    if tagged:
        return _tidy(tagged[-1])

    starts = list(_BARE_START_RE.finditer(body))
    if starts:
        return _tidy(body[starts[-1].start() :])

    return ""


def _tidy(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def validate_read_only(sql: str) -> Validation:
    """Check that ``sql`` is exactly one read-only statement."""
    text = (sql or "").strip()
    if not text:
        return Validation(False, "", "empty query")

    statements = split_statements(text)
    if not statements:
        return Validation(False, "", "no executable statement")
    if len(statements) > 1:
        return Validation(False, "", f"expected 1 statement, found {len(statements)}")

    statement = statements[0]
    keyword = leading_keyword(statement)
    if keyword not in READ_ONLY_LEADING:
        return Validation(False, statement, f"statement starts with {keyword or '<none>'}")

    for word, _depth, next_char in iter_code_words(statement):
        if word not in FORBIDDEN_WORDS:
            continue
        if word in FUNCTION_FORMS and next_char == "(":
            continue
        return Validation(False, statement, f"forbidden keyword {word}")

    return Validation(True, statement, None)
