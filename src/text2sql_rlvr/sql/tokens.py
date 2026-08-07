"""A minimal SQLite-aware scanner.

Downstream code needs to tell real SQL code apart from string literals, quoted
identifiers and comments. Regexes get this wrong often enough to matter: BIRD
gold queries are full of backtick identifiers, and string values legitimately
contain ``;`` and ``--``. One small scanner removes that whole class of bug.

The scanner is deliberately lexical only. It does not parse SQL and makes no
attempt to validate grammar; that is the database's job.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

CODE = "code"
STRING = "string"
IDENT = "ident"
COMMENT = "comment"

# open char -> (close char, whether doubling the close char escapes it)
_QUOTES: dict[str, tuple[str, bool]] = {
    "'": ("'", True),  # string literal
    '"': ('"', True),  # standard quoted identifier
    "`": ("`", True),  # MySQL-style identifier, used heavily in BIRD
    "[": ("]", False),  # MS Access-style identifier, no escape mechanism
}

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9$]*")


@dataclass(frozen=True)
class Segment:
    """A contiguous span of the input classified as code, string, ident or comment."""

    kind: str
    text: str
    start: int
    end: int


def _scan_quoted(sql: str, start: int, close: str, doubled_escape: bool) -> int:
    """Return the index just past the closing quote, or ``len(sql)`` if unterminated."""
    i = start + 1
    n = len(sql)
    while i < n:
        if sql[i] == close:
            if doubled_escape and i + 1 < n and sql[i + 1] == close:
                i += 2
                continue
            return i + 1
        i += 1
    return n


def scan(sql: str) -> list[Segment]:
    """Split ``sql`` into segments. Concatenating ``seg.text`` reproduces the input."""
    segments: list[Segment] = []
    n = len(sql)
    i = 0
    code_start = 0

    def flush_code(end: int) -> None:
        if end > code_start:
            segments.append(Segment(CODE, sql[code_start:end], code_start, end))

    while i < n:
        ch = sql[i]
        if ch in _QUOTES:
            close, doubled = _QUOTES[ch]
            flush_code(i)
            end = _scan_quoted(sql, i, close, doubled)
            segments.append(Segment(STRING if ch == "'" else IDENT, sql[i:end], i, end))
            i = code_start = end
            continue
        if ch == "-" and sql.startswith("--", i):
            flush_code(i)
            nl = sql.find("\n", i)
            end = n if nl == -1 else nl + 1
            segments.append(Segment(COMMENT, sql[i:end], i, end))
            i = code_start = end
            continue
        if ch == "/" and sql.startswith("/*", i):
            flush_code(i)
            close = sql.find("*/", i + 2)
            end = n if close == -1 else close + 2
            segments.append(Segment(COMMENT, sql[i:end], i, end))
            i = code_start = end
            continue
        i += 1

    flush_code(n)
    return segments


def strip_comments(sql: str) -> str:
    """Remove comments, replacing each with a space so tokens cannot fuse."""
    return "".join(" " if seg.kind == COMMENT else seg.text for seg in scan(sql))


def normalize_whitespace(sql: str) -> str:
    """Collapse runs of whitespace outside of string literals into single spaces."""
    out = []
    for seg in scan(sql):
        if seg.kind == COMMENT:
            out.append(" ")
        elif seg.kind == STRING:
            out.append(seg.text)
        else:
            out.append(re.sub(r"\s+", " ", seg.text))
    return "".join(out).strip()


def _is_blank(statement: str) -> bool:
    return not strip_comments(statement).strip()


def split_statements(sql: str) -> list[str]:
    """Split on semicolons that appear in code, dropping blank/comment-only pieces."""
    statements: list[str] = []
    buffer: list[str] = []

    for seg in scan(sql):
        if seg.kind != CODE:
            buffer.append(seg.text)
            continue
        chunk: list[str] = []
        for ch in seg.text:
            if ch == ";":
                buffer.append("".join(chunk))
                statements.append("".join(buffer))
                buffer, chunk = [], []
            else:
                chunk.append(ch)
        buffer.append("".join(chunk))

    statements.append("".join(buffer))
    return [s for s in (raw.strip() for raw in statements) if not _is_blank(s)]


def iter_code_words(sql: str) -> Iterator[tuple[str, int, str]]:
    """Yield ``(upper_word, paren_depth, next_non_space_char)`` for words in code spans.

    Quoted identifiers, string literals and comments are skipped entirely, so a
    column literally named ``[delete]`` is never mistaken for the statement.
    """
    depth = 0
    for seg in scan(sql):
        if seg.kind != CODE:
            continue
        text = seg.text
        pos = 0
        while pos < len(text):
            ch = text[pos]
            if ch == "(":
                depth += 1
                pos += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                pos += 1
                continue
            match = _WORD_RE.match(text, pos)
            if match:
                tail = text[match.end() :].lstrip()
                yield match.group(0).upper(), depth, tail[:1]
                pos = match.end()
                continue
            pos += 1


def leading_keyword(sql: str) -> str:
    """Return the first keyword of the statement in upper case, or ``""`` if none."""
    for word, _depth, _next_char in iter_code_words(sql):
        return word
    return ""


def has_top_level_order_by(sql: str) -> bool:
    """True if the statement has an ``ORDER BY`` that constrains its own output order.

    ``ORDER BY`` inside a subquery, a CTE, a window frame or ``group_concat`` sits at
    paren depth > 0 and does not determine the order of the rows the caller sees.
    """
    previous = ""
    for word, depth, _next_char in iter_code_words(sql):
        if previous == "ORDER" and word == "BY" and depth == 0:
            return True
        previous = word
    return False
