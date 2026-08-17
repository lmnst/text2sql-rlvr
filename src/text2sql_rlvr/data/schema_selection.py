"""Per-question schema selection for prompt-length ablations.

``full`` is the production baseline. ``linked`` is a deterministic lexical
schema linker that only sees the question and BIRD evidence. ``oracle`` is a
diagnostic upper bound that is allowed to inspect the gold SQL; it must never
be used for a reported model score or for RL training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.schema import DatabaseSchema, Table, format_schema
from text2sql_rlvr.sql.tokens import COMMENT, STRING, scan

SCHEMA_MODES = ("full", "linked", "oracle")

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "all", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "each", "every", "for", "from", "give", "how", "in", "is", "it", "list",
    "many", "of", "on", "or", "show", "that", "the", "their", "there", "to",
    "what", "when", "where", "which", "who", "with", "id",
}


@dataclass(frozen=True)
class SchemaSelection:
    schema: DatabaseSchema
    mode: str
    selected_tables: tuple[str, ...]
    total_tables: int
    rendered_chars: int = 0
    exceeded_char_budget: bool = False


def _tokens(text: str) -> list[str]:
    # BIRD schemas heavily use CamelCase (EmployeePayHistory, UnitMeasure),
    # while questions use normal words. Lowercasing before this split would
    # erase exactly the boundary the linker needs.
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).replace("_", " ")
    return _WORD_RE.findall(expanded.casefold())


def _words(text: str) -> set[str]:
    return {word for word in _tokens(text) if word not in _STOP_WORDS}


def _phrase(text: str) -> str:
    return " ".join(_tokens(text))


def _table_score(table: Table, question: str, evidence: str) -> int:
    text = f"{question} {evidence}".casefold()
    text_phrase = _phrase(text)
    query_words = _words(text)
    score = 0

    table_phrase = _phrase(table.name)
    table_words = _words(table.name)
    if table_phrase and table_phrase in text_phrase:
        score += 12
    score += 4 * len(query_words & table_words)

    for column in table.columns:
        column_phrase = _phrase(column.name)
        column_words = _words(column.name)
        if column_phrase and column_phrase in text_phrase:
            score += 5
        score += len(query_words & column_words)
    return score


def _fk_neighbours(schema: DatabaseSchema, names: set[str]) -> set[str]:
    lower_to_name = {table.name.casefold(): table.name for table in schema.tables}
    neighbours: set[str] = set()
    lowered = {name.casefold() for name in names}
    for table in schema.tables:
        for foreign_key in table.foreign_keys:
            target = lower_to_name.get(foreign_key.to_table.casefold())
            if target is None:
                continue
            if table.name.casefold() in lowered:
                neighbours.add(target)
            if target.casefold() in lowered:
                neighbours.add(table.name)
    return neighbours


def foreign_key_graph(schema: DatabaseSchema) -> dict[str, set[str]]:
    """Return the schema's declared foreign-key graph with canonical table names."""
    lower_to_name = {table.name.casefold(): table.name for table in schema.tables}
    graph = {table.name: set() for table in schema.tables}
    for table in schema.tables:
        for foreign_key in table.foreign_keys:
            target = lower_to_name.get(foreign_key.to_table.casefold())
            if target is None:
                continue
            graph[table.name].add(target)
            graph[target].add(table.name)
    return graph


def required_tables_fk_connected(
    schema: DatabaseSchema,
    required_tables: tuple[str, ...],
    *,
    retained_tables: tuple[str, ...] | None = None,
) -> bool:
    """Whether retained tables preserve an FK path connecting all required tables.

    This is a schema-level diagnostic, not proof that the gold SQL uses those
    exact foreign-key predicates. Intermediate bridge tables are allowed.
    """
    canonical = {table.name.casefold(): table.name for table in schema.tables}
    required = {
        canonical[name.casefold()]
        for name in required_tables
        if name.casefold() in canonical
    }
    if len(required) != len({name.casefold() for name in required_tables}):
        return False
    if len(required) <= 1:
        return True

    allowed = (
        set(canonical.values())
        if retained_tables is None
        else {
            canonical[name.casefold()]
            for name in retained_tables
            if name.casefold() in canonical
        }
    )
    if not required <= allowed:
        return False

    graph = foreign_key_graph(schema)
    start = next(iter(required))
    seen = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for neighbour in graph[current]:
            if neighbour in allowed and neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
    return required <= seen


def linked_table_names(
    schema: DatabaseSchema,
    question: str,
    evidence: str = "",
    *,
    min_tables: int = 3,
) -> tuple[str, ...]:
    """Rank tables using only lexical evidence available at inference time."""
    ranked = sorted(
        schema.tables,
        key=lambda table: (-_table_score(table, question, evidence), table.name.casefold()),
    )
    direct = [table.name for table in ranked if _table_score(table, question, evidence) > 0]
    selected = direct or [table.name for table in ranked[:min_tables]]
    selected = list(dict.fromkeys(selected + sorted(_fk_neighbours(schema, set(selected)))))
    for table in ranked:
        if len(selected) >= min(min_tables, len(schema.tables)):
            break
        if table.name not in selected:
            selected.append(table.name)
    rank = {table.name: index for index, table in enumerate(ranked)}
    return tuple(sorted(selected, key=lambda name: rank.get(name, len(rank))))


def _sql_without_values_or_comments(sql: str) -> str:
    parts = []
    for segment in scan(sql):
        if segment.kind in (STRING, COMMENT):
            parts.append(" ")
        else:
            parts.append(segment.text.strip('"`[]'))
    return " ".join(parts).casefold()


def oracle_table_names(schema: DatabaseSchema, gold_sql: str) -> tuple[str, ...]:
    """Return schema tables mentioned by gold SQL for an oracle-only ablation."""
    sql = _sql_without_values_or_comments(gold_sql)
    found = []
    for table in schema.tables:
        pattern = rf"(?<![a-z0-9_$]){re.escape(table.name.casefold())}(?![a-z0-9_$])"
        if re.search(pattern, sql):
            found.append(table.name)
    return tuple(found)


def _subset(schema: DatabaseSchema, names: tuple[str, ...]) -> DatabaseSchema:
    by_name = {table.name.casefold(): table for table in schema.tables}
    tables = tuple(by_name[name.casefold()] for name in names if name.casefold() in by_name)
    return DatabaseSchema(db_id=schema.db_id, tables=tables)


def select_schema(
    schema: DatabaseSchema,
    example: BirdExample,
    *,
    mode: str = "full",
    min_tables: int = 3,
) -> SchemaSelection:
    if mode not in SCHEMA_MODES:
        raise ValueError(f"schema mode must be one of {SCHEMA_MODES}, got {mode!r}")
    if mode == "full":
        names = tuple(table.name for table in schema.tables)
    elif mode == "linked":
        names = linked_table_names(
            schema,
            example.question,
            example.evidence,
            min_tables=min_tables,
        )
    else:
        if not example.gold_sql:
            raise ValueError("oracle schema selection requires gold SQL")
        names = oracle_table_names(schema, example.gold_sql)
        if not names:
            raise ValueError(f"oracle selector found no tables for question {example.question_id}")
    return SchemaSelection(
        schema=_subset(schema, names),
        mode=mode,
        selected_tables=names,
        total_tables=len(schema.tables),
    )


def render_selected_schema(
    schema: DatabaseSchema,
    example: BirdExample,
    *,
    mode: str = "full",
    style: str = "ddl",
    include_descriptions: bool = False,
    sample_rows: dict[str, list[tuple[object, ...]]] | None = None,
    max_chars: int = 0,
) -> tuple[str, SchemaSelection]:
    """Select and render whole table blocks, never cutting a DDL statement in half."""
    selection = select_schema(schema, example, mode=mode)
    selected = list(selection.schema.tables)
    exceeded = False
    if max_chars > 0 and mode != "full":
        kept: list[Table] = []
        for table in selected:
            candidate = DatabaseSchema(schema.db_id, tuple(kept + [table]))
            text = format_schema(
                candidate,
                style=style,
                include_descriptions=include_descriptions,
                sample_rows=sample_rows,
            )
            if kept and len(text) > max_chars:
                exceeded = True
                continue
            kept.append(table)
        selected = kept

    rendered_schema = DatabaseSchema(schema.db_id, tuple(selected))
    text = format_schema(
        rendered_schema,
        style=style,
        include_descriptions=include_descriptions,
        sample_rows=sample_rows,
    )
    result = SchemaSelection(
        schema=rendered_schema,
        mode=mode,
        selected_tables=tuple(table.name for table in selected),
        total_tables=len(schema.tables),
        rendered_chars=len(text),
        exceeded_char_budget=exceeded or (max_chars > 0 and len(text) > max_chars),
    )
    return text, result
