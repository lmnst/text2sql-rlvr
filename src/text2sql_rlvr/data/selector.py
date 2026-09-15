"""Supervised data for a trained schema selector.

The lexical linker in :mod:`text2sql_rlvr.data.schema_selection` keeps every
table with any word overlap and then adds their foreign-key neighbours. On the
fixed val split that gave 99% per-table recall by keeping about 60% of each
database: the gold SQL uses 2 tables on average, the linker kept 25. A trained
selector has to be judged on two numbers at once, the share of questions whose
gold tables are *all* kept and the share of kept tables that are actually used.

Labels come from the gold SQL, so nothing is annotated by hand. That also means
the selector must never train on questions from a database it is later
evaluated on: it would have seen those databases' answers.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.schema import Column, DatabaseSchema, Table
from text2sql_rlvr.data.schema_selection import (
    foreign_key_graph,
    lexical_table_ranking,
    linked_table_names,
    oracle_table_names,
)
from text2sql_rlvr.sql.tokens import COMMENT, IDENT, STRING, scan

SELECTOR_SYSTEM_PROMPT = (
    "You are an expert data analyst. Given a database schema and a question, "
    "you decide which tables a SQL query needs. You answer with a JSON list of "
    "table names and nothing else."
)

SELECTOR_INSTRUCTION = (
    "Which tables from the schema above does a SQLite query answering the question "
    "need? Include every table the query must read from or join through, and no "
    "others.\n"
    "Return only a JSON list of table names, spelled exactly as in the schema."
)

#: ``columns`` target: the model first lists every column the query touches as
#: ``table.column`` and only then the tables. The first selector missed bridge
#: tables because it never had to say *which column* it needed; naming the
#: column forces it to locate the attribute in the schema.
SELECTOR_INSTRUCTION_COLUMNS = (
    "Which columns and tables from the schema above does a SQLite query answering "
    "the question need? List every column the query reads, filters, joins on, groups "
    "by or orders by as table.column, then every table the query must read from or "
    "join through.\n"
    'Return only a JSON object of the form {"columns": [...], "tables": [...]}, '
    "with names spelled exactly as in the schema."
)

TARGET_FORMATS = ("tables", "columns")
SELECTOR_INSTRUCTIONS = {
    "tables": SELECTOR_INSTRUCTION,
    "columns": SELECTOR_INSTRUCTION_COLUMNS,
}

#: Database-size buckets (by table count) used to stratify held-out databases,
#: so the selector is measured on small, medium and large schemas rather than
#: on whichever sizes a plain random draw happens to pick.
TABLE_COUNT_BUCKETS: tuple[tuple[int, int], ...] = ((1, 4), (5, 8), (9, 15), (16, 10**6))

#: A gold table ranked below this position by the lexical scorer is one that
#: no word-overlap rule finds; those questions are the reason to train a model.
HARD_LEXICAL_RANK = 10

#: Questions needing at least this many tables are kept preferentially: they
#: are where the lexical linker misses most and where a short answer teaches
#: the model the least.
HARD_MIN_GOLD_TABLES = 3


# --------------------------------------------------------------------------- split


@dataclass(frozen=True)
class SelectorSplitPlan:
    """Which databases the selector trains on and which it is measured on."""

    train_db_ids: tuple[str, ...]
    heldout_db_ids: tuple[str, ...]
    train_ids: tuple[int, ...]
    heldout_ids: tuple[int, ...]
    criteria: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "n_train": len(self.train_ids),
            "n_heldout": len(self.heldout_ids),
            "n_train_databases": len(self.train_db_ids),
            "n_heldout_databases": len(self.heldout_db_ids),
            "criteria": self.criteria,
        }


def bucket_of(n_tables: int) -> int:
    for index, (low, high) in enumerate(TABLE_COUNT_BUCKETS):
        if low <= n_tables <= high:
            return index
    raise ValueError(f"table count {n_tables} falls in no bucket")


def _quotas(bucket_sizes: Sequence[int], total: int, floor: int) -> list[int]:
    """Split ``total`` across buckets: ``floor`` each where possible, the rest by size."""
    quotas = [min(floor, size) for size in bucket_sizes]
    remaining = total - sum(quotas)
    if remaining < 0:
        raise ValueError(f"cannot hold out {total} databases with floor {floor} per bucket")
    room = [size - quota for size, quota in zip(bucket_sizes, quotas, strict=True)]
    if remaining > sum(room):
        raise ValueError(f"only {sum(bucket_sizes)} eligible databases, asked for {total}")
    weight = sum(size for size, r in zip(bucket_sizes, room, strict=True) if r > 0) or 1
    shares = [
        (remaining * size / weight if r > 0 else 0.0)
        for size, r in zip(bucket_sizes, room, strict=True)
    ]
    extra = [min(int(share), r) for share, r in zip(shares, room, strict=True)]
    leftover = remaining - sum(extra)
    order = sorted(
        range(len(bucket_sizes)),
        key=lambda i: (-(shares[i] - int(shares[i])), -bucket_sizes[i], i),
    )
    while leftover > 0:
        progressed = False
        for i in order:
            if leftover == 0:
                break
            if room[i] - extra[i] > 0:
                extra[i] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return [q + e for q, e in zip(quotas, extra, strict=True)]


def plan_selector_split(
    examples: Sequence[BirdExample],
    table_counts: Mapping[str, int],
    *,
    n_heldout: int = 10,
    seed: int = 0,
    min_questions: int = 40,
    floor_per_bucket: int = 2,
    exclude_db_ids: Iterable[str] = (),
) -> SelectorSplitPlan:
    """Choose held-out databases stratified by size; everything else trains.

    ``exclude_db_ids`` are databases that belong to another evaluation set (the
    SQL generator's fixed val) and must appear on neither side. A database with
    fewer than ``min_questions`` questions is never held out: a metric over a
    handful of questions would swing too much to compare anything.
    """
    excluded = set(exclude_db_ids)
    by_db: dict[str, list[BirdExample]] = defaultdict(list)
    for example in examples:
        if example.db_id in excluded:
            continue
        if example.db_id not in table_counts:
            raise ValueError(f"no table count for database {example.db_id!r}")
        by_db[example.db_id].append(example)
    if n_heldout >= len(by_db):
        raise ValueError(f"cannot hold out {n_heldout} of {len(by_db)} databases")

    rng = random.Random(seed)
    buckets: list[list[str]] = [[] for _ in TABLE_COUNT_BUCKETS]
    for db_id in sorted(by_db):
        if len(by_db[db_id]) >= min_questions:
            buckets[bucket_of(table_counts[db_id])].append(db_id)
    for bucket in buckets:
        rng.shuffle(bucket)

    quotas = _quotas([len(b) for b in buckets], n_heldout, floor_per_bucket)
    heldout = sorted(
        db_id
        for bucket, quota in zip(buckets, quotas, strict=True)
        for db_id in bucket[:quota]
    )
    heldout_set = set(heldout)
    train_dbs = sorted(db_id for db_id in by_db if db_id not in heldout_set)
    train_set = set(train_dbs)

    return SelectorSplitPlan(
        train_db_ids=tuple(train_dbs),
        heldout_db_ids=tuple(heldout),
        train_ids=tuple(e.question_id for e in examples if e.db_id in train_set),
        heldout_ids=tuple(e.question_id for e in examples if e.db_id in heldout_set),
        criteria={
            "seed": seed,
            "n_heldout_requested": n_heldout,
            "min_questions": min_questions,
            "floor_per_bucket": floor_per_bucket,
            "table_count_buckets": [list(b) for b in TABLE_COUNT_BUCKETS],
            "excluded_db_ids": sorted(excluded),
            "n_input": len(examples),
        },
    )


# --------------------------------------------------------------------------- cases


@dataclass(frozen=True)
class SelectorCase:
    """One question with its label and how the lexical linker fares on it."""

    question_id: int
    db_id: str
    gold_tables: tuple[str, ...]
    n_tables_total: int
    linked_tables: tuple[str, ...]
    lexical_worst_rank: int

    @property
    def linked_keeps_all(self) -> bool:
        kept = {name.casefold() for name in self.linked_tables}
        return all(name.casefold() in kept for name in self.gold_tables)

    @property
    def hard_reasons(self) -> tuple[str, ...]:
        reasons = []
        if len(self.gold_tables) >= HARD_MIN_GOLD_TABLES:
            reasons.append("multi_table")
        if not self.linked_keeps_all:
            reasons.append("linker_misses_gold")
        if self.lexical_worst_rank > HARD_LEXICAL_RANK:
            reasons.append("gold_ranked_deep")
        return tuple(reasons)

    @property
    def is_hard(self) -> bool:
        return bool(self.hard_reasons)


def annotate_case(example: BirdExample, schema: DatabaseSchema) -> SelectorCase:
    if not example.gold_sql:
        raise ValueError(f"question {example.question_id} has no gold SQL")
    gold = oracle_table_names(schema, example.gold_sql)
    if not gold:
        raise ValueError(f"no schema table found in gold SQL of question {example.question_id}")
    ranking = [
        name.casefold()
        for name in lexical_table_ranking(schema, example.question, example.evidence)
    ]
    worst = max(ranking.index(name.casefold()) for name in gold) + 1
    return SelectorCase(
        question_id=example.question_id,
        db_id=example.db_id,
        gold_tables=gold,
        n_tables_total=len(schema.tables),
        linked_tables=linked_table_names(schema, example.question, example.evidence),
        lexical_worst_rank=worst,
    )


def select_training_cases(
    cases: Sequence[SelectorCase],
    *,
    cap_per_db: int,
    hard_fraction: float = 0.5,
    seed: int = 0,
) -> list[SelectorCase]:
    """Cap questions per database, filling the hard slots first.

    ``cap_per_db`` of 0 keeps everything. Otherwise each database contributes
    at most ``cap_per_db`` questions: up to ``ceil(cap * hard_fraction)`` hard
    ones, then random easy ones, then leftover hard ones if easy ran out.
    Seeing many databases matters more than seeing many questions per
    database, so the cap is per database rather than global.
    """
    if cap_per_db < 0:
        raise ValueError("cap_per_db must not be negative")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be within [0, 1]")
    by_db: dict[str, list[SelectorCase]] = defaultdict(list)
    for case in cases:
        by_db[case.db_id].append(case)

    rng = random.Random(seed)
    chosen: list[SelectorCase] = []
    for db_id in sorted(by_db):
        pool = sorted(by_db[db_id], key=lambda c: c.question_id)
        if cap_per_db == 0:
            chosen.extend(pool)
            continue
        hard = [c for c in pool if c.is_hard]
        easy = [c for c in pool if not c.is_hard]
        rng.shuffle(hard)
        rng.shuffle(easy)
        quota = math.ceil(cap_per_db * hard_fraction)
        picked = hard[:quota]
        picked += easy[: cap_per_db - len(picked)]
        picked += hard[quota : quota + cap_per_db - len(picked)]
        chosen.extend(sorted(picked, key=lambda c: c.question_id))
    return chosen


# --------------------------------------------------------------------------- records


def format_selector_target(
    tables: Sequence[str], columns: Sequence[str] | None = None
) -> str:
    """The assistant turn.

    ``tables`` only: a JSON list of table names in schema order. With
    ``columns`` (``table.column`` strings): a JSON object listing the columns
    first, then the tables.
    """
    if columns is None:
        return json.dumps(list(tables), ensure_ascii=False)
    return json.dumps({"columns": list(columns), "tables": list(tables)}, ensure_ascii=False)


_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_QUOTES = "\"'`"


def _known_tables(tables: DatabaseSchema | Sequence[str]) -> list[str]:
    return [t.name for t in tables.tables] if isinstance(tables, DatabaseSchema) else list(tables)


def parse_selector_output(
    text: str, tables: DatabaseSchema | Sequence[str]
) -> tuple[str, ...]:
    """Read table names back out of a model answer.

    ``tables`` is the schema or its table names in schema order. Accepts, in
    order of preference: a JSON object with ``tables`` and/or ``columns`` (a
    column's ``table.`` prefix counts as naming that table, so a table the
    model listed a column for is never lost); a JSON list; comma or newline
    separated names. Unknown names are dropped, matching is case-insensitive,
    and the result is returned in schema order so that two answers naming the
    same tables compare equal.
    """
    known = _known_tables(tables)
    names: list[str] = []
    obj = _OBJECT_RE.search(text)
    if obj:
        try:
            parsed = json.loads(obj.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            names = [str(item) for item in parsed.get("tables") or [] if item]
            names += [
                str(item).split(".", 1)[0]
                for item in parsed.get("columns") or []
                if item and "." in str(item)
            ]
    if not names:
        match = _LIST_RE.search(text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    names = [str(item) for item in parsed]
            except json.JSONDecodeError:
                names = []
        if not names:
            body = match.group(0).strip("[]") if match else text
            names = [part.strip().strip(_QUOTES) for part in re.split(r"[,\n]", body)]
    wanted = {name.casefold() for name in names if name}
    return tuple(name for name in known if name.casefold() in wanted)


def build_selector_messages(
    example: BirdExample, schema_text: str, target_format: str = "tables"
) -> list[dict[str, str]]:
    if target_format not in TARGET_FORMATS:
        raise ValueError(f"target_format must be one of {TARGET_FORMATS}, got {target_format!r}")
    parts = [f"Database schema:\n\n{schema_text}"]
    if example.evidence:
        parts.append(f"External knowledge: {example.evidence}")
    parts.append(f"Question: {example.question}")
    parts.append(SELECTOR_INSTRUCTIONS[target_format])
    return [
        {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_selector_record(
    example: BirdExample,
    schema: DatabaseSchema,
    schema_text: str,
    gold_tables: Sequence[str],
    *,
    target_format: str = "tables",
    gold_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    """One chat-format training example whose target parses back to its label."""
    if target_format == "columns":
        target = format_selector_target(gold_tables, list(gold_columns or ()))
    else:
        target = format_selector_target(gold_tables)
    if parse_selector_output(target, schema) != tuple(gold_tables):
        raise ValueError(f"target for question {example.question_id} does not round-trip")
    messages = build_selector_messages(example, schema_text, target_format)
    messages.append({"role": "assistant", "content": target})
    return {"question_id": example.question_id, "db_id": example.db_id, "messages": messages}


# --------------------------------------------------------------------------- gold columns

_SQL_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9$]*|\.|\(|\)|,|\*|[^\sA-Za-z_0-9$.(),*]+")
_NOT_AN_ALIAS = {
    "ON", "WHERE", "INNER", "LEFT", "RIGHT", "CROSS", "JOIN", "GROUP", "ORDER", "LIMIT",
    "SELECT", "HAVING", "AND", "OR", "UNION", "EXCEPT", "INTERSECT", "NATURAL", "USING",
}


def _sql_tokens(sql: str) -> list[tuple[str, str]]:
    """Lexical tokens of ``sql``: (kind, text) with kind in STR/IDENT/WORD/PUNCT."""
    out: list[tuple[str, str]] = []
    for segment in scan(sql):
        if segment.kind in (STRING, COMMENT):
            out.append(("STR", segment.text))
        elif segment.kind == IDENT:
            out.append(("IDENT", segment.text.strip('`"[]')))
        else:
            for match in _SQL_WORD_RE.finditer(segment.text):
                token = match.group(0)
                kind = "WORD" if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9$]*", token) else "PUNCT"
                out.append((kind, token))
    return out


def gold_columns(schema: DatabaseSchema, gold_sql: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``table.column`` references of ``gold_sql``, resolved through aliases.

    Returns ``(columns, unresolved)``. Qualified references (``T1.col``,
    ``table.col``) are mapped through the FROM/JOIN alias table; a bare column
    name is attributed to the single FROM/JOIN table that has it. Names that
    cannot be attributed (a bare column two used tables share, usually inside
    a subquery) land in ``unresolved`` rather than being guessed. Columns come
    back in schema order.
    """
    tokens = _sql_tokens(gold_sql)
    tables = {t.name.casefold(): t.name for t in schema.tables}
    columns = {
        t.name.casefold(): {c.name.casefold(): c.name for c in t.columns} for t in schema.tables
    }
    alias: dict[str, str] = {}
    for i, (kind, text) in enumerate(tokens):
        if kind == "WORD" and text.upper() in ("FROM", "JOIN") and i + 1 < len(tokens):
            kind2, name = tokens[i + 1]
            if kind2 in ("WORD", "IDENT") and name.casefold() in tables:
                table = tables[name.casefold()]
                alias[name.casefold()] = table
                j = i + 2
                if j < len(tokens) and tokens[j] == ("WORD", "AS"):
                    j += 1
                if (
                    j < len(tokens)
                    and tokens[j][0] in ("WORD", "IDENT")
                    and tokens[j][1].upper() not in _NOT_AN_ALIAS
                    and tokens[j][1].casefold() not in tables
                ):
                    alias[tokens[j][1].casefold()] = table
    used = set(alias.values())
    found: set[tuple[str, str]] = set()
    unresolved: list[str] = []
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        qualified = (
            kind in ("WORD", "IDENT")
            and i + 2 < len(tokens)
            and tokens[i + 1] == ("PUNCT", ".")
            and tokens[i + 2][0] in ("WORD", "IDENT", "PUNCT")
        )
        if qualified:
            table = alias.get(text.casefold()) or tables.get(text.casefold())
            column = tokens[i + 2][1]
            if table and column.casefold() in columns[table.casefold()]:
                found.add((table, columns[table.casefold()][column.casefold()]))
            elif column != "*":
                unresolved.append(f"{text}.{column}")
            i += 3
            continue
        after_dot = i > 0 and tokens[i - 1] == ("PUNCT", ".")
        if kind in ("WORD", "IDENT") and not after_dot:
            next_text = tokens[i + 1][1] if i + 1 < len(tokens) else ""
            bare = (
                next_text != "("
                and text.casefold() not in alias
                and text.casefold() not in tables
            )
            if bare:
                owners = [t for t in used if text.casefold() in columns[t.casefold()]]
                if len(owners) == 1:
                    found.add((owners[0], columns[owners[0].casefold()][text.casefold()]))
                elif len(owners) > 1:
                    unresolved.append(f"?{text}")
        i += 1
    ordered = [
        f"{t.name}.{c.name}"
        for t in schema.tables
        for c in t.columns
        if (t.name, c.name) in found
    ]
    return tuple(ordered), tuple(unresolved)


def trim_descriptions(schema: DatabaseSchema, max_chars: int) -> DatabaseSchema:
    """Keep only the column description, cut to ``max_chars``; drop value notes.

    BIRD value descriptions are long and often junk; for table selection the
    one-line column meaning is what helps, and the prompt has to stay short.
    """
    return DatabaseSchema(
        schema.db_id,
        tuple(
            Table(
                t.name,
                tuple(
                    Column(
                        c.name, c.type, c.notnull, c.primary_key,
                        ((c.description or "").strip()[:max_chars] or None) if max_chars else None,
                        None,
                    )
                    for c in t.columns
                ),
                t.foreign_keys,
                t.ddl,
            )
            for t in schema.tables
        ),
    )


# --------------------------------------------------------------------------- metrics


def selection_metrics(
    cases: Sequence[SelectorCase], predicted: Mapping[int, Sequence[str]]
) -> dict[str, object]:
    """Score a table selection against gold, per question.

    Reports the numbers that have to move together: how often every gold
    table survived, how much of what was kept is used, how much was kept, and
    per-table recall (kept only for comparison with the older diagnostics).
    """
    if not cases:
        return {
            "n": 0,
            "n_missing_predictions": 0,
            "all_gold_retained_rate": 0.0,
            "mean_precision": 0.0,
            "mean_table_recall": 0.0,
            "mean_selected_tables": 0.0,
            "mean_gold_tables": 0.0,
            "mean_total_tables": 0.0,
        }
    all_kept = 0
    precisions: list[float] = []
    recalls: list[float] = []
    selected: list[int] = []
    missing = 0
    for case in cases:
        gold = {name.casefold() for name in case.gold_tables}
        pred = {name.casefold() for name in predicted.get(case.question_id, ())}
        if case.question_id not in predicted:
            missing += 1
        hit = len(gold & pred)
        all_kept += gold <= pred
        precisions.append(hit / len(pred) if pred else 0.0)
        recalls.append(hit / len(gold))
        selected.append(len(pred))
    n = len(cases)
    return {
        "n": n,
        "n_missing_predictions": missing,
        "all_gold_retained_rate": round(all_kept / n, 4),
        "mean_precision": round(sum(precisions) / n, 4),
        "mean_table_recall": round(sum(recalls) / n, 4),
        "mean_selected_tables": round(sum(selected) / n, 2),
        "mean_gold_tables": round(sum(len(c.gold_tables) for c in cases) / n, 2),
        "mean_total_tables": round(sum(c.n_tables_total for c in cases) / n, 2),
    }


def selection_report(
    cases: Sequence[SelectorCase],
    predicted: Mapping[int, Sequence[str]],
    *,
    baseline: Mapping[int, Sequence[str]] | None = None,
    expanded: Mapping[int, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Overall, per-database and hard/easy metrics, next to a baseline selection.

    ``expanded`` is the model's prediction after :func:`expand_selection`; it
    is reported as a third column so the raw model and the operating point
    actually handed to the SQL generator stay distinguishable.
    """
    by_db: dict[str, list[SelectorCase]] = defaultdict(list)
    for case in cases:
        by_db[case.db_id].append(case)
    hard = [c for c in cases if c.is_hard]
    easy = [c for c in cases if not c.is_hard]

    def block(subset: Sequence[SelectorCase]) -> dict[str, object]:
        out: dict[str, object] = {"model": selection_metrics(subset, predicted)}
        if expanded is not None:
            out["expanded"] = selection_metrics(subset, expanded)
        if baseline is not None:
            out["baseline"] = selection_metrics(subset, baseline)
        return out

    return {
        "overall": block(cases),
        "hard": block(hard),
        "easy": block(easy),
        "per_db": {db_id: block(subset) for db_id, subset in sorted(by_db.items())},
        "n_empty_predictions": sum(
            1 for c in cases if not tuple(predicted.get(c.question_id, ()))
        ),
    }


def load_selected_tables(
    path: str | Path, field: str = "predicted_tables"
) -> dict[int, tuple[str, ...]]:
    """Read ``question_id -> tables`` from an evaluate_selector output.

    ``field`` is ``predicted_tables`` for the raw model answer or
    ``expanded_tables`` for the recall-oriented expansion.
    """
    selected: dict[int, tuple[str, ...]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if field not in record:
            raise KeyError(f"{path} has no field {field!r} (question {record.get('question_id')})")
        selected[int(record["question_id"])] = tuple(record[field])
    return selected


def expand_selection(
    schema: DatabaseSchema,
    predicted: Sequence[str],
    *,
    question: str = "",
    evidence: str = "",
    fk_hops: int = 1,
    lex_top_k: int = 0,
) -> tuple[str, ...]:
    """Recall-oriented expansion of a predicted table set.

    A missed gold table kills the question while an extra table costs little,
    and on the first selector most misses were bridge tables one foreign-key
    hop from a table the model did pick. So: add every table within
    ``fk_hops`` foreign-key hops of the prediction, then the ``lex_top_k``
    best lexical matches. Returned in schema order.
    """
    graph = foreign_key_graph(schema)
    canonical = {name.casefold(): name for name in graph}
    chosen = {canonical[p.casefold()] for p in predicted if p.casefold() in canonical}
    frontier = set(chosen)
    for _ in range(max(fk_hops, 0)):
        frontier = {nb for table in frontier for nb in graph[table]} - chosen
        chosen |= frontier
    if lex_top_k > 0:
        chosen |= set(lexical_table_ranking(schema, question, evidence)[:lex_top_k])
    return tuple(table.name for table in schema.tables if table.name in chosen)
