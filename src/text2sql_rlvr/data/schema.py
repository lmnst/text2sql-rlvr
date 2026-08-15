"""Read a database's structure and render it for a prompt.

Schema comes from the SQLite file itself rather than from BIRD's ``*_tables.json``:
the file is the thing the query will actually run against, and it also carries the
original ``CREATE TABLE`` text, which is closer to what the model saw in pretraining.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from text2sql_rlvr.rewards.sandbox import open_read_only

#: Encodings tried in order for BIRD's column-description CSVs, which are a mix
#: of UTF-8 and Windows codepages depending on the database.
_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

SCHEMA_STYLES = ("ddl", "compact")


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    notnull: bool = False
    primary_key: bool = False
    description: str | None = None
    value_description: str | None = None


@dataclass(frozen=True)
class ForeignKey:
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    ddl: str = ""


@dataclass(frozen=True)
class DatabaseSchema:
    db_id: str
    tables: tuple[Table, ...]

    def table(self, name: str) -> Table | None:
        lowered = name.lower()
        for table in self.tables:
            if table.name.lower() == lowered:
                return table
        return None


def quote_identifier(name: str) -> str:
    """Quote an identifier for interpolation into SQL we generate ourselves."""
    return '"' + name.replace('"', '""') + '"'


def _load_descriptions(directory: Path, table_name: str) -> dict[str, tuple[str, str]]:
    """Return ``{column_lower: (description, value_description)}`` for one table."""
    target = f"{table_name.lower()}.csv"
    path = next((p for p in directory.glob("*.csv") if p.name.lower() == target), None)
    if path is None:
        return {}

    for encoding in _CSV_ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
        out: dict[str, tuple[str, str]] = {}
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            clean = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            column = clean.get("original_column_name") or clean.get("column_name")
            if not column:
                continue
            out[column.lower()] = (
                clean.get("column_description", ""),
                clean.get("value_description", ""),
            )
        return out
    return {}


def load_schema(
    db_path: str | Path,
    *,
    db_id: str | None = None,
    descriptions_dir: str | Path | None = None,
) -> DatabaseSchema:
    """Introspect ``db_path``. ``descriptions_dir`` is BIRD's ``database_description``."""
    path = Path(db_path)
    descriptions = Path(descriptions_dir) if descriptions_dir else path.parent / "database_description"
    have_descriptions = descriptions.is_dir()

    conn = open_read_only(path)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        tables: list[Table] = []
        for table_name, ddl in rows:
            described = _load_descriptions(descriptions, table_name) if have_descriptions else {}
            quoted = quote_identifier(table_name)

            columns = []
            for _cid, col_name, col_type, notnull, _default, pk in conn.execute(
                f"PRAGMA table_info({quoted})"
            ):
                description, value_description = described.get(str(col_name).lower(), ("", ""))
                columns.append(
                    Column(
                        name=str(col_name),
                        type=str(col_type or ""),
                        notnull=bool(notnull),
                        primary_key=bool(pk),
                        description=description or None,
                        value_description=value_description or None,
                    )
                )

            foreign_keys = tuple(
                ForeignKey(from_column=str(row[3]), to_table=str(row[2]), to_column=str(row[4]))
                for row in conn.execute(f"PRAGMA foreign_key_list({quoted})")
                if row[4] is not None
            )
            tables.append(
                Table(
                    name=str(table_name),
                    columns=tuple(columns),
                    foreign_keys=foreign_keys,
                    ddl=str(ddl or "").strip(),
                )
            )
    finally:
        conn.close()

    return DatabaseSchema(db_id=db_id or path.stem, tables=tuple(tables))


def fetch_sample_rows(
    db_path: str | Path, schema: DatabaseSchema, n: int = 3
) -> dict[str, list[tuple[Any, ...]]]:
    """Read the first ``n`` rows of each table.

    Sample values teach the model how values are actually formatted ('CA' vs
    'California'), which is a large share of BIRD errors -- at a real cost in
    prompt tokens. Off by default; turn it on as an ablation.
    """
    if n <= 0:
        return {}
    conn = open_read_only(db_path)
    samples: dict[str, list[tuple[Any, ...]]] = {}
    try:
        for table in schema.tables:
            try:
                cursor = conn.execute(f"SELECT * FROM {quote_identifier(table.name)} LIMIT {int(n)}")
                samples[table.name] = [tuple(row) for row in cursor.fetchall()]
            except Exception:  # noqa: BLE001 - a virtual or corrupt table is not fatal
                samples[table.name] = []
    finally:
        conn.close()
    return samples


def _column_note(column: Column) -> str:
    parts = [p for p in (column.description, column.value_description) if p]
    return "; ".join(parts)


def _format_ddl(schema: DatabaseSchema, include_descriptions: bool) -> Iterable[str]:
    for table in schema.tables:
        yield table.ddl if table.ddl else _synthesise_ddl(table)
        if include_descriptions:
            notes = [f"--   {c.name}: {_column_note(c)}" for c in table.columns if _column_note(c)]
            if notes:
                yield f"-- {table.name} columns:\n" + "\n".join(notes)


def _synthesise_ddl(table: Table) -> str:
    """Fallback for tables created without stored DDL (views, odd releases)."""
    body = ",\n".join(
        f"  {quote_identifier(c.name)} {c.type}".rstrip()
        + (" PRIMARY KEY" if c.primary_key else "")
        + (" NOT NULL" if c.notnull and not c.primary_key else "")
        for c in table.columns
    )
    fks = "".join(
        f",\n  FOREIGN KEY ({quote_identifier(fk.from_column)}) "
        f"REFERENCES {quote_identifier(fk.to_table)}({quote_identifier(fk.to_column)})"
        for fk in table.foreign_keys
    )
    return f"CREATE TABLE {quote_identifier(table.name)} (\n{body}{fks}\n);"


def _format_compact(schema: DatabaseSchema, include_descriptions: bool) -> Iterable[str]:
    for table in schema.tables:
        lines = [f"# Table: {table.name}"]
        for column in table.columns:
            flags = []
            if column.primary_key:
                flags.append("primary key")
            if column.notnull:
                flags.append("not null")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            note = _column_note(column) if include_descriptions else ""
            note = f" -- {note}" if note else ""
            lines.append(f"#   {column.name} ({column.type or 'ANY'}){suffix}{note}")
        for fk in table.foreign_keys:
            lines.append(f"#   {fk.from_column} -> {fk.to_table}.{fk.to_column}")
        yield "\n".join(lines)


def format_schema(
    schema: DatabaseSchema,
    *,
    style: str = "ddl",
    include_descriptions: bool = False,
    sample_rows: dict[str, list[tuple[Any, ...]]] | None = None,
) -> str:
    """Render ``schema`` as prompt text."""
    if style not in SCHEMA_STYLES:
        raise ValueError(f"style must be one of {SCHEMA_STYLES}, got {style!r}")

    blocks = list(
        _format_ddl(schema, include_descriptions)
        if style == "ddl"
        else _format_compact(schema, include_descriptions)
    )

    if sample_rows:
        for table in schema.tables:
            rows = sample_rows.get(table.name) or []
            if not rows:
                continue
            header = " | ".join(c.name for c in table.columns)
            body = "\n".join(" | ".join("NULL" if v is None else str(v) for v in row) for row in rows)
            blocks.append(f"-- {len(rows)} example rows from {table.name}:\n-- {header}\n-- " +
                          body.replace("\n", "\n-- "))

    return "\n\n".join(blocks)
