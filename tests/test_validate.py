"""Extraction and read-only validation, including the adversarial cases."""

from __future__ import annotations

import pytest

from text2sql_rlvr.sql.validate import extract_sql, validate_read_only


def test_extract_from_fenced_block():
    text = "Here you go:\n```sql\nSELECT 1\n```\n"
    assert extract_sql(text) == "SELECT 1"


def test_extract_takes_the_last_fence():
    text = "Maybe:\n```sql\nSELECT 1\n```\nActually:\n```sql\nSELECT 2\n```"
    assert extract_sql(text) == "SELECT 2"


def test_extract_drops_thinking_block():
    text = "<think>\n```sql\nSELECT 999\n```\n</think>\n```sql\nSELECT 1\n```"
    assert extract_sql(text) == "SELECT 1"


def test_extract_handles_truncated_fence():
    assert extract_sql("```sql\nSELECT 1 FROM t") == "SELECT 1 FROM t"


def test_extract_falls_back_to_bare_select():
    assert extract_sql("The answer is\nSELECT a FROM t;") == "SELECT a FROM t"


def test_extract_returns_empty_when_absent():
    assert extract_sql("I cannot answer that.") == ""
    assert extract_sql("") == ""


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select a from t where b = 'x'",
        "WITH c AS (SELECT 1 AS x) SELECT x FROM c",
        "VALUES (1), (2)",
        "SELECT replace(name, 'a', 'b') FROM t",
        "SELECT CASE WHEN a > 1 THEN 'y' ELSE 'n' END FROM t",
        "SELECT `drop table`, [insert into] FROM t",
        "SELECT 'DROP TABLE t' AS warning",
    ],
)
def test_accepts_read_only_queries(sql):
    assert validate_read_only(sql).ok, sql


@pytest.mark.parametrize(
    "sql,fragment",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("SELECT 1; DROP TABLE t", "found 2"),
        ("DROP TABLE t", "starts with DROP"),
        ("PRAGMA table_info(t)", "starts with PRAGMA"),
        ("INSERT INTO t VALUES (1)", "starts with INSERT"),
        ("REPLACE INTO t VALUES (1)", "starts with REPLACE"),
        ("EXPLAIN SELECT 1", "starts with EXPLAIN"),
        ("WITH c AS (SELECT 1) DELETE FROM t", "forbidden keyword DELETE"),
        ("SELECT load_extension('evil.so')", "forbidden keyword LOAD_EXTENSION"),
        ("SELECT 1; SELECT 2", "found 2"),
    ],
)
def test_rejects_unsafe_queries(sql, fragment):
    result = validate_read_only(sql)
    assert not result.ok
    assert fragment in (result.reason or "")


def test_statement_is_normalised_on_success():
    result = validate_read_only("  SELECT 1 ;  ")
    assert result.ok
    assert result.statement == "SELECT 1"
