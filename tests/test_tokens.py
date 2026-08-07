"""The scanner exists to survive SQL that defeats regexes. Test exactly that."""

from __future__ import annotations

import pytest

from text2sql_rlvr.sql.tokens import (
    has_top_level_order_by,
    leading_keyword,
    normalize_whitespace,
    scan,
    split_statements,
    strip_comments,
)


def test_scan_is_lossless():
    sql = "SELECT `a b`, 'x''y' /* c */ FROM t -- tail\n"
    assert "".join(seg.text for seg in scan(sql)) == sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'a;b' FROM t",
        "SELECT `we;ird` FROM t",
        'SELECT "col;name" FROM t',
        "SELECT [odd;name] FROM t",
    ],
)
def test_semicolon_inside_quotes_does_not_split(sql):
    assert split_statements(sql) == [sql]


def test_real_semicolon_splits():
    assert len(split_statements("SELECT 1; SELECT 2")) == 2


def test_trailing_semicolon_yields_one_statement():
    assert split_statements("SELECT 1;") == ["SELECT 1"]


def test_comment_only_input_has_no_statements():
    assert split_statements("-- nothing here\n/* nor here */") == []


def test_double_dash_inside_string_is_not_a_comment():
    sql = "SELECT 'a--b' FROM t"
    assert strip_comments(sql) == sql


def test_strip_comments_does_not_fuse_tokens():
    assert normalize_whitespace(strip_comments("SELECT/*x*/1")) == "SELECT 1"


def test_unterminated_block_comment_is_consumed():
    assert strip_comments("SELECT 1 /* dangling").strip() == "SELECT 1"


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT a FROM t ORDER BY a", True),
        ("SELECT a FROM t", False),
        ("SELECT a FROM (SELECT a FROM t ORDER BY a) x", False),
        ("WITH c AS (SELECT a FROM t ORDER BY a) SELECT a FROM c", False),
        ("SELECT row_number() OVER (ORDER BY a) FROM t", False),
        ("SELECT group_concat(a ORDER BY a) FROM t", False),
        ("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1", True),
        ("SELECT 'ORDER BY x' FROM t", False),
        ("SELECT a FROM (SELECT a FROM t ORDER BY a) x ORDER BY a", True),
    ],
)
def test_top_level_order_by(sql, expected):
    assert has_top_level_order_by(sql) is expected


def test_leading_keyword_skips_comments_and_whitespace():
    assert leading_keyword("  -- pick\n  select 1") == "SELECT"
    assert leading_keyword("/* c */ WITH x AS (SELECT 1) SELECT * FROM x") == "WITH"
    assert leading_keyword("") == ""
