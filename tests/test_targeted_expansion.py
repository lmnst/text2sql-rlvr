"""Expansion preserves literal values and verifies replay identity/range semantics."""

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from text2sql_rlvr.rewards.sandbox import SqlExecutor
from text2sql_rlvr.sql import extract_sql


def load_script(name):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_grounded_apostrophe_is_sql_escaped_but_question_keeps_name():
    expand = load_script("expand_targeted_sft")
    template = {"template_id": "t", "slot_names": ["name"],
                "question": "Find {name}.", "sql": "SELECT x FROM t WHERE name={sql_name}"}
    p = expand.instantiate(template, ["O'Brien"])
    assert p["question"] == "Find O'Brien."
    assert p["sql"] == "SELECT x FROM t WHERE name='O''Brien'"


def test_literal_substitution_does_not_change_identifier_or_source_ids():
    expand = load_script("expand_targeted_sft")
    pair = {"source_question_ids": [1], "sql": "SELECT Name FROM t WHERE Name='Name'",
            "question": "Find Name", "reference": {"filter": [[0, "eq", "Name"]]}}
    result = expand.change_literal(pair, "Name", "O'Brien")
    assert result["sql"] == "SELECT Name FROM t WHERE Name='O''Brien'"
    assert result["source_question_ids"] == [1]
    assert result["reference"]["filter"][0][-1] == "O'Brien"


def test_replay_export_preserves_spaces_inside_sql_literal():
    builder = load_script("build_targeted_replay_v2")
    q = {"question_id": 1, "db_id": "x", "question": "Find a name.",
         "SQL": "SELECT name FROM t WHERE name='A  B'", "component": "replay"}
    chat = builder.make_chat(q, "CREATE TABLE t(name TEXT)")
    assert extract_sql(chat["messages"][-1]["content"]) == q["SQL"]


@pytest.mark.parametrize("start", [0, 1, 10])
def test_first_customer_batch_is_not_assumed_one_based(tmp_path, start):
    builder = load_script("build_targeted_replay_v2")
    path = tmp_path / "batch.sqlite"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE Customers(ID INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE Mailings1_2(REFID INTEGER PRIMARY KEY)")
        c.executemany("INSERT INTO Customers VALUES (?)", [(i,) for i in range(start, start + 5)])
        c.executemany("INSERT INTO Mailings1_2 VALUES (?)", [(i,) for i in range(start, start + 3)])
    with SqlExecutor() as executor:
        assert builder.mailing_scope(executor, path, n=3)[0]


def test_wrong_customer_batch_is_rejected_even_if_size_matches(tmp_path):
    builder = load_script("build_targeted_replay_v2")
    path = tmp_path / "wrong.sqlite"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE Customers(ID INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE Mailings1_2(REFID INTEGER PRIMARY KEY)")
        c.executemany("INSERT INTO Customers VALUES (?)", [(0,), (1,), (2,), (3,)])
        c.executemany("INSERT INTO Mailings1_2 VALUES (?)", [(1,), (2,), (3,)])
    with SqlExecutor() as executor:
        assert not builder.mailing_scope(executor, path, n=3)[0]


def test_sql_deduplication_does_not_fold_case_of_literals():
    builder = load_script("build_targeted_replay_v2")
    assert builder.sql_key("SELECT x FROM t WHERE x='a'") == builder.sql_key(
        "select X from T where X = 'a'")
    assert builder.sql_key("SELECT x FROM t WHERE x='a'") != builder.sql_key(
        "SELECT x FROM t WHERE x='A'")


def test_replay_selection_is_deterministic_and_covers_databases():
    builder = load_script("build_targeted_replay_v2")
    rows = [{"question_id": i, "db_id": "a" if i < 8 else "b"} for i in range(10)]
    first = builder.choose_replay(rows, 4, 0)
    assert first == builder.choose_replay(list(reversed(rows)), 4, 0)
    assert len({r["question_id"] for r in first}) == 4
    assert {r["db_id"] for r in first} == {"a", "b"}
