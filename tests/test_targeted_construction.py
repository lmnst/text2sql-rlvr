"""Audit gates must reject plausible but incorrect targeted data."""

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from text2sql_rlvr.rewards.sandbox import SqlExecutor


@pytest.fixture
def builder():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("targeted_builder", scripts /
                                                      "build_targeted_sft.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "sample.sqlite"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE item(group_id INTEGER, amount REAL)")
        c.executemany("INSERT INTO item VALUES (?,?)", [(1, 2), (1, 4), (2, 10), (3, None)])
    return path


def make_pair(sql_a, ref_a, sql_b, ref_b):
    return {"pair_id": "test", "family": "test", "db_id": "sample",
            "witness_sql": "SELECT group_id,amount FROM item",
            "variants": [{"variant": "a", "sql": sql_a, "reference": ref_a},
                         {"variant": "b", "sql": sql_b, "reference": ref_b}]}


def test_rejects_aggregate_at_wrong_grain(builder, db):
    pair = make_pair(
        "SELECT COUNT(*) FROM item WHERE group_id>1",
        {"kind": "group_counts", "key": 0, "threshold": 1, "mode": "count_groups"},
        "SELECT group_id FROM item GROUP BY group_id HAVING COUNT(*)>1",
        {"kind": "group_counts", "key": 0, "threshold": 1, "mode": "keys"},
    )
    with SqlExecutor() as executor:
        result = builder.verify_pair(pair, executor, db)
    assert result["variants"][0]["result"]["status"] == "ok"
    assert not result["accepted"]
    assert not result["variants"][0]["python_reference_match"]


def test_duplicate_difference_survives_official_set_equality(builder, db):
    pair = make_pair(
        "SELECT group_id FROM item", {"kind": "project", "columns": [0]},
        "SELECT DISTINCT group_id FROM item",
        {"kind": "project", "columns": [0], "distinct": True},
    )
    with SqlExecutor() as executor:
        result = builder.verify_pair(pair, executor, db)
    assert result["accepted"]
    assert result["counterpart_sql_rejected_by_strict_result"]
    assert not result["counterpart_sql_rejected_by_set_result"]


def test_capped_witness_cannot_approve_dataset(builder, db):
    pair = make_pair("SELECT group_id FROM item", {"kind": "project", "columns": [0]},
                     "SELECT amount FROM item", {"kind": "project", "columns": [1]})
    with SqlExecutor(max_rows=1) as executor:
        result = builder.verify_pair(pair, executor, db)
    assert "witness_unavailable_or_empty_or_capped" in result["failures"]


def test_equivalent_pair_is_not_discriminating(builder, db):
    pair = make_pair("SELECT group_id FROM item", {"kind": "project", "columns": [0]},
                     "SELECT group_id FROM item ORDER BY group_id",
                     {"kind": "project", "columns": [0]})
    with SqlExecutor() as executor:
        result = builder.verify_pair(pair, executor, db)
    assert "pair_answers_not_distinguishable" in result["failures"]


def test_mean_of_means_is_not_row_weighted_mean(builder):
    rows = [(1, 2), (1, 4), (2, 10), (3, None)]
    spec = {"kind": "mean", "key": 0, "value": 1, "mode": "group_means"}
    assert builder.reference_rows(rows, spec) == [(6.5,)]
    assert builder.reference_rows(rows, {**spec, "mode": "rows"}) == [(16 / 3,)]


def test_reference_concatenation_preserves_null(builder):
    spec = {"kind": "project", "columns": [{"concat": [0, 1], "separator": " "}]}
    assert builder.reference_rows([("A", None), ("A", "B")], spec) == [(None,), ("A B",)]


def test_validation_database_is_rejected_even_with_train_id(builder):
    cfg = {"pairs": [{"pair_id": "x", "db_id": "heldout", "reviewer": "codex",
                      "focus": "test", "source_question_ids": [1]}]}
    split = {"train_ids": [1], "val_ids": [2], "val_db_ids": ["heldout"]}
    with pytest.raises(ValueError, match="Validation database"):
        builder.validate_sources(cfg, [{"question_id": 1, "db_id": "heldout"}], split)
