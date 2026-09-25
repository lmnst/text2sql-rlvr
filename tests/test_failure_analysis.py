"""Misjoining historical ids would invalidate the entire semantic audit."""

import pytest

from text2sql_rlvr.eval.failure_analysis import (
    align_records,
    stratified_sample,
    symptom,
    training_seed_catalog,
)


def records():
    q = [{"question_id": 1359, "db_id": "db", "question": "height?", "SQL": "SELECT x"}]
    o = [{"question_id": 0, "db_id": "db", "gold_sql": "SELECT x", "pred_sql": "SELECT y",
          "gold_status": "ok", "pred_status": "error", "pred_error": "no such column: y",
          "official": False, "strict": False}]
    p = [{"question_id": 0, "db_id": "db", "sql": "SELECT y", "completion": "SELECT y"}]
    return q, o, p


def test_old_position_is_not_original_id():
    q, o, p = records()
    row = align_records(q, o, p, id_mode="position")[0]
    assert (row["question_id"], row["historical_question_id"]) == (1359, 0)
    with pytest.raises(ValueError, match="exact full"):
        align_records(q, o, p, id_mode="original")


@pytest.mark.parametrize("field,value", [("db_id", "other"), ("gold_sql", "SELECT z")])
def test_reordered_or_changed_questions_are_rejected(field, value):
    q, o, p = records()
    o[0][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        align_records(q, o, p, id_mode="position")


def test_prediction_content_and_duplicate_ids_are_checked():
    q, o, p = records()
    p[0]["sql"] = "SELECT z"
    with pytest.raises(ValueError, match="prediction SQL"):
        align_records(q, o, p, id_mode="position")
    with pytest.raises(ValueError, match="duplicate"):
        align_records(q, o + o, p, id_mode="position")


def test_incomplete_archive_cannot_masquerade_as_full_split():
    q, o, _p = records()
    with pytest.raises(ValueError, match="exact full"):
        align_records(q, o, [], id_mode="position")


def test_official_only_is_not_an_official_failure():
    row = {"gold_status": "ok", "official": True, "strict": False, "reason": "row_count"}
    assert symptom(row) == "official_only"


def test_column_count_does_not_infer_missing_join():
    row = {"gold_status": "ok", "pred_status": "ok", "official": False,
           "strict": False, "reason": "column_count"}
    assert symptom(row) == "result_column_count"


def test_sampling_is_repeatable_order_independent_and_covers_small_strata():
    rows = [{"question_id": i, "db_id": "a", "bucket": "common"} for i in range(20)]
    rows.append({"question_id": 99, "db_id": "b", "bucket": "rare"})
    sample = stratified_sample(rows, 4, 0)
    assert sample == stratified_sample(list(reversed(rows)), 4, 0)
    assert len({r["question_id"] for r in sample}) == 4
    assert 99 in {r["question_id"] for r in sample}
    assert len(stratified_sample(rows, 100, 0)) == 21


@pytest.mark.parametrize("qid,db", [(9, "train_db"), (1, "val_db")])
def test_training_seed_selection_rejects_id_or_database_leakage(qid, db):
    train = [{"question_id": qid, "db_id": db, "SQL": "SELECT COUNT(*) FROM x"}]
    val = [{"question_id": 9, "db_id": "val_db"}]
    with pytest.raises(ValueError, match="leakage"):
        training_seed_catalog(train, val, {qid}, per_family=1, seed=0)


def test_training_candidates_are_not_approved_or_synthetic():
    train = [{"question_id": 1, "db_id": "train_db", "SQL": "SELECT COUNT(*) FROM x"}]
    candidates = training_seed_catalog(train, [], {1}, per_family=1, seed=0)
    assert len(candidates) == 1
    assert candidates[0]["target_family"] == "aggregation_grain"
    assert candidates[0]["approved_for_training"] is False
    assert candidates[0]["synthetic"] is False
