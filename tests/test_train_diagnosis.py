"""Paired diagnosis must not mix populations, prompts or incomplete results."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from text2sql_rlvr.eval.train_diagnosis import (
    choose_subset,
    paired_summary,
    transition,
    verify_predictions,
)


def population():
    rows = [{"question_id": i, "db_id": "big" if i < 20 else "small"} for i in range(22)]
    manifest = {"train_ids": list(range(22)), "val_ids": [100], "val_db_ids": ["val_db"]}
    return rows, manifest


def test_subset_covers_databases_and_is_independent_of_file_order():
    rows, manifest = population()
    selected = choose_subset(rows, manifest, 10, 0)
    assert selected == choose_subset(list(reversed(rows)), manifest, 10, 0)
    assert len(selected) == len({r["question_id"] for r in selected}) == 10
    assert {r["db_id"] for r in selected} == {"big", "small"}
    assert choose_subset(rows, manifest, 22, 0) == rows


@pytest.mark.parametrize("mutation", ["database", "question", "incomplete"])
def test_subset_refuses_leakage_and_wrong_population(mutation):
    rows, manifest = population()
    if mutation == "database":
        rows[0]["db_id"] = "val_db"
    elif mutation == "question":
        manifest["val_ids"] = [0]
    else:
        rows.pop()
    with pytest.raises(ValueError):
        choose_subset(rows, manifest, 10, 0)


def test_degenerate_population_one_per_database():
    rows = [{"question_id": i, "db_id": str(i)} for i in range(3)]
    manifest = {"train_ids": [0, 1, 2], "val_ids": [], "val_db_ids": []}
    assert choose_subset(rows, manifest, 3, 0) == rows


@pytest.mark.parametrize("change", ["missing", "prompt", "transport"])
def test_inference_must_be_complete_and_identical_prompt(change):
    prompts = [{"question_id": 1, "db_id": "db", "prompt_sha256": "a"}]
    rows = [{**prompts[0], "completion": "SELECT 1", "error": None}]
    if change == "missing":
        rows = []
    elif change == "prompt":
        rows[0]["prompt_sha256"] = "b"
    else:
        rows[0]["error"] = "connection failed"
    with pytest.raises(ValueError):
        verify_predictions(rows, prompts, "base")


def verdict(correct, comparable=True):
    return {"official": correct, "strict": correct, "comparable": comparable,
            "bucket": "correct_both" if correct else "missing_column"}


def test_shared_denominator_and_direction_of_transitions():
    pairs = []
    for base, sft in [(False, True), (True, False), (False, False), (True, True)]:
        a, b = verdict(base), verdict(sft)
        pairs.append({"base": a, "sft": b, "transition": transition(a, b)})
    a, b = verdict(True), verdict(False, comparable=False)
    pairs.append({"base": a, "sft": b, "transition": transition(a, b)})
    summary = paired_summary(pairs)
    assert summary["n_pair_comparable"] == 4
    assert summary["n_pair_unscorable"] == 1
    assert summary["arms"]["base"]["official_ex_comparable"] == 50
    assert summary["arms"]["sft"]["official_ex_comparable"] == 50
    assert summary["transitions"] == {
        "both_correct": 1, "persistent_failure": 1, "regression": 1, "resolved": 1, "unscorable": 1,
    }
    assert summary["persistent_symptom_transitions"] == {"missing_column -> missing_column": 1}


def test_no_comparable_examples_is_not_zero_accuracy():
    a, b = verdict(True), verdict(False, False)
    summary = paired_summary([{"base": a, "sft": b, "transition": transition(a, b)}])
    assert summary["arms"]["base"]["official_ex_comparable"] is None


def test_sql_pair_marks_capped_results_unscorable(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("diagnosis_script", scripts /
                                                      "train_failure_diagnosis.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    from text2sql_rlvr.rewards.sandbox import ExecResult

    class Executor:
        def execute(self, _db, sql):
            if sql == "SELECT 2":
                return ExecResult("ok", rows=((2,),), columns=("x",), truncated=True)
            return ExecResult("ok", rows=((1,),), columns=("x",))

    pair = module.evaluate_pair(
        {"question_id": 1, "db_id": "db", "question": "q", "SQL": "SELECT 1"},
        {"completion": "SELECT 1", "finish_reason": "stop"},
        {"completion": "SELECT 2", "finish_reason": "stop"}, Executor(), tmp_path,
    )
    assert pair["transition"] == "unscorable"
    assert pair["base"]["official"] is True
    assert pair["sft"]["pred_truncated"] is True
