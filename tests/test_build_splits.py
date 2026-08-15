"""Regression test for the question_id stamping in build_splits.write_questions.

BIRD's train.json has no ``question_id`` field, so :func:`load_examples` falls
back to the list index. ``write_questions`` must stamp the original id back in,
otherwise filtering/rearranging silently turns the new index into a fake
question_id that no longer traces to the source split.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def build_splits_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_splits_script", SCRIPTS / "build_splits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_question_id_is_stamped_from_the_tuple(build_splits_module, tmp_path):
    out = tmp_path / "questions.json"
    records = [
        (5, {"db_id": "a", "question": "one"}),
        (10, {"db_id": "b", "question": "two", "SQL": "SELECT 1"}),
    ]
    build_splits_module.write_questions(out, records)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert [r["question_id"] for r in written] == [5, 10]
    assert written[0]["db_id"] == "a"
    assert written[1]["SQL"] == "SELECT 1"


def test_stamping_does_not_mutate_the_caller_dict(build_splits_module, tmp_path):
    out = tmp_path / "questions.json"
    item = {"db_id": "a", "question": "one"}
    build_splits_module.write_questions(out, [(7, item)])

    assert "question_id" not in item


def test_existing_question_id_is_overwritten_by_the_tuple(build_splits_module, tmp_path):
    """The tuple's id is authoritative; a stale value in the record must not win."""
    out = tmp_path / "questions.json"
    build_splits_module.write_questions(out, [(3, {"question_id": 999, "db_id": "a"})])

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written[0]["question_id"] == 3


def test_stamped_files_reload_to_the_same_ids(build_splits_module, tmp_path):
    """The stamped file must round-trip through load_examples with the original ids."""
    from text2sql_rlvr.data.bird import load_examples

    out = tmp_path / "questions.json"
    build_splits_module.write_questions(
        out,
        [(2, {"db_id": "a", "question": "q"}), (9, {"db_id": "b", "question": "q2"})],
    )

    examples = load_examples(out)
    assert [e.question_id for e in examples] == [2, 9]
