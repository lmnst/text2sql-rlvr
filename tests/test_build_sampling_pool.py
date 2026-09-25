"""build_sampling_pool leaves out SFT-trained and gold-empty questions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def pool_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_sampling_pool_script", SCRIPTS / "build_sampling_pool.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_excludes_trained_and_gold_empty_questions(pool_module, tmp_path):
    questions = [
        {"question_id": i, "db_id": "a" if i < 4 else "b", "question": f"q{i}", "SQL": "SELECT 1"}
        for i in range(8)
    ]
    outcomes = [
        {"question_id": i, "db_id": q["db_id"], "official": i % 2, "gold_empty": i == 5}
        for i, q in enumerate(questions)
    ]
    (tmp_path / "q.json").write_text(json.dumps(questions), encoding="utf-8")
    write_jsonl(tmp_path / "o.jsonl", outcomes)
    write_jsonl(tmp_path / "sft.jsonl", [{"question_id": 0}, {"question_id": 6}])

    pool_module.main([
        "--questions", str(tmp_path / "q.json"),
        "--outcomes", str(tmp_path / "o.jsonl"),
        "--exclude", str(tmp_path / "sft.jsonl"),
        "--size", "0",
        "--out", str(tmp_path / "pool.json"),
        "--manifest", str(tmp_path / "pool_manifest.json"),
    ])

    selected = json.loads((tmp_path / "pool.json").read_text(encoding="utf-8"))
    assert sorted(q["question_id"] for q in selected) == [1, 2, 3, 4, 7]
    assert selected[0].keys() == questions[0].keys()

    manifest = json.loads((tmp_path / "pool_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_excluded"] == 2
    assert manifest["n_selected"] == 5
    assert manifest["questions_per_db"] == {"a": 3, "b": 2}


def test_size_is_balanced_over_databases(pool_module, tmp_path):
    questions = [
        {"question_id": i, "db_id": "big" if i < 10 else "small", "question": "", "SQL": ""}
        for i in range(12)
    ]
    outcomes = [
        {"question_id": q["question_id"], "db_id": q["db_id"], "official": 0, "gold_empty": False}
        for q in questions
    ]
    (tmp_path / "q.json").write_text(json.dumps(questions), encoding="utf-8")
    write_jsonl(tmp_path / "o.jsonl", outcomes)

    pool_module.main([
        "--questions", str(tmp_path / "q.json"),
        "--outcomes", str(tmp_path / "o.jsonl"),
        "--size", "4",
        "--out", str(tmp_path / "pool.json"),
        "--manifest", str(tmp_path / "m.json"),
    ])

    manifest = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert manifest["questions_per_db"] == {"big": 2, "small": 2}
