"""SFT example construction.

The two properties that matter are both about *agreement with something else*:
the prompt must match what generation sends, and the target must match what the
evaluator parses. Both failures are silent at training time.
"""

from __future__ import annotations

import pytest

from text2sql_rlvr.data import PromptConfig, build_messages, discover_split
from text2sql_rlvr.data.sft import (
    build_sft_record,
    count_over_budget,
    format_target,
    length_report,
)
from text2sql_rlvr.sql import extract_sql, validate_read_only


@pytest.fixture
def example(bird_root):
    return discover_split(bird_root, "mini_dev").load()[2]


class TestTargetFormat:
    def test_target_is_a_fenced_sql_block(self):
        assert format_target("SELECT 1").startswith("```sql")
        assert format_target("SELECT 1").endswith("```")

    def test_target_round_trips_through_the_evaluator(self):
        """What we train the model to emit must be what the evaluator reads back."""
        for sql in (
            "SELECT count(*) FROM staff",
            "SELECT `head count` FROM dept WHERE name = 'Sales'",
            "SELECT a FROM t WHERE s = 'semi;colon'",
            "  SELECT 1 ;  ",
        ):
            assert extract_sql(format_target(sql)) == " ".join(sql.strip().rstrip(";").split())

    def test_target_survives_read_only_validation(self):
        assert validate_read_only(extract_sql(format_target("SELECT 1"))).ok

    def test_multiline_gold_is_flattened(self):
        assert "\n" not in format_target("SELECT a\nFROM t\nWHERE b = 1")[7:-4]


class TestRecord:
    def test_prompt_is_identical_to_what_generation_sends(self, example):
        """The whole point. If these drift, training optimises the wrong input."""
        config = PromptConfig(instruction_version="v1")
        record = build_sft_record(example, "SCHEMA", config)
        assert record.messages[:-1] == build_messages(example, "SCHEMA", config)

    def test_last_turn_is_the_assistant_answer(self, example):
        record = build_sft_record(example, "SCHEMA")
        assert [m["role"] for m in record.messages] == ["system", "user", "assistant"]
        assert extract_sql(record.messages[-1]["content"]) == example.gold_sql

    def test_prompt_config_reaches_the_record(self, example):
        with_evidence = build_sft_record(example, "SCHEMA", PromptConfig(include_evidence=True))
        without = build_sft_record(example, "SCHEMA", PromptConfig(include_evidence=False))
        assert "Research means" in with_evidence.messages[1]["content"]
        assert "Research means" not in without.messages[1]["content"]

    def test_missing_gold_is_refused(self, example):
        from dataclasses import replace

        with pytest.raises(ValueError, match="no gold SQL"):
            build_sft_record(replace(example, gold_sql=""), "SCHEMA")

    def test_serialised_form_carries_provenance(self, example):
        record = build_sft_record(example, "SCHEMA").as_dict()
        assert record["question_id"] == example.question_id
        assert record["db_id"] == example.db_id
        assert len(record["messages"]) == 3


class TestLengthReport:
    def test_percentiles_and_estimate(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        records = [build_sft_record(e, "SCHEMA" * 100) for e in split.load()]
        stats = length_report(records)

        assert stats["n"] == 5
        assert stats["chars_p50"] <= stats["chars_p99"] <= stats["chars_max"]
        assert stats["est_tokens_max"] > 0

    def test_empty_input_does_not_divide_by_zero(self):
        assert length_report([])["n"] == 0

    def test_over_budget_ids_are_listed(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        records = [build_sft_record(e, "S" * 5000) for e in split.load()]
        assert len(count_over_budget(records, 1000)) == 5
        assert count_over_budget(records, 10**9) == []


def _outcomes(spec: dict[str, tuple[int, int]]) -> list[dict]:
    """``{db: (n_failed, n_solved)}`` -> outcome records with unique ids."""
    records = []
    qid = 0
    for db, (failed, solved) in spec.items():
        for _ in range(failed):
            records.append({"question_id": qid, "db_id": db, "official": False})
            qid += 1
        for _ in range(solved):
            records.append({"question_id": qid, "db_id": db, "official": True})
            qid += 1
    return records


def test_failure_policy_takes_wrong_answers_plus_a_replay_share():
    from text2sql_rlvr.data.sft import select_sft_examples, selection_report

    outcomes = _outcomes({"big": (200, 200), "small": (20, 20), "tiny": (4, 4)})
    ids = select_sft_examples(
        outcomes, policy="failure", size=60, replay_fraction=0.25, seed=0
    )
    report = selection_report(ids, outcomes)

    assert report["n"] == 60
    assert report["n_from_failures"] == 45 and report["n_from_solved"] == 15
    # round-robin over databases: the 400-question database does not swamp the rest
    assert report["n_databases"] == 3
    # proportional sampling would give the 400-question database ~52 of the 60;
    # round-robin gives it far fewer and drains the 8-question one entirely
    assert report["questions_per_db"]["tiny"] == 8
    assert report["questions_per_db"]["big"] < 30
    assert ids == select_sft_examples(
        outcomes, policy="failure", size=60, replay_fraction=0.25, seed=0
    )


def test_random_policy_is_the_control_and_all_keeps_everything():
    from text2sql_rlvr.data.sft import select_sft_examples, selection_report

    outcomes = _outcomes({"a": (30, 30), "b": (30, 30)})
    control = select_sft_examples(outcomes, policy="random", size=40, seed=1)
    assert selection_report(control, outcomes)["n"] == 40
    assert len(select_sft_examples(outcomes)) == 120
    assert select_sft_examples(outcomes, policy="random", size=40, seed=1) == control
    assert select_sft_examples(outcomes, policy="random", size=40, seed=2) != control


def test_selection_caps_per_database_and_rejects_bad_arguments():
    import pytest

    from text2sql_rlvr.data.sft import select_sft_examples, selection_report

    outcomes = _outcomes({"a": (50, 0), "b": (50, 0), "c": (2, 0)})
    ids = select_sft_examples(outcomes, policy="failure", cap_per_db=10, seed=0)
    report = selection_report(ids, outcomes)
    assert report["n"] == 22 and report["per_db_max"] == 10

    # asking for more than exists returns what exists rather than raising
    assert len(select_sft_examples(outcomes, policy="failure", size=500)) == 102

    with pytest.raises(ValueError):
        select_sft_examples(outcomes, policy="nonsense")
    with pytest.raises(ValueError):
        select_sft_examples(outcomes, policy="failure", replay_fraction=1.5)
    with pytest.raises(ValueError):
        select_sft_examples(outcomes, policy="random", replay_fraction=0.2)


def test_builder_renders_linked_schema_and_honours_the_policy(bird_root, tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "build_sft_data_script", scripts / "build_sft_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    split = discover_split(bird_root, "mini_dev")
    questions = tmp_path / "questions.json"
    questions.write_text(split.questions_path.read_text(encoding="utf-8"), encoding="utf-8")
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        "".join(
            json.dumps({"question_id": e.question_id, "db_id": e.db_id,
                        "official": e.question_id in (0, 1)}) + "\n"
            for e in split.load()
        ),
        encoding="utf-8",
    )
    out, manifest = tmp_path / "train.jsonl", tmp_path / "manifest.json"

    assert module.main([
        "--root", str(bird_root), "--split", "mini_dev", "--questions", str(questions),
        "--out", str(out), "--manifest", str(manifest),
        "--schema-mode", "linked", "--outcomes", str(outcomes), "--policy", "failure",
    ]) == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert {r["question_id"] for r in records} == {2, 3, 4}  # the three it got wrong
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["schema_mode"] == "linked"
    assert meta["selection"]["policy"] == "failure"
    assert meta["selection"]["report"]["n_from_failures"] == 3
    assert meta["selection"]["outcomes_sha256"]

    # "Who works in the Research department?" links dept and staff, not the whole schema
    linked = next(r for r in records if r["question_id"] == 2)
    assert "CREATE TABLE staff" in linked["messages"][1]["content"]
    assert linked["messages"][-1]["content"].startswith("```sql")


def test_selection_can_drop_questions_whose_gold_returns_nothing():
    from text2sql_rlvr.data.sft import select_sft_examples

    outcomes = [
        {"question_id": 0, "db_id": "a", "official": False, "gold_empty": False},
        {"question_id": 1, "db_id": "a", "official": False, "gold_empty": True},
        {"question_id": 2, "db_id": "a", "official": True, "gold_empty": True},
    ]
    assert select_sft_examples(outcomes, policy="failure") == [0, 1]
    assert select_sft_examples(outcomes, policy="failure", drop_gold_empty=True) == [0]
    assert select_sft_examples(outcomes, policy="all", drop_gold_empty=True) == [0]
