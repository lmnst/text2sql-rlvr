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
