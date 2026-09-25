"""Preference pairs: grounded in execution, parsable back, and never degenerate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from text2sql_rlvr.data import discover_split, format_schema, load_schema
from text2sql_rlvr.data.preference import (
    GOLD_FALLBACK,
    MINED,
    MinedPair,
    build_preference_record,
    cap_gold_fallback,
    mine_pair,
    mined_pair_report,
    normalise_sql,
    pair_is_usable,
    pair_report,
    uses_gold_style_alias,
)
from text2sql_rlvr.sql import extract_sql

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_a_pair_needs_two_genuinely_different_queries():
    assert pair_is_usable("SELECT a FROM t", "SELECT b FROM t")
    # formatting-only differences leave nothing to learn from
    assert not pair_is_usable("SELECT a FROM t", "select   a\nfrom T;")
    assert not pair_is_usable("SELECT a FROM t", "")
    assert not pair_is_usable("", "SELECT a FROM t")
    assert normalise_sql("SELECT  a\nFROM T ;") == "select a from t"


def test_record_pairs_gold_against_what_the_model_wrote(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    example = split.load()[0]  # How many staff members are there? -> SELECT count(*) FROM staff

    record = build_preference_record(
        example, format_schema(schema), "SELECT count(*) FROM dept"
    )
    assert [m["role"] for m in record["prompt"]] == ["system", "user"]
    assert example.question in record["prompt"][1]["content"]
    assert extract_sql(record["chosen"][0]["content"]) == "SELECT count(*) FROM staff"
    assert extract_sql(record["rejected"][0]["content"]) == "SELECT count(*) FROM dept"
    assert record["question_id"] == example.question_id and record["db_id"] == "company"

    with pytest.raises(ValueError):
        build_preference_record(example, "s", "SELECT COUNT(*)   FROM  staff;")
    with pytest.raises(ValueError):
        build_preference_record(example, "s", "")


def test_report_counts_databases_and_why_each_rejection_was_wrong():
    records = [
        {"question_id": 1, "db_id": "a"},
        {"question_id": 2, "db_id": "a"},
        {"question_id": 3, "db_id": "b"},
    ]
    outcomes = {
        1: {"reason": "row_count", "pred_status": "ok"},
        2: {"reason": "pred_failed", "pred_status": "error"},
        3: {"reason": "row_count", "pred_status": "ok"},
    }
    report = pair_report(records, outcomes)
    assert report["n"] == 3 and report["n_databases"] == 2
    assert report["per_db_min"] == 1 and report["per_db_max"] == 2
    assert report["rejected_by_reason"] == {"row_count": 2, "pred_failed": 1}
    assert report["rejected_by_exec_status"] == {"ok": 2, "error": 1}


@pytest.fixture(scope="module")
def build_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_dpo_data_script", SCRIPTS / "build_dpo_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_keeps_only_failures_that_form_a_real_pair(build_module, bird_root, tmp_path):
    split = discover_split(bird_root, "mini_dev")
    examples = split.load()
    gold = {e.question_id: e.gold_sql for e in examples}
    rows = [
        # a real disagreement: kept
        {"question_id": 0, "db_id": "company", "official": False, "gold_empty": False,
         "reason": "row_count", "pred_status": "ok",
         "gold_sql": gold[0], "pred_sql": "SELECT count(*) FROM dept"},
        # the model wrote gold with different spacing: no gradient, dropped
        {"question_id": 1, "db_id": "company", "official": False, "gold_empty": False,
         "reason": "row_values", "pred_status": "ok",
         "gold_sql": gold[1], "pred_sql": "select   salary\nfrom staff;"},
        # crashed with no query at all: dropped
        {"question_id": 2, "db_id": "company", "official": False, "gold_empty": False,
         "reason": "pred_failed", "pred_status": "error",
         "gold_sql": gold[2], "pred_sql": ""},
        # answered correctly: not a pair
        {"question_id": 3, "db_id": "company", "official": True, "gold_empty": False,
         "reason": None, "pred_status": "ok",
         "gold_sql": gold[3], "pred_sql": gold[3]},
    ]
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(split.questions_path.read_text(encoding="utf-8"), encoding="utf-8")
    out, manifest = tmp_path / "dpo.jsonl", tmp_path / "manifest.json"

    assert build_module.main([
        "--root", str(bird_root), "--split", "mini_dev", "--questions", str(questions),
        "--outcomes", str(outcomes), "--out", str(out), "--manifest", str(manifest),
        "--schema-mode", "linked",
    ]) == 0

    pairs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert [p["question_id"] for p in pairs] == [0]
    assert extract_sql(pairs[0]["rejected"][0]["content"]) == "SELECT count(*) FROM dept"

    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["schema_mode"] == "linked"
    assert meta["report"]["rejected_by_reason"] == {"row_count": 1}
    assert meta["source_kind"] == "single_answer"
    assert meta["source_sha256"]
    # Every pair here is gold against the model: the style audit should say so.
    assert meta["provenance"]["by_source"] == {GOLD_FALLBACK: 1}


def test_builder_can_restrict_to_one_kind_of_mistake(build_module, bird_root, tmp_path):
    split = discover_split(bird_root, "mini_dev")
    gold = {e.question_id: e.gold_sql for e in split.load()}
    rows = [
        {"question_id": 0, "db_id": "company", "official": False, "gold_empty": False,
         "reason": "row_count", "pred_status": "ok",
         "gold_sql": gold[0], "pred_sql": "SELECT count(*) FROM dept"},
        {"question_id": 1, "db_id": "company", "official": False, "gold_empty": False,
         "reason": "column_count", "pred_status": "ok",
         "gold_sql": gold[1], "pred_sql": "SELECT name FROM staff"},
    ]
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(split.questions_path.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "dpo.jsonl"

    build_module.main([
        "--root", str(bird_root), "--split", "mini_dev", "--questions", str(questions),
        "--outcomes", str(outcomes), "--out", str(out),
        "--manifest", str(tmp_path / "m.json"), "--reasons", "column_count",
    ])
    pairs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert [p["question_id"] for p in pairs] == [1]


def sample(sql: str, official: bool) -> dict:
    return {"sql": sql, "official": official}


class TestMining:
    """One pair per question, both sides written by the model where possible."""

    def test_the_most_frequent_right_and_wrong_answers_are_paired(self):
        samples = [
            sample("SELECT a FROM t", True),
            sample("SELECT a FROM t", True),
            sample("SELECT DISTINCT a FROM t", True),
            sample("SELECT b FROM t", False),
            sample("SELECT b FROM t", False),
            sample("SELECT c FROM t", False),
        ]
        pair = mine_pair(samples, "GOLD", question_id=7, db_id="company")
        assert (pair.chosen_sql, pair.rejected_sql) == ("SELECT a FROM t", "SELECT b FROM t")
        assert pair.source == MINED
        assert (pair.chosen_count, pair.rejected_count) == (2, 2)
        assert (pair.n_samples, pair.n_correct) == (6, 3)

    def test_the_mode_ignores_spelling(self):
        """Two spellings of one query must not lose to a single different one."""
        samples = [
            sample("SELECT a FROM t", False),
            sample("select   a\nfrom T;", False),
            sample("SELECT b FROM t", False),
            sample("SELECT z FROM t", True),
        ]
        pair = mine_pair(samples, "GOLD", question_id=1, db_id="company")
        assert pair.rejected_sql == "SELECT a FROM t"
        assert pair.rejected_count == 2

    def test_a_question_never_answered_right_falls_back_to_gold(self):
        pair = mine_pair(
            [sample("SELECT b FROM t", False)], "SELECT a FROM t", question_id=1, db_id="company"
        )
        assert pair.source == GOLD_FALLBACK
        assert pair.chosen_sql == "SELECT a FROM t"
        assert pair.n_correct == 0

    def test_a_question_always_answered_right_yields_nothing(self):
        assert mine_pair(
            [sample("SELECT a FROM t", True)] * 4, "GOLD", question_id=1, db_id="company"
        ) is None

    def test_empty_samples_count_as_wrong_but_cannot_be_the_rejected_side(self):
        """Training against an empty query teaches nothing the model was going to do."""
        samples = [sample("", False), sample("", False), sample("SELECT b FROM t", False)]
        pair = mine_pair(samples, "SELECT a FROM t", question_id=1, db_id="company")
        assert pair.rejected_sql == "SELECT b FROM t"
        assert pair.n_samples == 3

    def test_no_pair_when_every_wrong_sample_is_empty(self):
        assert mine_pair(
            [sample("", False), sample("", False)], "GOLD", question_id=1, db_id="company"
        ) is None

    def test_a_degenerate_pair_is_dropped(self):
        """Gold and the wrong answer differing only in spacing has a zero gradient."""
        samples = [sample("select  a from T ;", False)]
        assert mine_pair(
            samples, "SELECT a FROM t", question_id=1, db_id="company"
        ) is None


def pair(question_id: int, source: str, chosen: str = "SELECT a FROM t") -> MinedPair:
    return MinedPair(
        question_id=question_id,
        db_id="company",
        chosen_sql=chosen,
        rejected_sql="SELECT b FROM t",
        source=source,
        n_samples=8,
        n_correct=1 if source == MINED else 0,
        chosen_count=1,
        rejected_count=2,
    )


class TestGoldFallbackCap:
    def test_the_cap_is_a_share_of_the_result_not_of_the_input(self):
        """9 mined pairs admit 3 fallbacks, which is a quarter of the 12 written."""
        pairs = [pair(i, MINED) for i in range(9)]
        pairs += [pair(100 + i, GOLD_FALLBACK) for i in range(20)]
        kept = cap_gold_fallback(pairs, max_fraction=0.25)
        assert len(kept) == 12
        assert sum(1 for p in kept if p.source == GOLD_FALLBACK) == 3

    def test_mined_pairs_are_never_dropped(self):
        pairs = [pair(i, MINED) for i in range(5)] + [pair(100, GOLD_FALLBACK)]
        kept = cap_gold_fallback(pairs, max_fraction=0.25)
        assert sum(1 for p in kept if p.source == MINED) == 5

    def test_a_fallback_bucket_under_the_cap_is_kept_whole(self):
        pairs = [pair(i, MINED) for i in range(10)] + [pair(100, GOLD_FALLBACK)]
        assert len(cap_gold_fallback(pairs, max_fraction=0.25)) == 11

    def test_zero_drops_the_bucket_entirely(self):
        pairs = [pair(0, MINED), pair(1, GOLD_FALLBACK)]
        assert [p.source for p in cap_gold_fallback(pairs, max_fraction=0.0)] == [MINED]

    def test_the_choice_is_reproducible(self):
        pairs = [pair(i, MINED) for i in range(3)]
        pairs += [pair(100 + i, GOLD_FALLBACK) for i in range(10)]
        first = [p.question_id for p in cap_gold_fallback(pairs, seed=5)]
        assert first == [p.question_id for p in cap_gold_fallback(pairs, seed=5)]

    def test_order_is_preserved(self):
        pairs = [pair(i, MINED) for i in range(4)]
        assert [p.question_id for p in cap_gold_fallback(pairs)] == [0, 1, 2, 3]


class TestStyleAudit:
    def test_gold_house_style_is_detected(self):
        assert uses_gold_style_alias("SELECT T1.a FROM t AS T1 JOIN u AS T2 ON T1.id = T2.id")
        assert not uses_gold_style_alias("SELECT a FROM t JOIN u ON t.id = u.id")

    def test_the_report_shows_the_gap_between_the_two_sides(self):
        """The 82.8%-against-5.5% trap, made visible before training instead of after."""
        pairs = [
            pair(0, GOLD_FALLBACK, chosen="SELECT T1.a FROM t AS T1"),
            pair(1, GOLD_FALLBACK, chosen="SELECT T1.a FROM t AS T1"),
            pair(2, MINED, chosen="SELECT a FROM t"),
            pair(3, MINED, chosen="SELECT a FROM t"),
        ]
        report = mined_pair_report(pairs)
        assert report["alias_rate_chosen"] == 0.5
        assert report["alias_rate_rejected"] == 0.0
        assert report["alias_rate_chosen_mined_only"] == 0.0
        assert report["gold_fallback_share"] == 0.5
        assert report["by_source"] == {GOLD_FALLBACK: 2, MINED: 2}

    def test_an_empty_set_reports_nothing_rather_than_dividing_by_zero(self):
        assert mined_pair_report([]) == {"n": 0}


def scored(question_id: int, index: int, pred_sql: str, official: bool, gold: str) -> dict:
    """One row in the shape score_samples.py writes."""
    return {
        "sample_index": index,
        "question_id": question_id,
        "db_id": "company",
        "pred_sql": pred_sql,
        "gold_sql": gold,
        "official": official,
        "strict": official,
        "reason": None if official else "row_count",
        "pred_status": "ok",
        "gold_empty": False,
    }


@pytest.fixture
def sampled_outcomes(bird_root, tmp_path):
    """q0 mixed, q2 never right, q3 always right."""
    gold = {e.question_id: e.gold_sql for e in discover_split(bird_root, "mini_dev").load()}
    rows = [
        scored(0, 0, "SELECT count(*) FROM staff", True, gold[0]),
        scored(0, 1, "SELECT count(*) FROM staff", True, gold[0]),
        scored(0, 2, "SELECT count(*) FROM dept", False, gold[0]),
        scored(2, 0, "SELECT name FROM staff WHERE dept_id = 2", False, gold[2]),
        scored(2, 1, "SELECT name FROM staff WHERE dept_id = 3", False, gold[2]),
        scored(3, 0, "SELECT name FROM dept", True, gold[3]),
    ]
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def run_builder(build_module, bird_root, tmp_path, *extra):
    questions = tmp_path / "questions.json"
    split = discover_split(bird_root, "mini_dev")
    questions.write_text(split.questions_path.read_text(encoding="utf-8"), encoding="utf-8")
    out, manifest = tmp_path / "dpo.jsonl", tmp_path / "manifest.json"
    code = build_module.main([
        "--root", str(bird_root), "--split", "mini_dev", "--questions", str(questions),
        "--out", str(out), "--manifest", str(manifest), "--schema-mode", "linked", *extra,
    ])
    pairs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    return code, pairs, json.loads(manifest.read_text(encoding="utf-8"))


class TestBuilderFromSamples:
    def test_pairs_are_the_model_own_answers_where_it_ever_got_one_right(
        self, build_module, bird_root, tmp_path, sampled_outcomes
    ):
        # One mined pair admits no fallback at 25%, so the cap is loosened here
        # to see both kinds; the cap itself is tested below.
        code, pairs, meta = run_builder(
            build_module, bird_root, tmp_path,
            "--samples", str(sampled_outcomes), "--gold-fallback-fraction", "0.5",
        )
        assert code == 0
        # q3 is always right (no pair); q0 is mixed; q2 never right, so gold stands in.
        assert sorted(p["question_id"] for p in pairs) == [0, 2]
        by_id = {p["question_id"]: p for p in pairs}
        assert extract_sql(by_id[0]["chosen"][0]["content"]) == "SELECT count(*) FROM staff"
        assert extract_sql(by_id[0]["rejected"][0]["content"]) == "SELECT count(*) FROM dept"
        assert by_id[0]["pair_source"] == MINED
        assert by_id[2]["pair_source"] == GOLD_FALLBACK
        assert meta["source_kind"] == "samples"

    def test_the_gold_fallback_bucket_can_be_dropped(
        self, build_module, bird_root, tmp_path, sampled_outcomes
    ):
        _, pairs, meta = run_builder(
            build_module, bird_root, tmp_path,
            "--samples", str(sampled_outcomes), "--gold-fallback-fraction", "0",
        )
        assert [p["question_id"] for p in pairs] == [0]
        assert meta["provenance"]["gold_fallback_share"] == 0.0

    def test_one_mined_pair_does_not_carry_a_fallback_at_the_default_cap(
        self, build_module, bird_root, tmp_path, sampled_outcomes
    ):
        """A quarter of a two-pair set is not a whole pair, so the bucket waits."""
        _, pairs, meta = run_builder(
            build_module, bird_root, tmp_path, "--samples", str(sampled_outcomes)
        )
        assert [p["pair_source"] for p in pairs] == [MINED]
        assert meta["selection"]["n_gold_fallback_before_cap"] == 1

    def test_the_manifest_records_the_style_audit(
        self, build_module, bird_root, tmp_path, sampled_outcomes
    ):
        _, _, meta = run_builder(
            build_module, bird_root, tmp_path, "--samples", str(sampled_outcomes)
        )
        assert "alias_rate_chosen" in meta["provenance"]
        assert "alias_rate_rejected" in meta["provenance"]
        assert meta["selection"]["gold_fallback_fraction"] == 0.25

    def test_the_rejected_reason_comes_from_the_rejected_sample(
        self, build_module, bird_root, tmp_path, sampled_outcomes
    ):
        _, _, meta = run_builder(
            build_module, bird_root, tmp_path,
            "--samples", str(sampled_outcomes), "--gold-fallback-fraction", "0.5",
        )
        assert meta["report"]["rejected_by_reason"] == {"row_count": 2}

    def test_passing_both_inputs_is_refused(
        self, build_module, bird_root, tmp_path, sampled_outcomes
    ):
        with pytest.raises(SystemExit):
            run_builder(
                build_module, bird_root, tmp_path,
                "--samples", str(sampled_outcomes), "--outcomes", str(sampled_outcomes),
            )
