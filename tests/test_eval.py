"""End-to-end evaluation over a BIRD-shaped split."""

from __future__ import annotations

import json

from text2sql_rlvr.data import discover_split
from text2sql_rlvr.eval import evaluate

#: One correct answer, one hack, one wrong answer, one unparsable completion,
#: and one question with no prediction at all.
PREDICTIONS = {
    0: "```sql\nSELECT count(*) FROM staff\n```",
    1: "```sql\nSELECT DISTINCT salary FROM staff\n```",
    2: "```sql\nSELECT name FROM staff WHERE dept_id = 2\n```",
    3: "I am not sure how to answer this.",
}


def report_for(bird_root, **kwargs):
    split = discover_split(bird_root, "mini_dev")
    return evaluate(split.load(), PREDICTIONS, split, n_workers=4, **kwargs)


def test_headline_metrics(bird_root):
    report = report_for(bird_root)
    assert report.n == 5
    # q0 exact, q1 credited by the set comparison despite dropping duplicates.
    assert report.official_ex == 40.0
    # The strict verifier keeps only q0.
    assert report.strict_ex == 20.0


def test_verifier_gap_is_attributed(bird_root):
    report = report_for(bird_root)
    assert report.n_official_not_strict == 1
    assert report.n_strict_not_official == 0

    disagreeing = [o for o in report.outcomes if o.official and not o.strict]
    assert [o.question_id for o in disagreeing] == [1]
    assert disagreeing[0].reason == "row_count"
    assert (disagreeing[0].pred_n_rows, disagreeing[0].gold_n_rows) == (3, 5)


def test_strict_failures_are_labelled(bird_root):
    report = report_for(bird_root)
    # q1 drops duplicates, q2 selects the wrong department: both row_count.
    # q3 produced no parsable SQL and q4 has no prediction: both pred_failed.
    assert report.strict_reason_counts == {"row_count": 2, "pred_failed": 2}


def test_missing_and_unparsable_predictions_are_counted(bird_root):
    report = report_for(bird_root)
    assert report.n_missing_predictions == 1
    assert report.n_unparsed == 1


def test_breakdown_by_difficulty(bird_root):
    report = report_for(bird_root)
    assert report.by_difficulty["simple"]["n"] == 2
    assert report.by_difficulty["simple"]["official_ex"] == 100.0
    assert report.by_difficulty["simple"]["strict_ex"] == 50.0
    assert report.by_difficulty["challenging"]["official_ex"] == 0.0


def test_gold_queries_all_execute(bird_root):
    report = report_for(bird_root)
    assert report.gold_status_counts == {"ok": 5}


def test_single_worker_matches_thread_pool(bird_root):
    split = discover_split(bird_root, "mini_dev")
    serial = evaluate(split.load(), PREDICTIONS, split, n_workers=1)
    parallel = evaluate(split.load(), PREDICTIONS, split, n_workers=8)
    assert serial.official_ex == parallel.official_ex
    assert serial.strict_ex == parallel.strict_ex
    assert [o.question_id for o in serial.outcomes] == [o.question_id for o in parallel.outcomes]


def test_order_policy_reaches_the_verifier(bird_root):
    strict_order = report_for(bird_root, order_policy="always")
    assert strict_order.official_ex == 40.0
    assert strict_order.strict_ex <= 20.0


def test_outcomes_round_trip_to_jsonl(bird_root, tmp_path):
    report = report_for(bird_root)
    path = report.write_outcomes(tmp_path / "outcomes.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert {"question_id", "official", "strict", "reason", "pred_sql"} <= set(rows[0])


def test_metrics_dict_is_flat_and_ledger_ready(bird_root):
    metrics = report_for(bird_root).metrics()
    assert metrics["official_ex"] == 40.0
    assert metrics["n_official_not_strict"] == 1
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_progress_callback_fires_once_per_example(bird_root):
    split = discover_split(bird_root, "mini_dev")
    seen = []
    evaluate(split.load(), PREDICTIONS, split, n_workers=1,
             on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(i, 5) for i in range(1, 6)]
