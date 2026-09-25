"""Scoring k samples per question, and sorting the questions into buckets.

The buckets are the point: only a question the model gets right *sometimes*
yields a preference pair made of its own answers. These tests use the real
sandbox against the fixture database, so "correct" here means the rows came
back equal, not that a string matched.
"""

from __future__ import annotations

from text2sql_rlvr.data import discover_split
from text2sql_rlvr.eval import score_samples

#: q0 is mixed (two right, one wrong), q2 is never right, q3 is always right.
SAMPLES = {
    0: [
        "SELECT count(*) FROM staff",
        "SELECT count(*) FROM dept",
        "SELECT count(staff_id) FROM staff",
    ],
    2: [
        "SELECT name FROM staff WHERE dept_id = 2",
        "SELECT name FROM staff WHERE dept_id = 3",
        "SELECT name FROM staff WHERE dept_id = 2",
    ],
    3: [
        "SELECT name FROM dept",
        "SELECT name FROM dept",
    ],
}


def report_for(bird_root, samples=None):
    split = discover_split(bird_root, "mini_dev")
    return score_samples(split.load(), samples or SAMPLES, split, n_workers=2)


def test_every_sample_is_executed(bird_root):
    report = report_for(bird_root)
    assert report.n_questions == 3
    assert report.n_samples_total == 8
    assert report.k_max == 3
    assert [len(o) for o in report.outcomes.values()] == [3, 3, 2]


def test_questions_land_in_the_right_bucket(bird_root):
    report = report_for(bird_root)
    assert (report.n_mixed, report.n_all_wrong, report.n_all_correct) == (1, 1, 1)


def test_pass_at_k_is_above_pass_at_1(bird_root):
    """The gap is what sampling buys: q0 is wrong on its first draw, right later."""
    samples = dict(SAMPLES)
    samples[0] = ["SELECT count(*) FROM dept", "SELECT count(*) FROM staff"]
    report = report_for(bird_root, samples)
    assert report.pass_at_1 == 33.33  # only q3
    assert report.pass_at_k == 66.67  # q0 and q3


def test_mean_pass_rate_weights_questions_equally(bird_root):
    """Per question, not per sample: a question with more samples must not count more."""
    report = report_for(bird_root)
    # q0 scores 2/3, q2 scores 0/3, q3 scores 2/2.
    assert report.mean_pass_rate == 55.56


def test_questions_without_samples_are_skipped_not_failed(bird_root):
    """q1 and q4 were never asked; counting them as wrong would understate the model."""
    report = report_for(bird_root)
    assert set(report.outcomes) == {0, 2, 3}
    assert report.n_questions == 3


def test_an_unparsable_sample_counts_as_wrong(bird_root):
    report = report_for(bird_root, {0: ["", "SELECT count(*) FROM staff"]})
    outcomes = report.outcomes[0]
    assert [o.official for o in outcomes] == [False, True]
    assert outcomes[0].pred_status == "rejected"
    assert report.n_mixed == 1


def test_outcomes_keep_sample_order(bird_root):
    """Position in the list is the sample index the outcomes file records."""
    report = report_for(bird_root)
    assert [o.pred_sql for o in report.outcomes[0]] == SAMPLES[0]


def test_metrics_are_flat_enough_for_the_ledger(bird_root):
    metrics = report_for(bird_root).metrics()
    assert metrics["n_mixed"] == 1 and metrics["k_max"] == 3
    assert all(isinstance(v, (int, float)) for v in metrics.values())
