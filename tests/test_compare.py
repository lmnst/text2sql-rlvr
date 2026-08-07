"""Where the official metric and the strict verifier agree, and where they must not.

The disagreement cases are the ones that matter: each is a way to collect
Execution Accuracy credit without having answered the question. They are
regression tests for the verifier, and the seed of the reward-hacking analysis.
"""

from __future__ import annotations

import pytest

from text2sql_rlvr.rewards.canonical import canon_value
from text2sql_rlvr.rewards.compare import (
    COLUMN_COUNT,
    GOLD_FAILED,
    PRED_FAILED,
    ROW_COUNT,
    ROW_ORDER,
    compare,
)
from text2sql_rlvr.rewards.sandbox import execute_sql


def run(db_path, sql):
    return execute_sql(db_path, sql)


def test_identical_queries_agree(db_path):
    sql = "SELECT name FROM staff WHERE dept_id = 1"
    verdict = compare(run(db_path, sql), run(db_path, sql), gold_sql=sql)
    assert verdict.official is True
    assert verdict.strict is True


def test_distinct_hides_duplicate_rows(db_path):
    """`SELECT DISTINCT` collapses a 5-row answer to 3 and the official set
    comparison cannot tell."""
    gold_sql = "SELECT salary FROM staff"
    pred_sql = "SELECT DISTINCT salary FROM staff"
    verdict = compare(run(db_path, pred_sql), run(db_path, gold_sql), gold_sql=gold_sql)
    assert verdict.official is True
    assert verdict.strict is False
    assert verdict.reason == ROW_COUNT
    assert verdict.disagrees is True


def test_empty_result_matches_any_other_empty_result(db_path):
    """Two empty result sets with different column counts are equal as sets.

    A model that learns to emit an always-empty query collects credit on every
    question whose gold answer happens to be empty.
    """
    gold_sql = "SELECT name FROM staff WHERE salary > 1000"
    pred_sql = "SELECT staff_id, name FROM staff WHERE 1 = 0"
    verdict = compare(run(db_path, pred_sql), run(db_path, gold_sql), gold_sql=gold_sql)
    assert verdict.official is True
    assert verdict.strict is False
    assert verdict.reason == COLUMN_COUNT


def test_empty_gold_is_flagged_even_when_both_verifiers_pass(db_path):
    """Same column count, both empty: no result comparison can catch this one,
    so the flag has to reach the analysis layer instead."""
    gold_sql = "SELECT name FROM staff WHERE salary > 1000"
    pred_sql = "SELECT name FROM staff WHERE 1 = 0"
    verdict = compare(run(db_path, pred_sql), run(db_path, gold_sql), gold_sql=gold_sql)
    assert verdict.official is True
    assert verdict.strict is True
    assert verdict.gold_empty is True


def test_float_noise_does_not_fail_the_strict_verifier(db_path):
    """The strict verifier is not uniformly stricter -- it is stricter about
    what matters and more forgiving about float representation."""
    gold_sql = "SELECT 0.1 + 0.2"
    pred_sql = "SELECT 0.3"
    verdict = compare(run(db_path, pred_sql), run(db_path, gold_sql), gold_sql=gold_sql)
    assert verdict.official is False
    assert verdict.strict is True


def test_column_count_mismatch_is_rejected(db_path):
    gold_sql = "SELECT name FROM staff"
    pred_sql = "SELECT staff_id, name FROM staff"
    verdict = compare(run(db_path, pred_sql), run(db_path, gold_sql), gold_sql=gold_sql)
    assert verdict.strict is False
    assert verdict.reason == COLUMN_COUNT


def test_failed_prediction_scores_zero(db_path):
    gold_sql = "SELECT name FROM staff"
    verdict = compare(run(db_path, "SELECT nope FROM staff"), run(db_path, gold_sql),
                      gold_sql=gold_sql)
    assert verdict.official is False
    assert verdict.strict is False
    assert verdict.reason == PRED_FAILED


def test_broken_gold_is_reported_separately(db_path):
    gold_sql = "SELECT nope FROM staff"
    verdict = compare(run(db_path, "SELECT name FROM staff"), run(db_path, gold_sql),
                      gold_sql=gold_sql)
    assert verdict.strict is False
    assert verdict.reason == GOLD_FAILED


class TestOrderPolicy:
    gold_sql = "SELECT name FROM staff ORDER BY salary DESC, name ASC"
    pred_sql = "SELECT name FROM staff ORDER BY salary ASC, name DESC"

    def _verdict(self, db_path, policy):
        return compare(
            run(db_path, self.pred_sql),
            run(db_path, self.gold_sql),
            gold_sql=self.gold_sql,
            order_policy=policy,
        )

    def test_default_ignores_row_order(self, db_path):
        verdict = self._verdict(db_path, "ignore")
        assert verdict.strict is True
        assert verdict.gold_has_order_by is True
        assert verdict.order_matches is False

    def test_gold_order_by_policy_enforces_it(self, db_path):
        verdict = self._verdict(db_path, "gold_order_by")
        assert verdict.strict is False
        assert verdict.reason == ROW_ORDER

    def test_always_policy_enforces_it(self, db_path):
        assert self._verdict(db_path, "always").strict is False

    def test_unknown_policy_raises(self, db_path):
        with pytest.raises(ValueError):
            self._verdict(db_path, "whatever")


@pytest.mark.parametrize(
    "left,right,equal",
    [
        (1, 1.0, True),
        (2.0000001, 2.0000002, True),
        (1.0, 1.1, False),
        (None, 0, False),
        (None, "", False),
        (0, "", False),
        (0, "0", False),
        (True, 1, True),
        (2**60 + 1, float(2**60 + 1), False),
    ],
)
def test_value_canonicalisation(left, right, equal):
    assert (canon_value(left) == canon_value(right)) is equal
