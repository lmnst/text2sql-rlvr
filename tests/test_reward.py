"""The reward function.

This is the one piece of the project that a training run cannot recover from.
A silently wrong reward does not crash: it produces a smooth loss curve, a
rising mean reward, and a model that has learned the wrong thing. So the tests
here are less about coverage and more about pinning down the exact payout for
every shape of output a policy might discover.
"""

from __future__ import annotations

import pytest

from text2sql_rlvr.data import discover_split
from text2sql_rlvr.rewards.reward import RewardConfig, RewardStats, compute_reward
from text2sql_rlvr.rewards.sandbox import SqlExecutor


@pytest.fixture
def executor():
    with SqlExecutor() as ex:
        yield ex


@pytest.fixture
def db(bird_root):
    return discover_split(bird_root, "mini_dev").db_path("company")


def fence(sql: str) -> str:
    return f"```sql\n{sql}\n```"


def score(completion, gold, db, executor, config=None):
    return compute_reward(completion, gold, db, executor=executor, config=config)


class TestDefaultConfig:
    """Correctness only, judged strictly. Nothing else pays."""

    def test_correct_answer_earns_the_full_reward(self, db, executor):
        gold = "SELECT name FROM staff WHERE dept_id = 1"
        result = score(fence(gold), gold, db, executor)
        assert result.reward == 1.0
        assert result.correct is True

    def test_a_different_but_equivalent_query_also_earns_it(self, db, executor):
        gold = "SELECT name FROM staff WHERE dept_id = 1"
        equivalent = (
            "SELECT s.name FROM staff s JOIN dept d ON s.dept_id = d.dept_id"
            " WHERE d.name = 'Research'"
        )
        assert score(fence(equivalent), gold, db, executor).reward == 1.0

    def test_wrong_answer_earns_nothing(self, db, executor):
        result = score(
            fence("SELECT name FROM staff WHERE dept_id = 2"),
            "SELECT name FROM staff WHERE dept_id = 1",
            db,
            executor,
        )
        assert result.reward == 0.0
        assert result.executed is True  # it ran; it was simply wrong

    def test_sql_that_runs_but_reads_nothing_earns_nothing(self, db, executor):
        """`SELECT 1` executes cleanly. Under the default config that is worth zero."""
        result = score(fence("SELECT 1"), "SELECT name FROM staff", db, executor)
        assert result.reward == 0.0
        assert result.executed is True
        assert result.no_from_clause is True

    def test_broken_sql_earns_nothing(self, db, executor):
        result = score(fence("SELECT nope FROM staff"), "SELECT name FROM staff", db, executor)
        assert result.reward == 0.0
        assert result.executed is False

    def test_no_sql_at_all_earns_nothing(self, db, executor):
        result = score("I am not sure.", "SELECT name FROM staff", db, executor)
        assert result.reward == 0.0
        assert result.parsed is False
        assert result.reason == "no_sql_in_completion"

    def test_write_attempts_are_refused_not_executed(self, db, executor):
        result = score(fence("DROP TABLE staff"), "SELECT name FROM staff", db, executor)
        assert result.reward == 0.0
        assert result.parsed is False
        assert "DROP" in (result.reason or "")

    def test_missing_gold_earns_nothing(self, db, executor):
        assert score(fence("SELECT 1"), "", db, executor).reward == 0.0


class TestStrictVsOfficial:
    """Why the strict verifier is the reward. See milestone 9."""

    gold = "SELECT DISTINCT salary FROM staff"
    forgot_distinct = "SELECT salary FROM staff"

    def test_omitting_distinct_earns_nothing_by_default(self, db, executor):
        result = score(fence(self.forgot_distinct), self.gold, db, executor)
        assert result.reward == 0.0
        assert result.correct is False
        assert result.official is True, "the official metric would have credited this"
        assert result.pred_n_rows > result.gold_n_rows

    def test_the_official_config_pays_for_it(self, db, executor):
        result = score(
            fence(self.forgot_distinct),
            self.gold,
            db,
            executor,
            RewardConfig(use_official=True),
        )
        assert result.reward == 1.0
        assert result.correct is False
        assert result.credited_without_answering is True


class TestPartialCredit:
    """The deliberate hack lever, and what it buys a policy that finds it."""

    def test_execution_bonus_pays_for_select_one(self, db, executor):
        config = RewardConfig(execution_bonus=0.1)
        result = score(fence("SELECT 1"), "SELECT name FROM staff", db, executor, config)
        assert result.reward == pytest.approx(0.1)
        assert result.credited_without_answering is True

    def test_format_bonus_pays_for_any_parsable_query(self, db, executor):
        config = RewardConfig(format_bonus=0.1)
        result = score(fence("SELECT nope FROM staff"), "SELECT name FROM staff", db, executor,
                       config)
        assert result.reward == pytest.approx(0.1)

    def test_bonuses_stack_under_the_correct_answer(self, db, executor):
        gold = "SELECT name FROM staff"
        config = RewardConfig(format_bonus=0.1, execution_bonus=0.1)
        assert score(fence(gold), gold, db, executor, config).reward == pytest.approx(1.2)
        assert config.max_reward == pytest.approx(1.2)

    def test_default_config_pays_no_partial_credit_at_all(self, db, executor):
        for completion in (fence("SELECT 1"), fence("SELECT nope FROM staff"), "nothing"):
            result = score(completion, "SELECT name FROM staff", db, executor)
            assert result.reward == 0.0, completion


class TestStats:
    def test_tracks_the_gap_the_project_is_about(self, db, executor):
        stats = RewardStats()
        gold = "SELECT DISTINCT salary FROM staff"
        stats.update(score(fence(gold), gold, db, executor))
        stats.update(score(fence("SELECT salary FROM staff"), gold, db, executor))

        summary = stats.as_dict()
        assert summary["n"] == 2
        assert summary["strict_acc"] == 0.5
        assert summary["official_acc"] == 1.0
        assert summary["official_minus_strict"] == 0.5

    def test_counts_degenerate_shapes(self, db, executor):
        stats = RewardStats()
        stats.update(score(fence("SELECT 1"), "SELECT name FROM staff", db, executor))
        stats.update(score(fence("SELECT name FROM staff WHERE 1 = 0"),
                           "SELECT name FROM staff", db, executor))

        summary = stats.as_dict()
        # SELECT 1 returns one row, so only the WHERE 1 = 0 query is empty.
        assert summary["no_from_rate"] == 0.5
        assert summary["empty_result_rate"] == 0.5

    def test_empty_stats_do_not_divide_by_zero(self):
        assert RewardStats().as_dict() == {"n": 0}


def test_config_is_serialisable_for_the_ledger():
    config = RewardConfig(execution_bonus=0.1)
    assert config.as_dict()["execution_bonus"] == 0.1
    assert config.as_dict()["use_official"] is False
