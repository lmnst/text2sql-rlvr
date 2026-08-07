"""Filtering and train/val construction.

The database-disjoint default is the part worth pinning: a random split would
share schemas with training and quietly report a val number that does not
predict dev.
"""

from __future__ import annotations

import pytest

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.splits import (
    NO_GOLD,
    NOT_EXECUTABLE,
    TOO_SLOW,
    plan_splits,
)
from text2sql_rlvr.eval.surface import GoldFacts


def example(qid: int, db_id: str = "db0", gold: str = "SELECT 1") -> BirdExample:
    return BirdExample(question_id=qid, db_id=db_id, question="q?", gold_sql=gold)


def fact(qid: int, *, status: str = "ok", elapsed: float = 0.1) -> GoldFacts:
    return GoldFacts(
        question_id=qid,
        db_id="db0",
        difficulty=None,
        status=status,
        n_rows=1,
        n_cols=1,
        n_duplicate_rows=0,
        has_top_level_order_by=False,
        elapsed_s=elapsed,
    )


def corpus(n_dbs: int = 10, per_db: int = 20):
    examples, facts = [], []
    qid = 0
    for d in range(n_dbs):
        for _ in range(per_db):
            examples.append(example(qid, db_id=f"db{d}"))
            facts.append(fact(qid))
            qid += 1
    return examples, facts


class TestFiltering:
    def test_unexecutable_gold_is_dropped(self):
        examples = [example(0), example(1), example(2)]
        facts = [fact(0), fact(1, status="error"), fact(2, status="timeout")]
        plan = plan_splits(examples, facts, val_size=0)

        assert plan.train_ids == (0,)
        assert set(plan.excluded[NOT_EXECUTABLE]) == {1, 2}

    def test_slow_gold_is_dropped_at_the_training_timeout(self):
        examples = [example(0), example(1)]
        facts = [fact(0, elapsed=2.0), fact(1, elapsed=25.0)]
        plan = plan_splits(examples, facts, val_size=0, max_gold_seconds=10.0)

        assert plan.train_ids == (0,)
        assert plan.excluded[TOO_SLOW] == (1,)

    def test_slow_filter_can_be_disabled(self):
        examples = [example(0), example(1)]
        facts = [fact(0, elapsed=2.0), fact(1, elapsed=25.0)]
        plan = plan_splits(examples, facts, val_size=0, max_gold_seconds=0)

        assert set(plan.train_ids) == {0, 1}

    def test_missing_gold_sql_is_dropped(self):
        plan = plan_splits([example(0, gold="")], [fact(0)], val_size=0)
        assert plan.excluded[NO_GOLD] == (0,)

    def test_unmeasured_questions_are_dropped(self):
        plan = plan_splits([example(0), example(1)], [fact(0)], val_size=0)
        assert plan.excluded["gold_not_measured"] == (1,)

    def test_nothing_is_silently_lost(self):
        examples, facts = corpus(n_dbs=4, per_db=5)
        facts[3] = fact(3, status="error")
        plan = plan_splits(examples, facts, val_size=5)

        accounted = len(plan.train_ids) + len(plan.val_ids) + plan.n_excluded
        assert accounted == len(examples)


class TestDatabaseDisjointSplit:
    def test_train_and_val_share_no_database(self):
        examples, facts = corpus()
        plan = plan_splits(examples, facts, val_size=50, seed=0)

        by_id = {e.question_id: e for e in examples}
        train_dbs = {by_id[i].db_id for i in plan.train_ids}
        val_dbs = {by_id[i].db_id for i in plan.val_ids}
        assert train_dbs & val_dbs == set()
        assert val_dbs == set(plan.val_db_ids)

    def test_val_reaches_roughly_the_requested_size(self):
        examples, facts = corpus()
        plan = plan_splits(examples, facts, val_size=50, seed=0)
        assert 50 <= len(plan.val_ids) < 50 + 20

    def test_split_is_deterministic_for_a_seed(self):
        examples, facts = corpus()
        a = plan_splits(examples, facts, val_size=50, seed=7)
        b = plan_splits(examples, facts, val_size=50, seed=7)
        assert a.val_ids == b.val_ids

    def test_different_seeds_give_different_splits(self):
        examples, facts = corpus()
        a = plan_splits(examples, facts, val_size=50, seed=1)
        b = plan_splits(examples, facts, val_size=50, seed=2)
        assert a.val_db_ids != b.val_db_ids

    def test_val_never_takes_every_database(self):
        examples, facts = corpus(n_dbs=3, per_db=5)
        plan = plan_splits(examples, facts, val_size=10_000, seed=0)
        assert len(plan.train_ids) > 0

    def test_zero_val_keeps_everything_for_training(self):
        examples, facts = corpus(n_dbs=4, per_db=5)
        plan = plan_splits(examples, facts, val_size=0)
        assert plan.val_ids == ()
        assert len(plan.train_ids) == 20


class TestRandomStrategy:
    def test_random_split_hits_the_exact_size(self):
        examples, facts = corpus()
        plan = plan_splits(examples, facts, val_size=50, strategy="random", seed=0)
        assert len(plan.val_ids) == 50

    def test_random_split_shares_databases(self):
        """Documents why it is not the default."""
        examples, facts = corpus()
        plan = plan_splits(examples, facts, val_size=50, strategy="random", seed=0)

        by_id = {e.question_id: e for e in examples}
        train_dbs = {by_id[i].db_id for i in plan.train_ids}
        val_dbs = {by_id[i].db_id for i in plan.val_ids}
        assert train_dbs & val_dbs


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        plan_splits([], [], strategy="kfold")


def test_negative_val_size_raises():
    with pytest.raises(ValueError, match="val_size"):
        plan_splits([], [], val_size=-1)


def test_criteria_are_recorded_for_the_manifest():
    examples, facts = corpus(n_dbs=4, per_db=5)
    plan = plan_splits(examples, facts, val_size=5, seed=3, max_gold_seconds=8.0)

    assert plan.criteria["seed"] == 3
    assert plan.criteria["max_gold_seconds"] == 8.0
    assert plan.criteria["strategy"] == "db_disjoint"
    assert plan.criteria["n_input"] == 20
