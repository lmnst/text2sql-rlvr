"""Turn the raw training split into a training set and a validation set.

Two jobs, both of which change what the final numbers mean:

**Filtering.** A question whose gold SQL does not execute can never produce a
positive reward. During RL every rollout for that prompt scores zero, the group
has no reward spread, and it contributes nothing to the policy gradient while
still costing generation time. Those questions are dropped.

**Splitting.** Validation exists to choose checkpoints and hyper-parameters
without touching dev. For that to work, val has to be hard in the same way dev
is hard. BIRD's dev databases do not appear in train at all, so dev measures
generalisation to *unseen schemas*. A randomly drawn val would share databases
with the training portion, and would therefore report a number that flatters
the model relative to dev. The default is a database-disjoint split.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.eval.surface import GoldFacts
from text2sql_rlvr.rewards.sandbox import OK

VAL_STRATEGIES = ("db_disjoint", "random")

#: Exclusion reasons, in the order they are checked.
NO_GOLD = "no_gold_sql"
NO_FACTS = "gold_not_measured"
NOT_EXECUTABLE = "gold_not_executable"
TOO_SLOW = "gold_too_slow"


@dataclass(frozen=True)
class SplitPlan:
    """Which question ids go where, and why the rest went nowhere."""

    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    excluded: dict[str, tuple[int, ...]]
    val_db_ids: tuple[str, ...]
    criteria: dict[str, object] = field(default_factory=dict)

    @property
    def n_excluded(self) -> int:
        return sum(len(ids) for ids in self.excluded.values())

    def summary(self) -> dict[str, object]:
        return {
            "n_train": len(self.train_ids),
            "n_val": len(self.val_ids),
            "n_excluded": self.n_excluded,
            "excluded_by_reason": {k: len(v) for k, v in self.excluded.items() if v},
            "n_val_databases": len(self.val_db_ids),
            "criteria": self.criteria,
        }


def _exclusion_reason(
    example: BirdExample, facts: Mapping[int, GoldFacts], max_gold_seconds: float
) -> str | None:
    if not example.gold_sql:
        return NO_GOLD
    fact = facts.get(example.question_id)
    if fact is None:
        return NO_FACTS
    if fact.status != OK:
        return NOT_EXECUTABLE
    if max_gold_seconds > 0 and fact.elapsed_s > max_gold_seconds:
        return TOO_SLOW
    return None


def _db_disjoint_val(
    eligible: Sequence[BirdExample], target: int, rng: random.Random
) -> set[str]:
    """Pick whole databases for val until it is at least ``target`` questions."""
    by_db: dict[str, int] = Counter(e.db_id for e in eligible)
    db_ids = sorted(by_db)
    rng.shuffle(db_ids)

    chosen: set[str] = set()
    taken = 0
    for db_id in db_ids:
        if taken >= target:
            break
        # Never let val swallow so much that training has nothing left.
        if len(chosen) + 1 >= len(db_ids):
            break
        chosen.add(db_id)
        taken += by_db[db_id]
    return chosen


def plan_splits(
    examples: Sequence[BirdExample],
    facts: Sequence[GoldFacts] | Mapping[int, GoldFacts],
    *,
    val_size: int = 500,
    strategy: str = "db_disjoint",
    seed: int = 0,
    max_gold_seconds: float = 10.0,
) -> SplitPlan:
    """Decide the train/val split without writing anything.

    ``max_gold_seconds`` drops questions whose gold was slower than the timeout
    training will use. Set it to 0 to keep them. The recorded timings come from a
    parallel run, so this filter errs on the side of dropping a few extra.
    """
    if strategy not in VAL_STRATEGIES:
        raise ValueError(f"strategy must be one of {VAL_STRATEGIES}, got {strategy!r}")
    if val_size < 0:
        raise ValueError(f"val_size must not be negative, got {val_size}")

    by_id = facts if isinstance(facts, Mapping) else {f.question_id: f for f in facts}

    eligible: list[BirdExample] = []
    excluded: dict[str, list[int]] = defaultdict(list)
    for example in examples:
        reason = _exclusion_reason(example, by_id, max_gold_seconds)
        if reason:
            excluded[reason].append(example.question_id)
        else:
            eligible.append(example)

    rng = random.Random(seed)
    if strategy == "db_disjoint":
        val_dbs = _db_disjoint_val(eligible, val_size, rng)
        val = [e for e in eligible if e.db_id in val_dbs]
        train = [e for e in eligible if e.db_id not in val_dbs]
    else:
        shuffled = list(eligible)
        rng.shuffle(shuffled)
        val, train = shuffled[:val_size], shuffled[val_size:]
        val_dbs = {e.db_id for e in val}

    return SplitPlan(
        train_ids=tuple(e.question_id for e in train),
        val_ids=tuple(e.question_id for e in val),
        excluded={reason: tuple(ids) for reason, ids in sorted(excluded.items())},
        val_db_ids=tuple(sorted(val_dbs)),
        criteria={
            "strategy": strategy,
            "seed": seed,
            "val_size_requested": val_size,
            "max_gold_seconds": max_gold_seconds,
            "n_input": len(examples),
        },
    )
