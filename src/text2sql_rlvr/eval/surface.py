"""Measure how much room the official metric leaves for result-shape tricks.

This is computed from gold alone, before any model exists. The official BIRD
check is ``set(pred) == set(gold)``, so it can be satisfied without answering
the question in exactly two ways:

* gold returns nothing, and any always-empty query matches it;
* gold contains duplicate rows, and a de-duplicated answer matches it.

Everything else the set comparison lets through requires actually producing the
right distinct values. So the size of those two groups *is* the exploitable
surface, and it is worth knowing before betting an experiment on it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.rewards.canonical import canon_rows
from text2sql_rlvr.rewards.sandbox import OK, ExecResult
from text2sql_rlvr.sql import has_top_level_order_by


@dataclass(frozen=True)
class GoldFacts:
    """What one gold query's result looks like, independent of any prediction."""

    question_id: int
    db_id: str
    difficulty: str | None
    status: str
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    has_top_level_order_by: bool
    elapsed_s: float

    @property
    def empty(self) -> bool:
        return self.status == OK and self.n_rows == 0

    @property
    def has_duplicates(self) -> bool:
        return self.n_duplicate_rows > 0

    @property
    def exploitable(self) -> bool:
        """True if the official set comparison can be satisfied without answering."""
        return self.status == OK and (self.empty or self.has_duplicates)


def gold_facts(example: BirdExample, result: ExecResult) -> GoldFacts:
    """Describe one executed gold query."""
    duplicates = 0
    if result.ok and result.rows:
        counts = Counter(canon_rows(result.rows))
        duplicates = sum(count - 1 for count in counts.values() if count > 1)

    return GoldFacts(
        question_id=example.question_id,
        db_id=example.db_id,
        difficulty=example.difficulty,
        status=result.status,
        n_rows=result.n_rows,
        n_cols=len(result.columns),
        n_duplicate_rows=duplicates,
        has_top_level_order_by=has_top_level_order_by(example.gold_sql),
        elapsed_s=round(result.elapsed_s, 3),
    )


@dataclass(frozen=True)
class GoldSurface:
    """Aggregate view of the exploitable surface across a split."""

    n: int
    n_executable: int
    n_empty: int
    n_with_duplicates: int
    n_exploitable: int
    n_single_row: int
    n_single_column: int
    n_ordered: int
    exploitable_ids: tuple[int, ...]
    failed_ids: tuple[int, ...]

    def rate(self, count: int) -> float:
        return round(100.0 * count / self.n, 2) if self.n else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "n_executable": self.n_executable,
            "n_empty": self.n_empty,
            "n_with_duplicates": self.n_with_duplicates,
            "n_exploitable": self.n_exploitable,
            "exploitable_rate": self.rate(self.n_exploitable),
            "n_single_row": self.n_single_row,
            "n_single_column": self.n_single_column,
            "n_ordered": self.n_ordered,
        }


def summarise(facts: Iterable[GoldFacts]) -> GoldSurface:
    """Aggregate per-question facts into the numbers that drive the plan."""
    items: Sequence[GoldFacts] = list(facts)
    executable = [f for f in items if f.status == OK]
    exploitable = [f for f in executable if f.exploitable]

    return GoldSurface(
        n=len(items),
        n_executable=len(executable),
        n_empty=sum(1 for f in executable if f.empty),
        n_with_duplicates=sum(1 for f in executable if f.has_duplicates),
        n_exploitable=len(exploitable),
        n_single_row=sum(1 for f in executable if f.n_rows == 1),
        n_single_column=sum(1 for f in executable if f.n_cols == 1),
        n_ordered=sum(1 for f in executable if f.has_top_level_order_by),
        exploitable_ids=tuple(f.question_id for f in exploitable),
        failed_ids=tuple(f.question_id for f in items if f.status != OK),
    )
