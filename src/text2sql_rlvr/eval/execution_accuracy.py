"""Execution Accuracy, reported under both verifiers at once.

``official_ex`` is the headline number and follows BIRD's definition exactly.
``strict_ex`` is the same run scored by our verifier. Reporting them together
costs nothing extra -- gold and prediction are already executed -- and the gap
between them is the quantity the reward-hacking analysis is about.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from text2sql_rlvr.data.bird import BirdExample, BirdSplit
from text2sql_rlvr.rewards.compare import DEFAULT_ORDER_POLICY, compare
from text2sql_rlvr.rewards.sandbox import REJECTED, ExecResult, SqlExecutor
from text2sql_rlvr.sql import extract_sql


@dataclass(frozen=True)
class ExampleOutcome:
    """Everything needed to re-derive any aggregate, or to slice the errors later."""

    question_id: int
    db_id: str
    difficulty: str | None
    pred_sql: str
    gold_sql: str
    official: bool
    strict: bool
    reason: str | None
    pred_status: str
    gold_status: str
    pred_n_rows: int
    gold_n_rows: int
    pred_n_cols: int
    gold_n_cols: int
    gold_empty: bool
    gold_has_order_by: bool
    order_matches: bool
    pred_error: str | None
    elapsed_s: float


@dataclass(frozen=True)
class EvalReport:
    split: str
    order_policy: str
    n: int
    official_ex: float
    strict_ex: float
    by_difficulty: dict[str, dict[str, float | int]]
    pred_status_counts: dict[str, int]
    gold_status_counts: dict[str, int]
    strict_reason_counts: dict[str, int]
    n_official_not_strict: int
    n_strict_not_official: int
    n_gold_empty: int
    n_gold_empty_official_pass: int
    n_missing_predictions: int
    n_unparsed: int
    executor_stats: dict[str, int] = field(default_factory=dict)
    outcomes: tuple[ExampleOutcome, ...] = ()

    def metrics(self) -> dict[str, float | int]:
        """Flat metric dict for the ledger."""
        return {
            "official_ex": self.official_ex,
            "strict_ex": self.strict_ex,
            "n_official_not_strict": self.n_official_not_strict,
            "n_strict_not_official": self.n_strict_not_official,
            "n_gold_empty": self.n_gold_empty,
            "n_gold_empty_official_pass": self.n_gold_empty_official_pass,
            "n_missing_predictions": self.n_missing_predictions,
            "n_unparsed": self.n_unparsed,
            **{f"official_ex_{k}": v["official_ex"] for k, v in self.by_difficulty.items()},
            **{f"n_{k}": v["n"] for k, v in self.by_difficulty.items()},
        }

    def write_outcomes(self, path: str | Path) -> Path:
        """Dump per-example outcomes so error analysis never needs a re-run."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for outcome in self.outcomes:
                handle.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")
        return target


def _rate(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def evaluate(
    examples: Sequence[BirdExample],
    predictions: Mapping[int, str],
    split: BirdSplit,
    *,
    executor: SqlExecutor | None = None,
    order_policy: str = DEFAULT_ORDER_POLICY,
    extract: bool = True,
    n_workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> EvalReport:
    """Score ``predictions`` against gold SQL.

    ``predictions`` maps question_id to the raw model completion; set
    ``extract=False`` if it already contains bare SQL.
    """
    owned = executor is None
    sql_executor = executor or SqlExecutor()

    def score(example: BirdExample) -> ExampleOutcome:
        raw = predictions.get(example.question_id) or ""
        pred_sql = extract_sql(raw) if extract else raw.strip()

        db_path = split.db_path(example.db_id)
        pred = (
            sql_executor.execute(db_path, pred_sql)
            if pred_sql
            else ExecResult(REJECTED, error="no SQL in completion")
        )
        gold = (
            sql_executor.execute(db_path, example.gold_sql)
            if example.gold_sql
            else ExecResult(REJECTED, error="no gold SQL")
        )
        verdict = compare(pred, gold, gold_sql=example.gold_sql, order_policy=order_policy)

        return ExampleOutcome(
            question_id=example.question_id,
            db_id=example.db_id,
            difficulty=example.difficulty,
            pred_sql=pred_sql,
            gold_sql=example.gold_sql,
            official=verdict.official,
            strict=verdict.strict,
            reason=verdict.reason,
            pred_status=pred.status,
            gold_status=gold.status,
            pred_n_rows=verdict.pred_n_rows,
            gold_n_rows=verdict.gold_n_rows,
            pred_n_cols=verdict.pred_n_cols,
            gold_n_cols=verdict.gold_n_cols,
            gold_empty=verdict.gold_empty,
            gold_has_order_by=verdict.gold_has_order_by,
            order_matches=verdict.order_matches,
            pred_error=pred.error,
            elapsed_s=round(pred.elapsed_s + gold.elapsed_s, 4),
        )

    try:
        if n_workers <= 1:
            outcomes = []
            for index, example in enumerate(examples, start=1):
                outcomes.append(score(example))
                if on_progress:
                    on_progress(index, len(examples))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                outcomes = []
                for index, outcome in enumerate(pool.map(score, examples), start=1):
                    outcomes.append(outcome)
                    if on_progress:
                        on_progress(index, len(examples))
        stats = sql_executor.stats.as_dict()
    finally:
        if owned:
            sql_executor.close()

    total = len(outcomes)
    # Derived after the fact rather than counted inside the worker: incrementing
    # a shared int from a thread pool is a race.
    missing = sum(1 for e in examples if not predictions.get(e.question_id))
    unparsed = sum(
        1
        for o in outcomes
        if predictions.get(o.question_id) and not o.pred_sql
    )

    by_difficulty: dict[str, dict[str, float | int]] = {}
    buckets: dict[str, list[ExampleOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.difficulty or "unknown", []).append(outcome)
    for name, bucket in sorted(buckets.items()):
        by_difficulty[name] = {
            "n": len(bucket),
            "official_ex": _rate(sum(o.official for o in bucket), len(bucket)),
            "strict_ex": _rate(sum(o.strict for o in bucket), len(bucket)),
        }

    return EvalReport(
        split=split.name,
        order_policy=order_policy,
        n=total,
        official_ex=_rate(sum(o.official for o in outcomes), total),
        strict_ex=_rate(sum(o.strict for o in outcomes), total),
        by_difficulty=by_difficulty,
        pred_status_counts=dict(Counter(o.pred_status for o in outcomes)),
        gold_status_counts=dict(Counter(o.gold_status for o in outcomes)),
        strict_reason_counts=dict(Counter(o.reason for o in outcomes if o.reason)),
        n_official_not_strict=sum(1 for o in outcomes if o.official and not o.strict),
        n_strict_not_official=sum(1 for o in outcomes if o.strict and not o.official),
        n_gold_empty=sum(1 for o in outcomes if o.gold_empty),
        n_gold_empty_official_pass=sum(1 for o in outcomes if o.gold_empty and o.official),
        n_missing_predictions=missing,
        n_unparsed=unparsed,
        executor_stats=stats,
        outcomes=tuple(outcomes),
    )
