"""Score every sample of a k-sample generation run, not just the first one.

Single-sample evaluation answers "is the model right?". Preference mining needs
a different question: "on this question, is the model *sometimes* right?" Only
those questions produce a pair of its own answers -- one to prefer, one to push
down -- and their number is what decides whether a sampling run was worth its
GPU time. So every sample is executed and compared against gold, and the
questions are then sorted into three buckets: all correct (nothing to prefer),
mixed (a pair), all wrong (a pair only if gold is allowed to stand in).

The scoring itself is :func:`text2sql_rlvr.eval.evaluate`, called once per
sample index with a shared executor. Reusing it is the point: a second
implementation of "did this query answer the question" would be a second thing
that can disagree with the official metric. Gold is executed repeatedly this
way, but the executor caches by query, so the repeats are cache hits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from text2sql_rlvr.data.bird import BirdExample, BirdSplit
from text2sql_rlvr.eval.execution_accuracy import ExampleOutcome, evaluate
from text2sql_rlvr.rewards.compare import DEFAULT_ORDER_POLICY
from text2sql_rlvr.rewards.sandbox import SqlExecutor


@dataclass(frozen=True)
class SamplingReport:
    """How a k-sample run scored, and how much of it is minable."""

    n_questions: int
    n_samples_total: int
    k_max: int
    pass_at_1: float
    pass_at_k: float
    mean_pass_rate: float
    n_all_correct: int
    n_mixed: int
    n_all_wrong: int
    outcomes: dict[int, tuple[ExampleOutcome, ...]] = field(default_factory=dict)

    def metrics(self) -> dict[str, float | int]:
        """Flat metric dict for the ledger."""
        return {
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "mean_pass_rate": self.mean_pass_rate,
            "k_max": self.k_max,
            "n_questions": self.n_questions,
            "n_samples_total": self.n_samples_total,
            "n_all_correct": self.n_all_correct,
            "n_mixed": self.n_mixed,
            "n_all_wrong": self.n_all_wrong,
        }


def _rate(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def score_samples(
    examples: Sequence[BirdExample],
    samples: Mapping[int, Sequence[str]],
    split: BirdSplit,
    *,
    executor: SqlExecutor | None = None,
    order_policy: str = DEFAULT_ORDER_POLICY,
    n_workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> SamplingReport:
    """Execute every sample of every question and bucket the questions.

    ``samples`` maps question_id to that question's SQL strings, already
    extracted from the completions. Questions with no samples are skipped rather
    than counted as failures: an absent question was not asked, and folding it
    into a pass rate would quietly understate the model.

    ``pass_at_1`` is the first sample only, which is the closest thing to a
    temperature-0 run; ``pass_at_k`` is the share of questions solved at least
    once. The gap between them is what sampling buys.
    """
    owned = executor is None
    sql_executor = executor or SqlExecutor()
    by_id = {e.question_id: e for e in examples if samples.get(e.question_id)}
    k_max = max((len(s) for s in samples.values()), default=0)

    collected: dict[int, list[ExampleOutcome]] = {qid: [] for qid in by_id}
    try:
        for index in range(k_max):
            predictions = {
                qid: samples[qid][index]
                for qid in by_id
                if index < len(samples[qid])
            }
            report = evaluate(
                [by_id[qid] for qid in predictions],
                predictions,
                split,
                executor=sql_executor,
                order_policy=order_policy,
                extract=False,
                n_workers=n_workers,
            )
            for outcome in report.outcomes:
                collected[outcome.question_id].append(outcome)
            if on_progress:
                on_progress(index + 1, k_max)
    finally:
        if owned:
            sql_executor.close()

    n_all_correct = n_mixed = n_all_wrong = 0
    first_correct = 0
    pass_rates: list[float] = []
    for outcomes in collected.values():
        correct = sum(1 for o in outcomes if o.official)
        pass_rates.append(correct / len(outcomes))
        first_correct += int(outcomes[0].official)
        if correct == len(outcomes):
            n_all_correct += 1
        elif correct:
            n_mixed += 1
        else:
            n_all_wrong += 1

    n_questions = len(collected)
    return SamplingReport(
        n_questions=n_questions,
        n_samples_total=sum(len(o) for o in collected.values()),
        k_max=k_max,
        pass_at_1=_rate(first_correct, n_questions),
        pass_at_k=_rate(n_questions - n_all_wrong, n_questions),
        mean_pass_rate=round(100.0 * sum(pass_rates) / n_questions, 2) if n_questions else 0.0,
        n_all_correct=n_all_correct,
        n_mixed=n_mixed,
        n_all_wrong=n_all_wrong,
        outcomes={qid: tuple(o) for qid, o in collected.items()},
    )
