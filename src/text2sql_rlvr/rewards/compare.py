"""Two comparison verdicts per example, on purpose.

``official`` reproduces the BIRD evaluator bit for bit -- ``set(pred) == set(gold)``
on the raw tuples. It is the number that goes in any table that claims to report
Execution Accuracy, and it must not be quietly "improved".

``strict`` is our own verifier: same column count, same rows *with multiplicity*,
numeric values compared after canonicalisation. It is what the RL reward should
use, because the official set comparison is trivially satisfiable in ways that
have nothing to do with answering the question.

Keeping both and recording where they disagree is the point. That gap is the
measurement, not a nuisance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from text2sql_rlvr.rewards.canonical import DEFAULT_FLOAT_SIG, canon_rows
from text2sql_rlvr.rewards.sandbox import ExecResult
from text2sql_rlvr.sql import has_top_level_order_by

# Reasons a prediction fails the strict verifier.
GOLD_FAILED = "gold_failed"
PRED_FAILED = "pred_failed"
TRUNCATED = "truncated"
COLUMN_COUNT = "column_count"
ROW_COUNT = "row_count"
ROW_VALUES = "row_values"
ROW_ORDER = "row_order"

#: Order policies for the strict verifier.
#:
#: ``ignore``        - multiset comparison only. Default.
#: ``gold_order_by`` - additionally require identical row order when the gold
#:                     query has a top-level ORDER BY.
#: ``always``        - always require identical row order.
ORDER_POLICIES = ("ignore", "gold_order_by", "always")

#: Why ``ignore`` is the default: a bare ``ORDER BY`` with ties leaves the order
#: of tied rows unspecified, so enforcing sequence equality marks correct SQL
#: wrong at a rate that depends on the storage layout. During RL that is reward
#: noise, and false negatives corrupt the advantage estimate more than mild
#: leniency does. The ``ORDER BY ... LIMIT k`` pattern that BIRD actually leans
#: on is already covered, because the limit makes the returned rows differ.
DEFAULT_ORDER_POLICY = "ignore"


@dataclass(frozen=True)
class Comparison:
    """Both verdicts plus enough context to analyse the gap between them."""

    official: bool
    strict: bool
    reason: str | None = None
    gold_has_order_by: bool = False
    order_matches: bool = False
    pred_n_rows: int = 0
    gold_n_rows: int = 0
    pred_n_cols: int = 0
    gold_n_cols: int = 0

    @property
    def gold_empty(self) -> bool:
        return self.gold_n_rows == 0

    @property
    def disagrees(self) -> bool:
        """True when the official metric credits a prediction the strict one rejects."""
        return self.official and not self.strict


def compare_official(pred: ExecResult, gold: ExecResult) -> bool:
    """Reproduce BIRD's Execution Accuracy check exactly.

    Note what this does *not* do: it ignores duplicate rows, ignores row order,
    and treats any two empty result sets as equal regardless of column count.
    """
    if not (pred.ok and gold.ok):
        return False
    return set(pred.rows) == set(gold.rows)


def compare(
    pred: ExecResult,
    gold: ExecResult,
    *,
    gold_sql: str = "",
    order_policy: str = DEFAULT_ORDER_POLICY,
    float_sig: int = DEFAULT_FLOAT_SIG,
) -> Comparison:
    """Evaluate one prediction under both the official and the strict verifier."""
    if order_policy not in ORDER_POLICIES:
        raise ValueError(f"order_policy must be one of {ORDER_POLICIES}, got {order_policy!r}")

    gold_order_by = has_top_level_order_by(gold_sql) if gold_sql else False
    official = compare_official(pred, gold)

    base = dict(
        official=official,
        gold_has_order_by=gold_order_by,
        pred_n_rows=pred.n_rows,
        gold_n_rows=gold.n_rows,
        pred_n_cols=len(pred.columns),
        gold_n_cols=len(gold.columns),
    )

    if not gold.ok:
        return Comparison(strict=False, reason=GOLD_FAILED, **base)
    if not pred.ok:
        return Comparison(strict=False, reason=PRED_FAILED, **base)
    if pred.truncated or gold.truncated:
        # A capped result set cannot be verified either way; refusing to guess
        # keeps a wrong reward out of training.
        return Comparison(strict=False, reason=TRUNCATED, **base)
    if len(pred.columns) != len(gold.columns):
        return Comparison(strict=False, reason=COLUMN_COUNT, **base)

    pred_rows = canon_rows(pred.rows, float_sig=float_sig)
    gold_rows = canon_rows(gold.rows, float_sig=float_sig)
    order_matches = pred_rows == gold_rows

    if Counter(pred_rows) != Counter(gold_rows):
        reason = ROW_COUNT if len(pred_rows) != len(gold_rows) else ROW_VALUES
        return Comparison(strict=False, reason=reason, order_matches=False, **base)

    enforce_order = order_policy == "always" or (
        order_policy == "gold_order_by" and gold_order_by
    )
    if enforce_order and not order_matches:
        return Comparison(strict=False, reason=ROW_ORDER, order_matches=False, **base)

    return Comparison(strict=True, reason=None, order_matches=order_matches, **base)
