"""Preference pairs for DPO, mined from execution results.

SFT shows the model a gold query and asks it to imitate. It never shows the
model what it *would have written*, so it cannot teach the distinction that
actually fails: writing ``COUNT(*)`` where the question needs
``COUNT(DISTINCT ...)``, or leaving off the ``LIMIT 1`` that makes "the most
popular film" a single row. A preference pair puts both side by side, which is
the contrast SFT is missing.

Nothing here is annotated. Both sides are queries that were executed on the
real database and compared against gold, so "this one is wrong" is a fact the
sandbox established rather than a label someone assigned.

Where the two sides come from matters more than it looks. Pairing BIRD's gold
against the model's answer leaks a surface feature: gold is written in a house
style (82.8% of the pairs mined that way use ``AS T1`` aliases, against 5.5% on
the model's side), and a preference model will happily learn the style instead
of the semantics. Sampling the same question several times and pairing the
model's own correct answer against its own wrong one removes that difference by
construction, which is what :func:`mine_pair` does. Falling back to gold is kept
for questions the model never gets right, and capped.

The pairs are only as useful as the policy that produced them. Negatives from
the model you are about to train are worth more than negatives from some other
model, so regenerate them after each round of training rather than reusing an
old file.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.prompt import PromptConfig, build_messages
from text2sql_rlvr.data.sft import format_target
from text2sql_rlvr.sql import extract_sql


def normalise_sql(sql: str) -> str:
    """Collapse whitespace and case so trivially different spellings compare equal."""
    # rstrip both characters: "SELECT a FROM t ;" leaves a trailing space otherwise
    return " ".join(sql.split()).rstrip("; ").casefold()


def pair_is_usable(gold_sql: str, rejected_sql: str) -> bool:
    """Whether the two queries form a pair worth training on.

    Both sides have to exist and differ by more than formatting; a pair whose
    two halves are the same string teaches nothing and its gradient is exactly
    zero, so it would only dilute the batch.
    """
    if not gold_sql.strip() or not rejected_sql.strip():
        return False
    return normalise_sql(gold_sql) != normalise_sql(rejected_sql)


def build_preference_record(
    example: BirdExample,
    schema_text: str,
    rejected_sql: str,
    config: PromptConfig | None = None,
    chosen_sql: str | None = None,
) -> dict[str, object]:
    """One preference pair in the conversational shape trainers expect.

    The prompt is built by the same :func:`build_messages` generation uses, so
    the sequence DPO optimises is the sequence the model is later asked to
    produce. Both responses are rendered by :func:`format_target`, the same
    fenced block SFT trains and the evaluator parses.

    ``chosen_sql`` defaults to gold. Passing one of the model own correct
    samples instead is the preferred form: see :func:`mine_pair`.
    """
    if not example.gold_sql:
        raise ValueError(f"question {example.question_id} has no gold SQL")
    chosen_sql = chosen_sql or example.gold_sql
    if not pair_is_usable(chosen_sql, rejected_sql):
        raise ValueError(
            f"question {example.question_id}: rejected query is missing or "
            "identical to the chosen one after normalisation"
        )

    chosen = format_target(chosen_sql)
    rejected = format_target(rejected_sql)
    for side, text in (("chosen", chosen), ("rejected", rejected)):
        if extract_sql(text) == "":
            raise ValueError(f"question {example.question_id}: {side} is not parsable back")

    return {
        "question_id": example.question_id,
        "db_id": example.db_id,
        "prompt": list(build_messages(example, schema_text, config or PromptConfig())),
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def pair_report(
    records: Sequence[Mapping[str, object]], outcomes: Mapping[int, Mapping[str, object]]
) -> dict[str, object]:
    """What the pair set contains, for the dataset manifest."""
    per_db = Counter(str(r["db_id"]) for r in records)
    reasons = Counter(
        str(outcomes[int(r["question_id"])].get("reason"))
        for r in records
        if int(r["question_id"]) in outcomes
    )
    statuses = Counter(
        str(outcomes[int(r["question_id"])].get("pred_status"))
        for r in records
        if int(r["question_id"]) in outcomes
    )
    counts = sorted(per_db.values())
    return {
        "n": len(records),
        "n_databases": len(per_db),
        "per_db_min": counts[0] if counts else 0,
        "per_db_max": counts[-1] if counts else 0,
        "rejected_by_reason": dict(reasons.most_common()),
        "rejected_by_exec_status": dict(statuses.most_common()),
        "questions_per_db": dict(sorted(per_db.items())),
    }


#: A query in BIRD's gold house style: ``FROM tbl AS T1 JOIN other AS T2``.
#: Crude on purpose -- this is a surface-feature audit, not a parser. It exists
#: so that a style difference between the two sides of the pairs can be
#: *measured* before training instead of discovered afterwards in the samples.
_ALIAS_PATTERN = re.compile(r"\bT\d+\b", re.IGNORECASE)


def uses_gold_style_alias(sql: str) -> bool:
    return bool(_ALIAS_PATTERN.search(sql))


#: Where the chosen side came from. ``mined`` means the model itself produced
#: it; ``gold_fallback`` means it never did and BIRD's answer was used.
MINED = "mined"
GOLD_FALLBACK = "gold_fallback"


@dataclass(frozen=True)
class MinedPair:
    """One preference pair, with enough provenance to audit the whole set."""

    question_id: int
    db_id: str
    chosen_sql: str
    rejected_sql: str
    source: str
    n_samples: int
    n_correct: int
    chosen_count: int
    rejected_count: int


def _mode(sqls: Sequence[str]) -> tuple[str, int]:
    """The most frequent query, compared after normalisation, spelled as first seen.

    Ties keep first appearance: ``Counter.most_common`` sorts stably, so the
    result does not depend on dict iteration luck and the same samples always
    mine the same pair.
    """
    counts: Counter[str] = Counter()
    spelling: dict[str, str] = {}
    for sql in sqls:
        key = normalise_sql(sql)
        counts[key] += 1
        spelling.setdefault(key, sql)
    key, count = counts.most_common(1)[0]
    return spelling[key], count


def mine_pair(
    samples: Sequence[Mapping[str, object]],
    gold_sql: str,
    *,
    question_id: int,
    db_id: str,
) -> MinedPair | None:
    """Turn one question's k samples into at most one preference pair.

    Each sample is a mapping with ``sql`` and ``official``. The rules:

    * some right and some wrong -- chosen is the most frequent correct query,
      rejected the most frequent wrong one. The mode rather than an arbitrary
      one, because the mistake the model makes *most often* is the one worth
      pushing down.
    * all wrong -- chosen falls back to gold. Those pairs carry the style
      difference this module's docstring describes, so the caller caps them.
    * all right -- no pair. There is nothing to prefer.

    One pair per question, never k of them: a question the model gets wrong in
    eight different ways would otherwise outweigh a question it nearly has.

    A sample with no parsable SQL counts as wrong but cannot represent the
    rejected side -- an empty rejected query teaches the model to avoid writing
    nothing, which it was not about to do anyway.
    """
    if not samples:
        return None
    correct = [str(s["sql"]) for s in samples if s["official"] and str(s["sql"]).strip()]
    wrong = [str(s["sql"]) for s in samples if not s["official"] and str(s["sql"]).strip()]
    n_correct = sum(1 for s in samples if s["official"])
    if not wrong:
        return None

    rejected_sql, rejected_count = _mode(wrong)
    if correct:
        chosen_sql, chosen_count = _mode(correct)
        source = MINED
    else:
        chosen_sql, chosen_count, source = gold_sql, 0, GOLD_FALLBACK

    if not pair_is_usable(chosen_sql, rejected_sql):
        return None
    return MinedPair(
        question_id=question_id,
        db_id=db_id,
        chosen_sql=chosen_sql,
        rejected_sql=rejected_sql,
        source=source,
        n_samples=len(samples),
        n_correct=n_correct,
        chosen_count=chosen_count,
        rejected_count=rejected_count,
    )


def cap_gold_fallback(
    pairs: Sequence[MinedPair], *, max_fraction: float = 0.25, seed: int = 0
) -> list[MinedPair]:
    """Keep every mined pair, and at most ``max_fraction`` of gold-fallback ones.

    The cap is on the *share of the result*, so with m mined pairs the budget is
    ``m * f / (1 - f)``. Capping at ``f * len(all pairs)`` instead would leave
    the fallback bucket above its intended share whenever it is the larger one.

    Which fallback pairs survive is a seeded random choice. Ordering by database
    was the alternative, but the mined pairs already set the database balance,
    and by construction the fallback bucket is the minority of the result.
    """
    if not 0.0 <= max_fraction < 1.0:
        raise ValueError(f"max_fraction must be within [0, 1), got {max_fraction}")
    mined = [p for p in pairs if p.source == MINED]
    fallback = [p for p in pairs if p.source == GOLD_FALLBACK]
    budget = int(len(mined) * max_fraction / (1 - max_fraction))
    if len(fallback) > budget:
        fallback = random.Random(seed).sample(fallback, budget)
    keep = {id(p) for p in mined} | {id(p) for p in fallback}
    return [p for p in pairs if id(p) in keep]


def mined_pair_report(pairs: Sequence[MinedPair]) -> dict[str, object]:
    """Provenance and surface-feature audit, for the dataset manifest.

    ``alias_rate_chosen`` against ``alias_rate_rejected`` is the number that
    matters: if the two sides differ a lot, DPO can reach a low loss by learning
    to write aliases, and the resulting model will look trained without being
    any better. Compare them before training, and compare them again on what the
    trained model writes.
    """
    if not pairs:
        return {"n": 0}

    def rate(flags: Sequence[bool]) -> float:
        return round(sum(flags) / len(flags), 4) if flags else 0.0

    sources = Counter(p.source for p in pairs)
    mined = [p for p in pairs if p.source == MINED]
    return {
        "n": len(pairs),
        "n_databases": len({p.db_id for p in pairs}),
        "by_source": dict(sources.most_common()),
        "gold_fallback_share": round(sources[GOLD_FALLBACK] / len(pairs), 4),
        "alias_rate_chosen": rate([uses_gold_style_alias(p.chosen_sql) for p in pairs]),
        "alias_rate_rejected": rate([uses_gold_style_alias(p.rejected_sql) for p in pairs]),
        "alias_rate_chosen_mined_only": rate(
            [uses_gold_style_alias(p.chosen_sql) for p in mined]
        ),
        "mean_correct_samples": round(sum(p.n_correct for p in pairs) / len(pairs), 3),
        "mean_rejected_mode_count": round(
            sum(p.rejected_count for p in pairs) / len(pairs), 3
        ),
    }
