"""Build supervised fine-tuning examples.

One rule dominates everything else here: **the prompt used for training must be
byte-identical to the prompt used at evaluation time.** If they drift, the model
is optimised for an input distribution it never sees again, and the resulting
score says nothing about the thing that was trained. So the messages are built
by the same :func:`build_messages` the generation script calls, with the same
:class:`PromptConfig`, and the config is written into the dataset manifest.

The second rule is that the target must survive the round trip through
:func:`extract_sql`. The model is trained to emit a fenced block; evaluation
parses a fenced block. A target the parser cannot read back is a silent
mismatch that shows up much later as a mysteriously low score.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.prompt import PromptConfig, build_messages
from text2sql_rlvr.sql import extract_sql

#: How training questions are chosen out of an evaluated pool.
#:
#: ``all`` keeps everything. ``random`` is the control: a plain sample, used to
#: show whether a targeted set beats an equally large arbitrary one. ``failure``
#: is the targeted policy: questions the current model answered wrong, plus a
#: ``replay_fraction`` share of questions it answered right so the model is not
#: trained exclusively on its own failure distribution.
SELECTION_POLICIES = ("all", "random", "failure")


def _round_robin(
    ids_by_db: Mapping[str, list[int]], limit: int, cap_per_db: int
) -> list[int]:
    """Take ids one database at a time, so no large database dominates."""
    remaining = {db: list(ids) for db, ids in ids_by_db.items() if ids}
    taken: list[int] = []
    per_db: dict[str, int] = defaultdict(int)
    while remaining and (limit <= 0 or len(taken) < limit):
        progressed = False
        for db in sorted(remaining):
            if limit > 0 and len(taken) >= limit:
                break
            if cap_per_db > 0 and per_db[db] >= cap_per_db:
                remaining.pop(db, None)
                continue
            taken.append(remaining[db].pop())
            per_db[db] += 1
            progressed = True
            if not remaining[db]:
                remaining.pop(db)
        if not progressed:
            break
    return taken


def select_sft_examples(
    outcomes: Sequence[Mapping[str, object]],
    *,
    policy: str = "all",
    size: int = 0,
    replay_fraction: float = 0.0,
    cap_per_db: int = 0,
    drop_gold_empty: bool = False,
    seed: int = 0,
) -> list[int]:
    """Choose which question ids to train on, given how a model scored on them.

    ``outcomes`` are the per-question records evaluate.py writes; only
    ``question_id``, ``db_id``, ``official`` and ``gold_empty`` are read.
    ``size`` of 0 keeps everything the policy allows. Selection is round-robin
    over databases: seeing many schemas matters more than seeing many questions
    from the one database that happens to be largest.

    ``drop_gold_empty`` skips questions whose gold SQL returns no rows. On BIRD
    train that is 265 of 8191; their gold teaches the model to answer with
    nothing, and several are simply mislabelled.
    """
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"policy must be one of {SELECTION_POLICIES}, got {policy!r}")
    if not 0.0 <= replay_fraction <= 1.0:
        raise ValueError(f"replay_fraction must be within [0, 1], got {replay_fraction}")
    if policy != "failure" and replay_fraction:
        raise ValueError("replay_fraction only applies to the 'failure' policy")

    rng = random.Random(seed)
    solved: dict[str, list[int]] = defaultdict(list)
    failed: dict[str, list[int]] = defaultdict(list)
    for record in outcomes:
        if drop_gold_empty and record.get("gold_empty"):
            continue
        target = solved if record["official"] else failed
        target[str(record["db_id"])].append(int(record["question_id"]))
    for group in (solved, failed):
        for ids in group.values():
            rng.shuffle(ids)

    if policy == "failure":
        n_replay = round(size * replay_fraction) if size else 0
        primary = _round_robin(failed, size - n_replay if size else 0, cap_per_db)
        replay = _round_robin(solved, n_replay, cap_per_db) if n_replay else []
        return sorted(primary + replay)

    everything: dict[str, list[int]] = defaultdict(list)
    for group in (solved, failed):
        for db, ids in group.items():
            everything[db].extend(ids)
    for ids in everything.values():
        rng.shuffle(ids)
    return sorted(_round_robin(everything, size, cap_per_db))


def selection_report(
    selected: Sequence[int], outcomes: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """What the selection actually contains, for the dataset manifest."""
    by_id = {int(r["question_id"]): r for r in outcomes}
    chosen = [by_id[i] for i in selected if i in by_id]
    per_db: dict[str, int] = defaultdict(int)
    for record in chosen:
        per_db[str(record["db_id"])] += 1
    counts = sorted(per_db.values())
    return {
        "n": len(selected),
        "n_databases": len(per_db),
        "n_from_failures": sum(1 for r in chosen if not r["official"]),
        "n_from_solved": sum(1 for r in chosen if r["official"]),
        "per_db_min": counts[0] if counts else 0,
        "per_db_max": counts[-1] if counts else 0,
        "questions_per_db": dict(sorted(per_db.items())),
    }


def format_target(gold_sql: str) -> str:
    """Render the assistant turn: exactly the fenced block the prompt asks for."""
    sql = " ".join(gold_sql.strip().rstrip(";").split())
    return f"```sql\n{sql}\n```"


@dataclass(frozen=True)
class SftRecord:
    """One training example, plus the fields needed to trace it back."""

    question_id: int
    db_id: str
    messages: list[dict[str, str]]
    n_prompt_chars: int
    n_target_chars: int

    def as_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "messages": self.messages,
        }


def build_sft_record(
    example: BirdExample,
    schema_text: str,
    config: PromptConfig | None = None,
) -> SftRecord:
    """Turn one BIRD question into a chat-format training example."""
    if not example.gold_sql:
        raise ValueError(f"question {example.question_id} has no gold SQL")

    config = config or PromptConfig()
    messages = list(build_messages(example, schema_text, config))
    target = format_target(example.gold_sql)

    # Cheap insurance against the two files drifting apart later.
    if extract_sql(target) == "":
        raise ValueError(f"target for question {example.question_id} is not parsable back")

    messages.append({"role": "assistant", "content": target})
    prompt_chars = sum(len(m["content"]) for m in messages[:-1])
    return SftRecord(
        question_id=example.question_id,
        db_id=example.db_id,
        messages=messages,
        n_prompt_chars=prompt_chars,
        n_target_chars=len(target),
    )


#: Characters per token, calibrated against real vLLM `usage.prompt_tokens`
#: rather than guessed: the measured ratio ranged from 3.79 (short prompts, more
#: natural language) to 5.57 (the 65-table works_cycles schema, where repeated
#: long identifiers tokenise efficiently). The initial guess of 3.3 overstated
#: token counts by up to 70%.
#:
#: The conservative end is used deliberately -- overestimating tokens sets a
#: budget that is too generous, which wastes a little memory; underestimating
#: silently drops or truncates examples.
CHARS_PER_TOKEN = 3.6


def length_report(records: Mapping[int, SftRecord] | list[SftRecord]) -> dict[str, object]:
    """Percentiles of total example length, and what to set the cutoff to.

    A trainer that silently truncates long examples cuts the *end* of the
    sequence -- which is the answer. Those examples then teach the model to
    produce nothing. Knowing the tail before training is cheaper than finding
    out from a flat loss curve.
    """
    items = list(records.values()) if isinstance(records, Mapping) else list(records)
    if not items:
        return {"n": 0}

    totals = sorted(r.n_prompt_chars + r.n_target_chars for r in items)

    def pct(p: float) -> int:
        return totals[min(len(totals) - 1, int(p * len(totals)))]

    longest = max(items, key=lambda r: r.n_prompt_chars + r.n_target_chars)
    return {
        "n": len(items),
        "chars_p50": pct(0.50),
        "chars_p95": pct(0.95),
        "chars_p99": pct(0.99),
        "chars_max": totals[-1],
        "longest_db_id": longest.db_id,
        "est_tokens_p99": int(pct(0.99) / CHARS_PER_TOKEN),
        "est_tokens_max": int(totals[-1] / CHARS_PER_TOKEN),
        "chars_per_token_assumed": CHARS_PER_TOKEN,
    }


def count_over_budget(records: list[SftRecord], max_chars: int) -> list[int]:
    """Question ids whose full example exceeds ``max_chars``."""
    return [
        r.question_id for r in records if r.n_prompt_chars + r.n_target_chars > max_chars
    ]
