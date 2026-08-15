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

from collections.abc import Mapping
from dataclasses import dataclass

from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.prompt import PromptConfig, build_messages
from text2sql_rlvr.sql import extract_sql


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
