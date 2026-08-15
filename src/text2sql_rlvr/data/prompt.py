"""Prompt construction.

The output-format instruction and :func:`text2sql_rlvr.sql.extract_sql` are one
contract: if you loosen one, loosen the other, or a chunk of the eval turns into
parse failures that look like model errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from text2sql_rlvr.data.bird import BirdExample

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite queries. "
    "You answer with a single query and nothing else."
)

#: v1: the original instruction, kept so the first baseline stays reproducible.
INSTRUCTION_V1 = (
    "Write one SQLite SELECT query that answers the question using the schema above.\n"
    "Return only the query, inside a ```sql code block. Do not explain it."
)

#: v2 adds three rules, each one earned by a failure mode measured on real output
#: rather than guessed at. See the milestone 6 entry in docs/PROGRESS.md.
#:
#: * BIRD's external knowledge is written with pseudo-functions -- ``DIVIDE(a, b)``,
#:   ``SUBTRACT(a, b)`` -- which the model copied verbatim into SQL. That is
#:   notation in the hint, not SQL, and it accounted for the single largest group
#:   of "no such function" failures.
#: * The model reached for MySQL date functions (``YEAR``, ``MONTH``) and Oracle
#:   set operators (``MINUS``), which SQLite does not have.
#: * A handful of answers were two statements separated by a semicolon.
INSTRUCTION_V2 = (
    "Write one SQLite SELECT query that answers the question using the schema above.\n"
    "\n"
    "Rules:\n"
    "- Target SQLite. Do not use functions from other databases: there is no YEAR(), "
    "MONTH(), DATEDIFF() or MINUS. Use strftime() or SUBSTR() for dates and EXCEPT "
    "for set difference.\n"
    "- The external knowledge above may be written with pseudo-functions such as "
    "DIVIDE(a, b) or SUBTRACT(a, b). Those are notation, not SQL. Write them as "
    "a / b and a - b.\n"
    "- A column can only be used if it belongs to a table in your FROM or JOIN "
    "clause. Join the tables you need.\n"
    "- Answer with exactly one statement.\n"
    "\n"
    "Return only the query, inside a ```sql code block. Do not explain it."
)

INSTRUCTIONS = {"v1": INSTRUCTION_V1, "v2": INSTRUCTION_V2}

#: Back-compat name for the current default. v1 is the pinned prompt since
#: milestone 7 (v2 was measured and rejected on the validation set); see
#: docs/PROGRESS.md.
INSTRUCTION = INSTRUCTION_V1


@dataclass(frozen=True)
class PromptConfig:
    """Knobs that change the prompt, and therefore change the metric.

    Record this alongside every result: the same checkpoint scores differently
    under a different schema style or instruction, so a number without its prompt
    config is not comparable to anything.
    """

    schema_style: str = "ddl"
    include_descriptions: bool = False
    include_evidence: bool = True
    sample_rows: int = 0
    instruction_version: str = "v1"

    def __post_init__(self) -> None:
        if self.instruction_version not in INSTRUCTIONS:
            raise ValueError(
                f"instruction_version must be one of {sorted(INSTRUCTIONS)}, "
                f"got {self.instruction_version!r}"
            )

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[self.instruction_version]

    def as_dict(self) -> dict[str, object]:
        return dict(vars(self))


def build_user_prompt(
    example: BirdExample, schema_text: str, config: PromptConfig | None = None
) -> str:
    """Render the user turn for one example."""
    config = config or PromptConfig()
    parts = [f"Database schema:\n\n{schema_text}"]
    if config.include_evidence and example.evidence:
        parts.append(f"External knowledge: {example.evidence}")
    parts.append(f"Question: {example.question}")
    parts.append(config.instruction)
    return "\n\n".join(parts)


def build_messages(
    example: BirdExample,
    schema_text: str,
    config: PromptConfig | None = None,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Render one example as chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(example, schema_text, config)},
    ]
