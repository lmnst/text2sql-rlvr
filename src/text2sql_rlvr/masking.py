"""Turn a chat into training ids, keeping loss on *every* assistant turn.

Single-turn SFT can get away with "mask everything before the last message":
one question, one answer, and the boundary between them is the only boundary
there is. A trajectory is different. ``user, assistant, user(execution result),
assistant, ...`` has several answers, and the naive rule supervises only the
last one -- the middle turns silently become prompt. Nothing errors, the loss
curve looks normal, and the model is simply never trained on the reasoning
steps the trajectory was collected for.

So the span of each assistant turn is located explicitly. The chat is rendered
by the tokenizer's own template, which is also what generation sends, and each
turn's boundaries come from rendering the conversation twice: up to that turn
with a generation prompt (where the answer starts) and through that turn
(where it ends). The assistant's end-of-turn token stays inside the span, since
stopping is part of what the model has to learn.

That trick assumes rendering a prefix produces a prefix of the full rendering,
which every ``<|im_start|>``-style template satisfies. It is checked rather
than assumed: a template that violates it would produce a mask that is subtly
misaligned instead of obviously broken, which is exactly the failure this
module exists to prevent.

This lives at the package root, not under ``data``, on purpose: train_sft.py is
copied to a rented GPU box by hand, and importing it must not drag in the whole
data package. Only the package docstring and this file have to make the trip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

#: The label value PyTorch's cross entropy skips.
IGNORE_INDEX = -100


class ChatTokenizer(Protocol):
    """The slice of a HuggingFace tokenizer this module needs."""

    def apply_chat_template(self, conversation: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        ...

    def encode(self, text: str, **kwargs: Any) -> Sequence[int]:
        ...


@dataclass(frozen=True)
class MaskedChat:
    """One tokenised example, plus what was actually supervised in it."""

    input_ids: list[int]
    labels: list[int]
    n_assistant_turns: int
    n_supervised_tokens: int
    truncated: bool

    def as_dict(self) -> dict[str, object]:
        return {"input_ids": self.input_ids, "labels": self.labels}


def mask_assistant_turns(
    tokenizer: ChatTokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    max_length: int = 0,
    enable_thinking: bool = False,
) -> MaskedChat:
    """Tokenise ``messages`` and mask everything that is not an assistant turn.

    ``max_length`` of 0 disables truncation. Truncation cuts the end of the
    sequence, which is where the answer is, so ``n_supervised_tokens`` can come
    back 0 for an example that was too long; the caller decides what to do with
    those rather than training on a sequence that teaches nothing.

    ``enable_thinking`` is passed to the chat template and must match what
    generation sends (this project keeps it off).
    """
    if not messages:
        raise ValueError("messages is empty")

    def render(upto: int, add_generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            list(messages[:upto]),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    input_ids = encode(render(len(messages), False))

    spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prefix_ids = encode(render(index, True))
        through_ids = encode(render(index + 1, False))
        if input_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"message {index}: the chat template does not render prefixes as "
                "prefixes, so assistant spans cannot be located by rendering twice"
            )
        if input_ids[: len(through_ids)] != through_ids:
            raise ValueError(
                f"message {index}: rendering through this turn does not match the "
                "full rendering; the template rewrites earlier turns"
            )
        spans.append((len(prefix_ids), len(through_ids)))

    truncated = bool(max_length) and len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]

    labels = [IGNORE_INDEX] * len(input_ids)
    kept_turns = 0
    for start, end in spans:
        start, end = min(start, len(input_ids)), min(end, len(input_ids))
        if end <= start:
            continue
        labels[start:end] = input_ids[start:end]
        kept_turns += 1

    return MaskedChat(
        input_ids=input_ids,
        labels=labels,
        n_assistant_turns=kept_turns,
        n_supervised_tokens=sum(1 for label in labels if label != IGNORE_INDEX),
        truncated=truncated,
    )
