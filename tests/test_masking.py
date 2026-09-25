"""Assistant-turn masking.

The failure being pinned here is silent by construction: a wrong mask trains
without complaint and only shows up as a model that ignored most of its
training data. So the tests assert on *which tokens* carry loss, not just on
how many.
"""

from __future__ import annotations

import pytest

from text2sql_rlvr.masking import IGNORE_INDEX, mask_assistant_turns


class FakeTokenizer:
    """A Qwen-shaped chat template over whitespace tokenisation.

    Real tokenisers cannot run here (no torch in the dev environment), and the
    property under test is about spans, not about subword merges. Whitespace
    tokens keep the mapping between ids and text readable in assertions.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def apply_chat_template(
        self, conversation, *, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ) -> str:
        assert tokenize is False, "this module must render text, not ids"
        assert enable_thinking is False
        parts = [
            f"<|im_start|> {m['role']}\n{m['content']} <|im_end|>\n" for m in conversation
        ]
        if add_generation_prompt:
            parts.append("<|im_start|> assistant\n")
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [self.vocab.setdefault(tok, len(self.vocab) + 1) for tok in text.split()]

    def decode(self, ids) -> list[str]:
        back = {i: tok for tok, i in self.vocab.items()}
        return [back[i] for i in ids]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def supervised_text(tokenizer, masked) -> list[str]:
    return tokenizer.decode([label for label in masked.labels if label != IGNORE_INDEX])


SINGLE_TURN = [
    {"role": "system", "content": "you write sql"},
    {"role": "user", "content": "how many staff"},
    {"role": "assistant", "content": "SELECT count(*)"},
]

TRAJECTORY = [
    {"role": "system", "content": "you write sql"},
    {"role": "user", "content": "how many staff"},
    {"role": "assistant", "content": "first try"},
    {"role": "user", "content": "execution result empty"},
    {"role": "assistant", "content": "second try"},
    {"role": "user", "content": "execution result three rows"},
    {"role": "assistant", "content": "final answer"},
]


class TestSingleTurn:
    def test_only_the_answer_carries_loss(self, tokenizer):
        masked = mask_assistant_turns(tokenizer, SINGLE_TURN)
        assert supervised_text(tokenizer, masked) == ["SELECT", "count(*)", "<|im_end|>"]

    def test_labels_and_ids_stay_aligned(self, tokenizer):
        masked = mask_assistant_turns(tokenizer, SINGLE_TURN)
        assert len(masked.labels) == len(masked.input_ids)
        for label, token in zip(masked.labels, masked.input_ids, strict=True):
            assert label in (IGNORE_INDEX, token)

    def test_the_prompt_is_fully_masked(self, tokenizer):
        masked = mask_assistant_turns(tokenizer, SINGLE_TURN)
        prompt_len = len(tokenizer.encode(
            tokenizer.apply_chat_template(SINGLE_TURN[:-1], add_generation_prompt=True)
        ))
        assert set(masked.labels[:prompt_len]) == {IGNORE_INDEX}


class TestMultiTurn:
    def test_every_assistant_turn_carries_loss(self, tokenizer):
        """The regression: masking only up to the last message loses the middle turns."""
        masked = mask_assistant_turns(tokenizer, TRAJECTORY)
        assert masked.n_assistant_turns == 3
        assert supervised_text(tokenizer, masked) == [
            "first", "try", "<|im_end|>",
            "second", "try", "<|im_end|>",
            "final", "answer", "<|im_end|>",
        ]

    def test_intermediate_user_turns_are_masked(self, tokenizer):
        """Execution results come from the sandbox; training to predict them is noise."""
        masked = mask_assistant_turns(tokenizer, TRAJECTORY)
        assert "execution" not in supervised_text(tokenizer, masked)
        assert "rows" not in supervised_text(tokenizer, masked)

    def test_it_supervises_more_than_the_old_last_message_rule(self, tokenizer):
        """Quantifies the bug: the old rule trained on one turn out of three."""
        masked = mask_assistant_turns(tokenizer, TRAJECTORY)
        old_rule_start = len(tokenizer.encode(
            tokenizer.apply_chat_template(TRAJECTORY[:-1], add_generation_prompt=True)
        ))
        old_rule_supervised = len(masked.input_ids) - old_rule_start
        assert old_rule_supervised == 3  # "final answer <|im_end|>" and nothing else
        assert masked.n_supervised_tokens == 9

    def test_a_chat_without_an_assistant_turn_supervises_nothing(self, tokenizer):
        masked = mask_assistant_turns(tokenizer, TRAJECTORY[:2])
        assert masked.n_supervised_tokens == 0
        assert masked.n_assistant_turns == 0


class TestTruncation:
    def test_truncation_cuts_both_sides_together(self, tokenizer):
        masked = mask_assistant_turns(tokenizer, TRAJECTORY, max_length=12)
        assert masked.truncated is True
        assert len(masked.input_ids) == len(masked.labels) == 12

    def test_a_turn_cut_off_entirely_is_not_counted(self, tokenizer):
        """An example whose answers are all past the cutoff teaches nothing."""
        masked = mask_assistant_turns(tokenizer, SINGLE_TURN, max_length=6)
        assert masked.n_supervised_tokens == 0
        assert masked.n_assistant_turns == 0

    def test_a_partly_cut_turn_keeps_what_survived(self, tokenizer):
        full = mask_assistant_turns(tokenizer, SINGLE_TURN)
        cut = mask_assistant_turns(tokenizer, SINGLE_TURN, max_length=len(full.input_ids) - 1)
        assert cut.n_assistant_turns == 1
        assert 0 < cut.n_supervised_tokens < full.n_supervised_tokens

    def test_no_truncation_flag_when_it_fits(self, tokenizer):
        masked = mask_assistant_turns(tokenizer, SINGLE_TURN, max_length=8192)
        assert masked.truncated is False


class TestTemplateContract:
    def test_a_template_that_rewrites_history_is_rejected(self, tokenizer):
        """Better a loud failure than a mask that is off by a few tokens."""

        class RewritingTokenizer(FakeTokenizer):
            def apply_chat_template(self, conversation, **kwargs):
                # Keeps only the last two messages, the way some templates drop
                # history: prefixes then stop being prefixes.
                return super().apply_chat_template(list(conversation)[-2:], **kwargs)

        with pytest.raises(ValueError, match="prefix|rewrites"):
            mask_assistant_turns(RewritingTokenizer(), TRAJECTORY)

    def test_empty_messages_is_an_error(self, tokenizer):
        with pytest.raises(ValueError, match="empty"):
            mask_assistant_turns(tokenizer, [])
