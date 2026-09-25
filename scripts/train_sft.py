"""Lightweight LoRA SFT for Qwen3 on BIRD, using transformers + peft.

Replicates the LLaMA-Factory config (configs/sft/qwen3_1.7b_lora.yaml) without
pulling in LLaMA-Factory itself, so the carefully-tuned verl environment is not
disturbed. The prompt/chat-template contract is the same one generate.py and
vLLM use: qwen3 template with enable_thinking=False.

Loss is taken on every assistant turn, not only the final one, so multi-turn
trajectories train the way they read; the span logic and its tests live in
text2sql_rlvr.masking.

    python scripts/train_sft.py \
        --model /root/autodl-tmp/Qwen3-4B \
        --data /root/autodl-tmp/sft_data/sft_train.jsonl \
        --out /root/autodl-tmp/out/qwen3-4b-sft-lora
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from text2sql_rlvr.masking import mask_assistant_turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to the base model")
    parser.add_argument("--data", required=True, help="sharegpt-format jsonl (messages)")
    parser.add_argument("--out", required=True, help="directory to save the LoRA adapter")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()

    def tokenize(example: dict) -> dict:
        """Tokenize one chat, keeping loss on the assistant turns only."""
        masked = mask_assistant_turns(
            tokenizer, example["messages"], max_length=args.max_length
        )
        return {
            "input_ids": masked.input_ids,
            "labels": masked.labels,
            "n_turns": masked.n_assistant_turns,
            "n_supervised": masked.n_supervised_tokens,
            "truncated": masked.truncated,
        }

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Report the mask before training on it. An example whose answer was cut off
    # by --max-length contributes no gradient at all, and a turn count of 1 on
    # data that was meant to be trajectories means the file is not what the
    # command assumes it is -- both are invisible once training starts.
    turns, supervised = dataset["n_turns"], dataset["n_supervised"]
    truncated = sum(dataset["truncated"])
    kept = dataset.filter(lambda example: example["n_supervised"] > 0)
    dropped = len(dataset) - len(kept)
    print(f"examples       {len(kept)} trained, {dropped} dropped with no answer tokens left")
    print(f"assistant turns per example   min {min(turns)} max {max(turns)}")
    print(f"supervised tokens             total {sum(supervised)}, "
          f"mean {sum(supervised) / max(len(supervised), 1):.1f}")
    print(f"truncated at {args.max_length} tokens    {truncated} examples")
    dataset = kept.remove_columns(["n_turns", "n_supervised", "truncated"])

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nLoRA adapter saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
