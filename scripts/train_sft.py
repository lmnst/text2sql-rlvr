"""Lightweight LoRA SFT for Qwen3-1.7B on BIRD, using transformers + peft.

Replicates the LLaMA-Factory config (configs/sft/qwen3_1.7b_lora.yaml) without
pulling in LLaMA-Factory itself, so the carefully-tuned verl environment is not
disturbed. The prompt/chat-template contract is the same one generate.py and
vLLM use: qwen3 template with enable_thinking=False.

    python scripts/train_sft.py \
        --model /root/autodl-tmp/Qwen3-1.7B \
        --data /root/autodl-tmp/sft_data/sft_train.jsonl \
        --out /root/autodl-tmp/out/qwen3-1.7b-sft-lora
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


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
        """Tokenize one chat example, masking the prompt so loss is answer-only."""
        messages = example["messages"]
        # Prompt ids: everything up to and including the assistant role marker.
        # tokenize=False + encode keeps a plain list[int] regardless of the
        # transformers version's return shape.
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) > args.max_length:
            full_ids = full_ids[: args.max_length]
        labels = [-100] * len(full_ids)
        # The answer occupies full_ids[len(prompt_ids):], provided the prompt is
        # a prefix of the full sequence. When truncation cut into the answer we
        # still mask whatever remained of the prompt.
        answer_start = min(len(prompt_ids), len(full_ids))
        labels[answer_start:] = full_ids[answer_start:]
        return {"input_ids": full_ids, "labels": labels}

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

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
