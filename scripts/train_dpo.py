"""Direct Preference Optimisation on execution-mined pairs, transformers + peft only.

    python scripts/train_dpo.py \\
        --model /root/autodl-tmp/Qwen3-4B \\
        --adapter /root/autodl-tmp/out/qwen3-4b-sft-random \\
        --data /root/autodl-tmp/dpo/dpo_4b.jsonl \\
        --out /root/autodl-tmp/out/qwen3-4b-dpo

DPO needs a reference policy to measure movement against. Here the reference
log-probabilities are computed once, before the first update, with the policy
exactly as it is loaded, and stored next to each pair. With --adapter that is
the SFT policy. Switching the adapter off instead, the usual LoRA shortcut, is
wrong for this recipe: when training continues on the SFT adapter, "adapter
off" is the untrained base model, not the SFT policy. Precomputing also saves
one forward pass per step and keeps a single model in memory.

Sanity check: because policy and reference start identical, the first logged
loss should sit near ln 2 = 0.693. A start far from that means the reference
is not the starting policy.

An adapter is saved every --save-steps optimizer steps (adapter weights only,
no optimizer state) under <out>/checkpoint-<step>. Pick among them on val 788:
serve them all as LoRA modules in one vllm process and score each with the
usual generate.py + evaluate.py path. DPO overshoots easily, so the last
checkpoint is not assumed to be the best one.

Deliberately no TRL. The GRPO environment in requirements-train.txt is a
pinned, fragile set that took days to assemble; the DPO objective is six lines
and is not worth another dependency that pulls its own transformers range.
The loss is the standard one:

    -log sigmoid( beta * [ (logp_chosen - logp_ref_chosen)
                         - (logp_rejected - logp_ref_rejected) ] )

Start from the SFT adapter with --adapter so the reference is the SFT policy
rather than the base model; that is the usual SFT-then-DPO recipe.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SIDES = ("chosen", "rejected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to the base model")
    parser.add_argument("--adapter", default="",
                        help="SFT LoRA to continue from; omitted starts from the base model")
    parser.add_argument("--data", required=True, help="jsonl from build_dpo_data.py")
    parser.add_argument("--out", required=True, help="directory to save the LoRA adapter")

    parser.add_argument("--beta", type=float, default=0.1,
                        help="how hard to push apart; higher stays closer to the reference")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5.0e-6,
                        help="an order of magnitude below SFT: DPO overshoots easily")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1, help="pairs per device step")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=50,
                        help="save an adapter checkpoint every this many optimizer steps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    if args.adapter:
        # Continue training the SFT adapter; the reference is precomputed below
        # from these weights, because disabling the adapter would give the base.
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
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

    def render(messages: list[dict], add_generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )

    def tokenize(example: dict) -> dict:
        """One pair: prompt masked out, only the two responses carry loss."""
        prompt_ids = tokenizer.encode(
            render(example["prompt"], True), add_special_tokens=False
        )
        out: dict[str, list[int]] = {}
        for side in SIDES:
            full_ids = tokenizer.encode(
                render(example["prompt"] + example[side], False), add_special_tokens=False
            )[: args.max_length]
            labels = [-100] * len(full_ids)
            answer_start = min(len(prompt_ids), len(full_ids))
            labels[answer_start:] = full_ids[answer_start:]
            out[f"{side}_input_ids"] = full_ids
            out[f"{side}_labels"] = labels
        return out

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    pad_id = tokenizer.pad_token_id

    def collate(features: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(f[f"{side}_input_ids"]) for f in features for side in SIDES)
        batch: dict[str, torch.Tensor] = {}
        for side in SIDES:
            ids, labels, mask = [], [], []
            for feature in features:
                seq = feature[f"{side}_input_ids"]
                lab = feature[f"{side}_labels"]
                pad = width - len(seq)
                ids.append(seq + [pad_id] * pad)
                labels.append(lab + [-100] * pad)
                mask.append([1] * len(seq) + [0] * pad)
            batch[f"{side}_input_ids"] = torch.tensor(ids, dtype=torch.long)
            batch[f"{side}_labels"] = torch.tensor(labels, dtype=torch.long)
            batch[f"{side}_attention_mask"] = torch.tensor(mask, dtype=torch.long)
        return batch

    def sequence_logps(model, batch: dict, side: str) -> torch.Tensor:
        """Total log-probability the model assigns to one side's response."""
        logits = model(
            input_ids=batch[f"{side}_input_ids"],
            attention_mask=batch[f"{side}_attention_mask"],
        ).logits[:, :-1]
        labels = batch[f"{side}_labels"][:, 1:]
        mask = labels.ne(-100)
        picked = torch.gather(logits, 2, labels.masked_fill(~mask, 0).unsqueeze(2)).squeeze(2)
        # log softmax without materialising a second full-vocabulary tensor
        normaliser = torch.logsumexp(logits.float(), dim=-1)
        return ((picked.float() - normaliser) * mask).sum(-1)

    # Reference = the policy before any update. eval() turns LoRA dropout off.
    model.to("cuda")
    model.eval()
    reference: dict[str, list[float]] = {side: [] for side in SIDES}
    with torch.no_grad():
        for index, example in enumerate(dataset):
            batch = {key: value.to(model.device) for key, value in collate([example]).items()}
            for side in SIDES:
                reference[side].append(sequence_logps(model, batch, side).item())
            if (index + 1) % 200 == 0:
                print(f"reference log-probs: {index + 1}/{len(dataset)}")
    model.train()
    for side in SIDES:
        dataset = dataset.add_column(f"reference_{side}", reference[side])

    def collate_with_reference(features: list[dict]) -> dict[str, torch.Tensor]:
        batch = collate(features)
        for side in SIDES:
            batch[f"reference_{side}"] = torch.tensor(
                [f[f"reference_{side}"] for f in features], dtype=torch.float32
            )
        return batch

    class DpoTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
            policy = {side: sequence_logps(model, inputs, side) for side in SIDES}
            reference = {side: inputs[f"reference_{side}"] for side in SIDES}

            margin = (policy["chosen"] - reference["chosen"]) - (
                policy["rejected"] - reference["rejected"]
            )
            loss = -F.logsigmoid(args.beta * margin).mean()
            self.log_extra = {
                "margin": margin.mean().item(),
                # how often the pair is already ordered correctly
                "accuracy": (margin > 0).float().mean().item(),
                "logp_chosen": policy["chosen"].mean().item(),
                "logp_rejected": policy["rejected"].mean().item(),
            }
            return (loss, policy) if return_outputs else loss

        def log(self, logs: dict, *log_args, **log_kwargs):
            extra = getattr(self, "log_extra", None)
            if extra and "loss" in logs:
                logs.update({k: round(v, 4) for k, v in extra.items()})
            return super().log(logs, *log_args, **log_kwargs)

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_only_model=True,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = DpoTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_with_reference,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(f"{args.out}/dpo_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=1)
    print(f"\nLoRA adapter saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
