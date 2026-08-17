"""Convert a verl GRPO checkpoint into a plain HuggingFace model directory.

Single-GPU verl runs save a full (unsharded) FSDP state dict, whose keys carry
FSDP wrappers. This script strips the wrappers and writes a normal HF directory
that vLLM and generate.py can load directly.

    python scripts/convert_checkpoint.py \
        --ckpt /root/autodl-tmp/out/grpo/grpo_strict/checkpoints/global_step_150/actor \
        --base /root/autodl-tmp/Qwen3-1.7B \
        --out /root/autodl-tmp/Qwen3-1.7B-grpo

If the checkpoint is already a HF directory (contains config.json next to the
weights), it is copied straight through.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_weights(ckpt: Path) -> list[Path]:
    """Locate model weights saved by verl or HuggingFace."""
    tensors = sorted(ckpt.rglob("*.safetensors"))
    hf = [t for t in tensors if (t.parent / "config.json").is_file()]
    if hf or tensors:
        return hf or tensors
    return sorted(ckpt.glob("model_world_size_*_rank_*.pt"))


def strip_fsdp_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove FSDP/accelerate wrapper prefixes from state dict keys."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        cleaned = key
        # Common wrappers observed in verl FSDP checkpoints.
        for prefix in ("_fsdp_wrapped_module.", "_forward_module.", "model.model."):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        out[cleaned] = value
    return out


def load_state(weights: list[Path]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for path in weights:
        print(f"  reading {path}")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state.update(load_file(path))
        else:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            state.update(loaded)
    return state


def merge_lora_checkpoint(
    state: dict[str, torch.Tensor], ckpt: Path, base: str
):
    from peft import LoraConfig, get_peft_model

    meta_path = ckpt / "lora_train_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"LoRA checkpoint is missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_modules = sorted(
        {
            key.split(".lora_A.", 1)[0].rsplit(".", 1)[-1]
            for key in state
            if ".lora_A." in key
        }
    )
    if not target_modules:
        raise SystemExit("LoRA tensors were found but target modules could not be inferred")

    print(f"loading GRPO start model from {base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=int(meta["r"]),
            lora_alpha=int(meta["lora_alpha"]),
            target_modules=target_modules,
            task_type=meta.get("task_type", "CAUSAL_LM"),
            bias="none",
        ),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            "checkpoint does not exactly cover the PEFT model; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    print(f"loaded {len(state)} tensors; merging LoRA into the start model")
    return model.merge_and_unload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt", required=True, help="verl checkpoint dir (e.g. .../global_step_150/actor)"
    )
    parser.add_argument(
        "--base", required=True,
        help="model used to start GRPO (required to reconstruct LoRA checkpoints)",
    )
    parser.add_argument("--out", required=True, help="output HF model directory")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        raise SystemExit(f"checkpoint dir not found: {ckpt}")

    # Case 1: already a HF directory -- copy as-is.
    if (ckpt / "config.json").is_file():
        print(f"{ckpt} is already a HF directory; copying")
        shutil.copytree(ckpt, args.out, dirs_exist_ok=True)
        return 0

    # Case 2: raw FSDP state dict -> load into a fresh model and save as HF.
    weights = find_weights(ckpt)
    if not weights:
        raise SystemExit(f"no safetensors found under {ckpt}; "
                         f"inspect the checkpoint layout and report it")

    state = load_state(weights)
    if any(".lora_A." in key for key in state):
        model = merge_lora_checkpoint(state, ckpt, args.base)
    else:
        print(f"loading base model config from {args.base}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        state = strip_fsdp_prefix(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise SystemExit(
                f"state dict does not cover the model; missing keys: {missing[:10]}"
        )
        print(f"loaded {len(state)} tensors ({len(unexpected)} unexpected, ignored)")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nsaved HF model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
