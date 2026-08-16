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
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_weights(ckpt: Path) -> list[Path]:
    """Locate safetensors files; prefer ones sitting next to a config.json."""
    tensors = sorted(ckpt.rglob("*.safetensors"))
    hf = [t for t in tensors if (t.parent / "config.json").is_file()]
    return hf or tensors


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt", required=True, help="verl checkpoint dir (e.g. .../global_step_150/actor)"
    )
    parser.add_argument("--base", required=True, help="base model dir for config/tokenizer")
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

    print(f"loading base model config/tokenizer from {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    state = {}
    for path in weights:
        print(f"  reading {path}")
        from safetensors.torch import load_file

        state.update(load_file(path))
    state = strip_fsdp_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(f"state dict does not cover the model; missing keys: {missing[:10]}")
    print(f"loaded {len(state)} tensors ({len(unexpected)} unexpected, ignored)")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nsaved HF model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
