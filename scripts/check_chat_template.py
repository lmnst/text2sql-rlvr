"""Check that training and inference agree on the chat format.

Runs on the training box only -- it needs `transformers` and the model files.

The question this answers is not "does the template insert a think block".
Qwen3's non-thinking format legitimately contains an empty ``<think></think>``
pair, and seeing one is fine. The question is whether the string inference hands
to the model is **exactly a prefix of** the string training optimises.

If it is, the model is trained to continue from precisely where generation will
start it, and there is nothing to worry about. If it is not, the model learns to
produce its answer in a context it will never be given, and the resulting score
measures something other than what was trained.

    python scripts/check_chat_template.py \\
        --model /root/autodl-tmp/Qwen3-1.7B \\
        --data /root/autodl-tmp/sft_data/sft_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to the base model")
    parser.add_argument("--data", required=True, help="sft jsonl produced by build_sft_data.py")
    parser.add_argument("--n", type=int, default=3, help="examples to check")
    parser.add_argument("--thinking", action="store_true", help="render with thinking enabled")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is not installed. This script runs on the training box only.")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {"enable_thinking": args.thinking}

    lines = []
    with Path(args.data).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(json.loads(line))
            if len(lines) >= args.n:
                break

    if not lines:
        print(f"no examples in {args.data}")
        return 1

    all_ok = True
    for index, record in enumerate(lines):
        messages = record["messages"]
        try:
            training = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs
            )
            inference = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True, **kwargs
            )
        except TypeError:
            # Older templates do not take enable_thinking.
            print("template does not accept enable_thinking; rendering without it")
            training = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inference = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )

        prefix_ok = training.startswith(inference)
        all_ok &= prefix_ok

        if index == 0:
            print("=" * 70)
            print("what generation hands the model (last 200 chars):")
            print(repr(inference[-200:]))
            print()
            print("what training optimises, from that point on (next 200 chars):")
            print(repr(training[len(inference):][:200]) if prefix_ok
                  else repr(training[-260:]))
            print("=" * 70)

        status = "OK" if prefix_ok else "MISMATCH"
        print(f"example {index} (question_id={record.get('question_id')}): {status}")

    print()
    if all_ok:
        print("PASS: the inference prompt is an exact prefix of the training sequence.")
        print("Training and generation agree; the empty <think></think> block, if")
        print("present, is on both sides and is Qwen3's normal non-thinking format.")
        return 0

    print("FAIL: generation would hand the model a string training never saw.")
    print("Do not start the full run. Options, in order of preference:")
    print("  1. change the trainer's template setting so the two agree;")
    print("  2. match the serving side to whatever the trainer produced.")
    print("Either way, the two must be made identical before training is worth paying for.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
