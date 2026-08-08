"""Write the SFT training file from the filtered train split.

    python scripts/build_sft_data.py

Output is chat-format jsonl (`{"messages": [...]}`), which LLaMA-Factory, TRL and
verl's SFT trainer all accept. The prompt config used is written to a manifest
next to it, because a training set built with a different prompt than the one
evaluation uses is worse than useless -- it is misleading.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import (
    PromptConfig,
    discover_split,
    fetch_sample_rows,
    format_schema,
    load_schema,
)
from text2sql_rlvr.data.sft import build_sft_record, count_over_budget, length_report
from text2sql_rlvr.ledger import file_sha256, git_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--questions", type=Path,
                        default=Path("data/processed/train_filtered.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/sft/dataset.json"))

    parser.add_argument("--instruction-version", choices=("v1", "v2"), default="v1",
                        help="v1 is the frozen prompt; see milestone 7 in docs/PROGRESS.md")
    parser.add_argument("--schema-style", choices=("ddl", "compact"), default="ddl")
    parser.add_argument("--descriptions", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--no-evidence", action="store_true")

    parser.add_argument("--cutoff-tokens", type=int, default=4096,
                        help="trainer sequence budget, used only to report what would truncate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = PromptConfig(
        schema_style=args.schema_style,
        include_descriptions=args.descriptions,
        include_evidence=not args.no_evidence,
        sample_rows=args.sample_rows,
        instruction_version=args.instruction_version,
    )

    split = replace(discover_split(args.root, "train"), questions_path=args.questions)
    examples = split.load()
    print(f"input   {len(examples)} questions from {args.questions}")
    print(f"prompt  {config.as_dict()}")

    schema_cache: dict[str, str] = {}
    records = []
    for example in examples:
        if example.db_id not in schema_cache:
            db_path = split.db_path(example.db_id)
            schema = load_schema(db_path, db_id=example.db_id)
            samples = (
                fetch_sample_rows(db_path, schema, config.sample_rows)
                if config.sample_rows
                else None
            )
            schema_cache[example.db_id] = format_schema(
                schema,
                style=config.schema_style,
                include_descriptions=config.include_descriptions,
                sample_rows=samples,
            )
        records.append(build_sft_record(example, schema_cache[example.db_id], config))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    stats = length_report(records)
    print(f"\ndatabases rendered  {len(schema_cache)}")
    print("\nexample length (characters, prompt + answer)")
    for key in ("chars_p50", "chars_p95", "chars_p99", "chars_max"):
        print(f"  {key:<12}{stats[key]:>8}")
    print(f"  longest is in database '{stats['longest_db_id']}'")
    print(f"\nestimated tokens at {stats['chars_per_token_assumed']} chars/token")
    print(f"  p99  ~{stats['est_tokens_p99']}")
    print(f"  max  ~{stats['est_tokens_max']}")

    budget_chars = int(args.cutoff_tokens * stats["chars_per_token_assumed"])
    over = count_over_budget(records, budget_chars)
    print(f"\nwith cutoff_len={args.cutoff_tokens}: {len(over)} of {len(records)} examples "
          f"({100.0 * len(over) / len(records):.1f}%) would be truncated")
    if over:
        print("  truncation cuts the END of the sequence, which is the answer -- those")
        print("  examples would teach the model to produce nothing. Raise the cutoff or")
        print("  drop them explicitly rather than letting the trainer do it silently.")

    sha, dirty = git_state(ignore_paths=(Path("results/runs.jsonl"),))
    manifest = {
        "source_questions": str(args.questions),
        "source_questions_sha256": file_sha256(args.questions),
        "output": str(args.out),
        "n_examples": len(records),
        "n_databases": len(schema_cache),
        "prompt_config": config.as_dict(),
        "length_stats": stats,
        "cutoff_tokens_checked": args.cutoff_tokens,
        "n_over_budget": len(over),
        "git_sha": sha,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nwrote {args.out}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
