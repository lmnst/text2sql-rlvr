"""Write the GRPO train/val parquet files that verl reads.

    python scripts/build_rl_data.py

Only the prompt goes in -- the answer does not. That is the whole difference
from SFT: the policy proposes, and the reward function decides. The gold SQL
rides along in `reward_model.ground_truth` so the reward can execute it, and
`extra_info.db_id` tells the reward which database to open.

The prompt is built by the same code path generation and SFT use. That is not a
nicety: if the RL prompt differs from the evaluation prompt, the policy is
optimised against an input distribution the final number never measures.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import (
    SCHEMA_MODES,
    PromptConfig,
    build_messages,
    discover_split,
    fetch_sample_rows,
    load_schema,
    render_selected_schema,
)
from text2sql_rlvr.data.sft import CHARS_PER_TOKEN
from text2sql_rlvr.ledger import file_sha256, git_state

DATA_SOURCE = "bird"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--train", type=Path,
                        default=Path("data/processed/train_filtered.json"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/rl"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/grpo/dataset.json"))
    parser.add_argument("--instruction-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--schema-style", choices=("ddl", "compact"), default="ddl")
    parser.add_argument("--schema-mode", choices=tuple(m for m in SCHEMA_MODES if m != "oracle"),
                        default="full")
    parser.add_argument("--schema-max-chars", type=int, default=0,
                        help="linked schema character budget; 0 keeps all linked tables")
    parser.add_argument("--descriptions", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=200,
                        help="validation rows kept; verl runs this every eval step")
    parser.add_argument("--val-seed", type=int, default=0,
                        help="fixed seed used to sample the validation subset")
    parser.add_argument("--val-subset-json", type=Path,
                        default=Path("data/processed/val_seed0_n200.json"),
                        help="write the exact sampled questions for external checkpoint eval")
    return parser.parse_args(argv)


def build_rows(questions_path, split, config, tag, *, schema_mode="full", schema_max_chars=0):
    schema_cache: dict[str, tuple[object, dict | None]] = {}
    rows = []
    for index, example in enumerate(replace(split, questions_path=questions_path).load()):
        if example.db_id not in schema_cache:
            db_path = split.db_path(example.db_id)
            schema = load_schema(db_path, db_id=example.db_id)
            samples = (
                fetch_sample_rows(db_path, schema, config.sample_rows)
                if config.sample_rows
                else None
            )
            schema_cache[example.db_id] = (schema, samples)
        schema, samples = schema_cache[example.db_id]
        schema_text, selection = render_selected_schema(
            schema,
            example,
            mode=schema_mode,
            style=config.schema_style,
            include_descriptions=config.include_descriptions,
            sample_rows=samples,
            max_chars=schema_max_chars,
        )
        messages = build_messages(example, schema_text, config)
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": messages,
                "ability": "text2sql",
                "reward_model": {"style": "rule", "ground_truth": example.gold_sql},
                "extra_info": {
                    "split": tag,
                    "index": index,
                    "question_id": example.question_id,
                    "db_id": example.db_id,
                    "schema_mode": selection.mode,
                    "selected_tables": list(selection.selected_tables),
                },
            }
        )
    return rows, schema_cache


def prompt_chars(row) -> int:
    return sum(len(m["content"]) for m in row["prompt"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import pandas as pd
    except ImportError:
        print("pandas and pyarrow are required to write parquet: pip install pandas pyarrow")
        return 1

    config = PromptConfig(
        schema_style=args.schema_style,
        include_descriptions=args.descriptions,
        include_evidence=True,
        sample_rows=args.sample_rows,
        instruction_version=args.instruction_version,
    )
    split = discover_split(args.root, "train")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    stats = {}
    val_selected_ids: list[int] = []
    for tag, path, limit in (("train", args.train, 0), ("val", args.val, args.val_subset)):
        rows, _schemas = build_rows(
            path,
            split,
            config,
            tag,
            schema_mode=args.schema_mode,
            schema_max_chars=args.schema_max_chars,
        )
        if limit:
            rng = random.Random(args.val_seed)
            rows = rng.sample(rows, min(limit, len(rows)))
        if tag == "val":
            val_selected_ids = [row["extra_info"]["question_id"] for row in rows]
        out = args.out_dir / f"{tag}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        written[tag] = out

        lengths = sorted(prompt_chars(r) for r in rows)
        stats[tag] = {
            "n": len(rows),
            "n_databases": len({r["extra_info"]["db_id"] for r in rows}),
            "prompt_chars_p50": lengths[len(lengths) // 2],
            "prompt_chars_p99": lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))],
            "prompt_chars_max": lengths[-1],
            "est_prompt_tokens_max": int(lengths[-1] / CHARS_PER_TOKEN),
        }
        print(f"{tag:<6} {len(rows):>6} rows -> {out}")

    val_examples = {
        example.question_id: example
        for example in replace(split, questions_path=args.val).load()
    }
    subset_records = []
    for question_id in val_selected_ids:
        example = val_examples[question_id]
        subset_records.append(
            {
                "question_id": example.question_id,
                "db_id": example.db_id,
                "question": example.question,
                "evidence": example.evidence,
                "SQL": example.gold_sql,
                "difficulty": example.difficulty,
            }
        )
    args.val_subset_json.parent.mkdir(parents=True, exist_ok=True)
    args.val_subset_json.write_text(
        json.dumps(subset_records, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"val JSON {len(subset_records):>6} rows -> {args.val_subset_json}")

    print("\nprompt length (characters; the answer is NOT in the prompt for RL)")
    for tag, s in stats.items():
        print(f"  {tag:<6} p50={s['prompt_chars_p50']:>6}  p99={s['prompt_chars_p99']:>6}  "
              f"max={s['prompt_chars_max']:>6}  (~{s['est_prompt_tokens_max']} tokens)")

    suggested = 512 * (max(s["est_prompt_tokens_max"] for s in stats.values()) // 512 + 2)
    print(f"\nconservative character estimate suggests max_prompt_length >= {suggested}")
    print("This is a warning, not an exact tokenizer count. Run diagnose_prompts.py")
    print("with the Qwen tokenizer before changing the limit or claiming prompts are dropped.")

    sha, dirty = git_state(ignore_paths=(Path("results/runs.jsonl"),))
    manifest = {
        "train_parquet": str(written["train"]),
        "val_parquet": str(written["val"]),
        "source_train": str(args.train),
        "source_train_sha256": file_sha256(args.train),
        "source_val": str(args.val),
        "val_subset_seed": args.val_seed,
        "val_subset_json": str(args.val_subset_json),
        "val_subset_question_ids": val_selected_ids,
        "prompt_config": config.as_dict(),
        "schema_selection": {
            "mode": args.schema_mode,
            "max_chars": args.schema_max_chars,
        },
        "stats": stats,
        "suggested_max_prompt_length": suggested,
        "git_sha": sha,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
