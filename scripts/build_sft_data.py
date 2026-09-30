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
    SCHEMA_MODES,
    SPLITS,
    PromptConfig,
    discover_split,
    fetch_sample_rows,
    load_schema,
    render_selected_schema,
)
from text2sql_rlvr.data.sft import (
    SELECTION_POLICIES,
    build_sft_record,
    count_over_budget,
    length_report,
    select_sft_examples,
    selection_report,
)
from text2sql_rlvr.ledger import file_sha256, git_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--split", choices=SPLITS, default="train",
                        help="which split's databases the questions belong to")
    parser.add_argument("--questions", type=Path,
                        default=Path("data/processed/train_filtered.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/sft/dataset.json"))

    parser.add_argument("--instruction-version", choices=("v1", "v2"), default="v1",
                        help="v1 is the frozen prompt; v2 measured worse on the validation set")
    parser.add_argument("--schema-style", choices=("ddl", "compact"), default="ddl")
    parser.add_argument("--schema-mode", choices=tuple(m for m in SCHEMA_MODES if m != "oracle"),
                        default="full",
                        help="must match the schema the model is evaluated with; oracle is "
                             "excluded because training on gold tables leaks the answer")
    parser.add_argument("--descriptions", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--no-evidence", action="store_true")

    parser.add_argument("--outcomes", type=Path, default=None,
                        help="per-question outcomes from evaluate.py; enables --policy")
    parser.add_argument("--policy", choices=SELECTION_POLICIES, default="all",
                        help="which of the evaluated questions to train on: all; random (the "
                             "control); failure (the ones the model got wrong)")
    parser.add_argument("--size", type=int, default=0, help="training examples; 0 keeps all")
    parser.add_argument("--replay-fraction", type=float, default=0.0,
                        help="failure policy: share of the output drawn from solved questions")
    parser.add_argument("--cap-per-db", type=int, default=0)
    parser.add_argument("--drop-gold-empty", action="store_true",
                        help="skip questions whose gold SQL returns no rows")
    parser.add_argument("--seed", type=int, default=0)

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

    split = replace(discover_split(args.root, args.split), questions_path=args.questions)
    examples = split.load()
    print(f"input   {len(examples)} questions from {args.questions}")
    print(f"prompt  {config.as_dict()} schema_mode={args.schema_mode}")

    outcomes = []
    selection = None
    if args.outcomes:
        outcomes = [
            json.loads(line)
            for line in args.outcomes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keep = set(select_sft_examples(
            outcomes,
            policy=args.policy,
            size=args.size,
            replay_fraction=args.replay_fraction,
            cap_per_db=args.cap_per_db,
            drop_gold_empty=args.drop_gold_empty,
            seed=args.seed,
        ))
        examples = [e for e in examples if e.question_id in keep]
        selection = selection_report(sorted(keep), outcomes)
        solved = sum(1 for r in outcomes if r["official"])
        print(f"scored  {len(outcomes)} questions, {solved} solved, "
              f"{len(outcomes) - solved} failed")
        print(f"policy  {args.policy} size={args.size or 'all'} "
              f"replay={args.replay_fraction} cap_per_db={args.cap_per_db or 'none'} "
              f"seed={args.seed}")
        print(f"kept    {selection['n']} questions from {selection['n_databases']} databases "
              f"({selection['n_from_failures']} failed, {selection['n_from_solved']} solved), "
              f"{selection['per_db_min']} to {selection['per_db_max']} per database")
    elif args.policy != "all" or args.size:
        raise SystemExit("--policy and --size need --outcomes")

    schema_cache: dict[str, object] = {}
    sample_cache: dict[str, dict | None] = {}
    records = []
    for example in examples:
        if example.db_id not in schema_cache:
            db_path = split.db_path(example.db_id)
            schema_cache[example.db_id] = load_schema(db_path, db_id=example.db_id)
            sample_cache[example.db_id] = (
                fetch_sample_rows(db_path, schema_cache[example.db_id], config.sample_rows)
                if config.sample_rows
                else None
            )
        schema_text, _ = render_selected_schema(
            schema_cache[example.db_id],
            example,
            mode=args.schema_mode,
            style=config.schema_style,
            include_descriptions=config.include_descriptions,
            sample_rows=sample_cache[example.db_id],
        )
        records.append(build_sft_record(example, schema_text, config))

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
        "schema_mode": args.schema_mode,
        "selection": {
            "outcomes": str(args.outcomes) if args.outcomes else None,
            "outcomes_sha256": file_sha256(args.outcomes),
            "policy": args.policy,
            "size": args.size,
            "replay_fraction": args.replay_fraction,
            "cap_per_db": args.cap_per_db,
            "drop_gold_empty": args.drop_gold_empty,
            "seed": args.seed,
            "report": selection,
        },
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
