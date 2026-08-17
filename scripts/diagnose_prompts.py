"""Measure prompt length for full, linked and oracle schema variants.

This script does not call a model. Give it the Qwen tokenizer directory for
exact counts; without one it emits the project's conservative character-based
estimate and labels it as such.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import (
    SCHEMA_MODES,
    PromptConfig,
    build_messages,
    discover_split,
    load_schema,
    oracle_table_names,
    render_selected_schema,
    required_tables_fk_connected,
)
from text2sql_rlvr.data.sft import CHARS_PER_TOKEN
from text2sql_rlvr.ledger import file_sha256, git_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--split", choices=("train", "val", "mini_dev"), default="train")
    parser.add_argument("--questions", type=Path, default=Path("data/processed/val.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=SCHEMA_MODES,
                        default=list(SCHEMA_MODES))
    parser.add_argument("--tokenizer", default="",
                        help="Qwen model/tokenizer directory; omitted means estimated counts")
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--schema-max-chars", type=int, default=0,
                        help="budget for linked/oracle table blocks; 0 keeps all selected tables")
    parser.add_argument("--schema-style", choices=("ddl", "compact"), default="ddl")
    parser.add_argument("--instruction-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--subset", type=int, default=0,
                        help="fixed random subset size; applied before --limit")
    parser.add_argument("--subset-seed", type=int, default=0)
    return parser.parse_args(argv)


def _load_tokenizer(path: str):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("--tokenizer requires transformers on the diagnostic machine") from exc
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _summarise(rows: list[dict], max_tokens: int, exact: bool) -> dict[str, object]:
    token_counts = [row["prompt_tokens"] for row in rows]
    fk_rows = [row for row in rows if row["fk_connectivity_checkable"]]
    distractor_counts = [row["n_distractor_tables"] for row in rows]
    return {
        "n": len(rows),
        "token_count_kind": "exact_qwen_chat_template" if exact else "estimated_chars_div_3.6",
        "prompt_tokens_p50": _percentile(token_counts, 0.50),
        "prompt_tokens_p95": _percentile(token_counts, 0.95),
        "prompt_tokens_p99": _percentile(token_counts, 0.99),
        "prompt_tokens_max": max(token_counts),
        "n_over_budget": sum(count > max_tokens for count in token_counts),
        "over_budget_rate": round(sum(count > max_tokens for count in token_counts) / len(rows), 4),
        "mean_selected_tables": round(statistics.mean(row["n_selected_tables"] for row in rows), 2),
        "all_required_tables_retained_rate": round(
            statistics.mean(row["all_required_tables_retained"] for row in rows), 4
        ),
        "mean_required_table_recall": round(
            statistics.mean(row["required_table_recall"] for row in rows), 4
        ),
        "n_fk_connectivity_checkable": len(fk_rows),
        "fk_connectivity_retained_rate": (
            round(statistics.mean(row["fk_connectivity_retained"] for row in fk_rows), 4)
            if fk_rows
            else None
        ),
        "mean_distractor_tables": round(statistics.mean(distractor_counts), 2),
        "distractor_tables_p95": _percentile(distractor_counts, 0.95),
        "mean_distractor_rate": round(
            statistics.mean(row["distractor_table_rate"] for row in rows), 4
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer = _load_tokenizer(args.tokenizer)
    config = PromptConfig(
        schema_style=args.schema_style,
        include_descriptions=False,
        include_evidence=True,
        sample_rows=0,
        instruction_version=args.instruction_version,
    )
    split = discover_split(args.root, args.split)
    if args.questions:
        split = replace(split, questions_path=args.questions)
    examples = split.load()
    if args.subset:
        examples = random.Random(args.subset_seed).sample(
            examples,
            min(args.subset, len(examples)),
        )
    if args.limit:
        examples = examples[:args.limit]

    schemas = {}
    rows: list[dict] = []
    for example in examples:
        schema = schemas.get(example.db_id)
        if schema is None:
            schema = load_schema(split.db_path(example.db_id), db_id=example.db_id)
            schemas[example.db_id] = schema
        if not example.gold_sql:
            raise ValueError("linker diagnostics require gold SQL")
        required_tables = oracle_table_names(schema, example.gold_sql)
        if not required_tables:
            raise ValueError(f"found no required tables for question {example.question_id}")
        required_set = {name.casefold() for name in required_tables}
        full_fk_connected = required_tables_fk_connected(schema, required_tables)
        for variant in args.variants:
            schema_text, selection = render_selected_schema(
                schema,
                example,
                mode=variant,
                style=args.schema_style,
                max_chars=args.schema_max_chars,
            )
            messages = build_messages(example, schema_text, config)
            prompt_chars = sum(len(message["content"]) for message in messages)
            prompt_tokens = (
                _token_count(tokenizer, messages)
                if tokenizer is not None
                else int(prompt_chars / CHARS_PER_TOKEN)
            )
            selected_set = {name.casefold() for name in selection.selected_tables}
            retained_required = required_set & selected_set
            distractors = selected_set - required_set
            fk_checkable = len(required_set) >= 2 and full_fk_connected
            rows.append(
                {
                    "question_id": example.question_id,
                    "db_id": example.db_id,
                    "variant": variant,
                    "prompt_chars": prompt_chars,
                    "prompt_tokens": prompt_tokens,
                    "over_budget": prompt_tokens > args.max_prompt_tokens,
                    "selected_tables": list(selection.selected_tables),
                    "n_selected_tables": len(selection.selected_tables),
                    "n_tables_total": selection.total_tables,
                    "required_tables": list(required_tables),
                    "missing_required_tables": sorted(required_set - selected_set),
                    "all_required_tables_retained": required_set <= selected_set,
                    "required_table_recall": round(len(retained_required) / len(required_set), 4),
                    "n_distractor_tables": len(distractors),
                    "distractor_table_rate": round(
                        len(distractors) / len(selected_set) if selected_set else 0.0,
                        4,
                    ),
                    "fk_connectivity_checkable": fk_checkable,
                    "fk_connectivity_retained": (
                        required_tables_fk_connected(
                            schema,
                            required_tables,
                            retained_tables=selection.selected_tables,
                        )
                        if fk_checkable
                        else None
                    ),
                    "schema_chars": selection.rendered_chars,
                    "schema_char_budget_exceeded": selection.exceeded_char_budget,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    git_sha, git_dirty = git_state(ignore_paths=(Path("results/runs.jsonl"),))
    tokenizer_json = Path(args.tokenizer) / "tokenizer.json" if args.tokenizer else None
    summary = {
        "command": shlex.join(sys.argv),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "questions_path": str(split.questions_path),
        "questions_sha256": file_sha256(split.questions_path),
        "n_examples": len(examples),
        "subset_seed": args.subset_seed if args.subset else None,
        "max_prompt_tokens": args.max_prompt_tokens,
        "tokenizer": args.tokenizer or None,
        "tokenizer_json_sha256": (
            file_sha256(tokenizer_json) if tokenizer_json and tokenizer_json.is_file() else None
        ),
        "schema_max_chars": args.schema_max_chars,
        "variants": {
            variant: _summarise(
                [row for row in rows if row["variant"] == variant],
                args.max_prompt_tokens,
                tokenizer is not None,
            )
            for variant in args.variants
        },
    }
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for variant, stats in summary["variants"].items():
        print(
            f"{variant:<7} n={stats['n']:>4} p50={stats['prompt_tokens_p50']:>5} "
            f"p99={stats['prompt_tokens_p99']:>5} max={stats['prompt_tokens_max']:>5} "
            f"over={stats['n_over_budget']:>3} tables={stats['mean_selected_tables']} "
            f"all_gold={stats['all_required_tables_retained_rate']:.3f} "
            f"fk={stats['fk_connectivity_retained_rate']} "
            f"distractors={stats['mean_distractor_tables']}"
        )
    print(f"wrote {args.out}")
    print(f"summary {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
