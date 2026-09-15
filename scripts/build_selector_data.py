"""Write the schema-selector training file and its two evaluation files.

    python scripts/build_selector_data.py --tokenizer data/tokenizers/qwen3-1.7b

Reads the selector split manifest and the generator's question files, labels
every question with the tables its gold SQL uses, and writes:

* ``train.jsonl``   chat-format examples, capped per database, hard cases first;
* ``heldout.jsonl`` prompts plus labels for the selector's own held-out databases;
* ``val788.jsonl``  prompts plus labels for the generator's fixed val, so the
  selector can later be plugged into the two-stage pipeline on the same 788.

The manifest records how the training questions were chosen and what the
current lexical linker scores on both evaluation files, which is the number a
trained selector has to beat.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import (
    BirdExample,
    DatabaseSchema,
    discover_split,
    format_schema,
    load_examples,
    load_schema,
)
from text2sql_rlvr.data.selector import (
    TARGET_FORMATS,
    SelectorCase,
    annotate_case,
    build_selector_messages,
    build_selector_record,
    gold_columns,
    select_training_cases,
    selection_metrics,
    trim_descriptions,
)
from text2sql_rlvr.ledger import file_sha256, git_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--split", type=Path, default=Path("configs/splits/selector_split.json"))
    parser.add_argument("--questions", type=Path,
                        default=Path("data/processed/train_filtered.json"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.json"),
                        help="the generator's fixed val; written as an evaluation file")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/selector/v1"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/selector/dataset_v1.json"))
    parser.add_argument("--cap-per-db", type=int, default=12,
                        help="training questions per database; 0 keeps all")
    parser.add_argument("--hard-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schema-style", choices=("ddl", "compact"), default="ddl")
    parser.add_argument("--descriptions", action="store_true",
                        help="append BIRD column descriptions to the schema text")
    parser.add_argument("--description-max-chars", type=int, default=80,
                        help="cut each column description to this many characters")
    parser.add_argument("--target", choices=TARGET_FORMATS, default="tables",
                        help="tables: JSON list of tables; columns: JSON object listing the "
                             "gold SQL's table.column references first, then the tables")
    parser.add_argument("--tokenizer", default="",
                        help="Qwen tokenizer directory for exact token counts; omitted skips them")
    parser.add_argument("--max-tokens", type=int, default=8192)
    return parser.parse_args(argv)


def _load_tokenizer(path: str):
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    generation = messages[-1]["role"] != "assistant"
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def _percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def _length_stats(chars: list[int], tokens: list[int], max_tokens: int) -> dict[str, object]:
    stats: dict[str, object] = {
        "n": len(chars),
        "chars_p50": _percentile(chars, 0.5),
        "chars_p99": _percentile(chars, 0.99),
        "chars_max": max(chars),
    }
    if tokens:
        stats.update({
            "token_count_kind": "exact_qwen_chat_template",
            "tokens_p50": _percentile(tokens, 0.5),
            "tokens_p95": _percentile(tokens, 0.95),
            "tokens_p99": _percentile(tokens, 0.99),
            "tokens_max": max(tokens),
            "n_over_max_tokens": sum(t > max_tokens for t in tokens),
        })
    return stats


def _case_stats(cases: list[SelectorCase]) -> dict[str, object]:
    reasons = Counter(reason for c in cases for reason in c.hard_reasons)
    return {
        "n": len(cases),
        "n_databases": len({c.db_id for c in cases}),
        "n_hard": sum(c.is_hard for c in cases),
        "hard_reasons": dict(sorted(reasons.items())),
        "gold_table_count": {
            str(k): v for k, v in sorted(Counter(len(c.gold_tables) for c in cases).items())
        },
        "questions_per_db": dict(sorted(Counter(c.db_id for c in cases).items())),
    }


class SchemaStore:
    def __init__(
        self, split, style: str, *, descriptions: bool, description_max_chars: int
    ) -> None:
        self._split = split
        self._style = style
        self._descriptions = descriptions
        self._max_chars = description_max_chars
        self._schemas: dict[str, DatabaseSchema] = {}
        self._texts: dict[str, str] = {}

    def schema(self, db_id: str) -> DatabaseSchema:
        if db_id not in self._schemas:
            self._schemas[db_id] = load_schema(self._split.db_path(db_id), db_id=db_id)
        return self._schemas[db_id]

    def text(self, db_id: str) -> str:
        if db_id not in self._texts:
            schema = self.schema(db_id)
            if self._descriptions:
                schema = trim_descriptions(schema, self._max_chars)
            self._texts[db_id] = format_schema(
                schema, style=self._style, include_descriptions=self._descriptions
            )
        return self._texts[db_id]


def _write_eval(
    path: Path,
    examples: list[BirdExample],
    store: SchemaStore,
    tokenizer,
    max_tokens: int,
    target_format: str,
) -> tuple[list[SelectorCase], dict[str, object]]:
    cases: list[SelectorCase] = []
    chars: list[int] = []
    tokens: list[int] = []
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            case = annotate_case(example, store.schema(example.db_id))
            messages = build_selector_messages(example, store.text(example.db_id), target_format)
            cases.append(case)
            chars.append(sum(len(m["content"]) for m in messages))
            if tokenizer is not None:
                tokens.append(_token_count(tokenizer, messages))
            handle.write(json.dumps({
                "question_id": case.question_id,
                "db_id": case.db_id,
                "question": example.question,
                "evidence": example.evidence,
                "messages": messages,
                "gold_tables": list(case.gold_tables),
                "all_tables": [t.name for t in store.schema(example.db_id).tables],
                "n_tables_total": case.n_tables_total,
                "linked_tables": list(case.linked_tables),
                "lexical_worst_rank": case.lexical_worst_rank,
                "hard_reasons": list(case.hard_reasons),
            }, ensure_ascii=False) + "\n")
    linked = {c.question_id: c.linked_tables for c in cases}
    report = {
        "path": str(path),
        **_case_stats(cases),
        "length": _length_stats(chars, tokens, max_tokens),
        "lexical_linker_baseline": selection_metrics(cases, linked),
    }
    return cases, report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_in = json.loads(args.split.read_text(encoding="utf-8"))
    train_dbs = {row["db_id"] for row in manifest_in["train_databases"]}
    heldout_dbs = {row["db_id"] for row in manifest_in["heldout_databases"]}
    excluded = set(manifest_in.get("generator_val_db_ids", ()))

    split = discover_split(args.root, "train")
    store = SchemaStore(
        split, args.schema_style,
        descriptions=args.descriptions, description_max_chars=args.description_max_chars,
    )
    tokenizer = _load_tokenizer(args.tokenizer)

    pool = load_examples(args.questions)
    val = load_examples(args.val)
    if {e.db_id for e in val} & (train_dbs | heldout_dbs):
        raise SystemExit("generator val databases overlap the selector split; rebuild the split")
    if {e.db_id for e in pool} & excluded:
        raise SystemExit("generator val databases are present in the selector question pool")
    train_pool = [e for e in pool if e.db_id in train_dbs]
    heldout = [e for e in pool if e.db_id in heldout_dbs]
    print(f"pool       {len(train_pool)} train candidates on {len(train_dbs)} databases, "
          f"{len(heldout)} held-out on {len(heldout_dbs)} databases, {len(val)} generator val")

    by_id = {e.question_id: e for e in train_pool}
    all_cases = [annotate_case(e, store.schema(e.db_id)) for e in train_pool]
    chosen = select_training_cases(
        all_cases, cap_per_db=args.cap_per_db, hard_fraction=args.hard_fraction, seed=args.seed
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    chars: list[int] = []
    tokens: list[int] = []
    n_unresolved = 0
    n_columns: list[int] = []
    with train_path.open("w", encoding="utf-8") as handle:
        for case in chosen:
            example = by_id[case.question_id]
            columns = None
            if args.target == "columns":
                columns, unresolved = gold_columns(store.schema(case.db_id), example.gold_sql)
                n_unresolved += bool(unresolved)
                n_columns.append(len(columns))
            record = build_selector_record(
                example, store.schema(case.db_id), store.text(case.db_id), case.gold_tables,
                target_format=args.target, gold_columns=columns,
            )
            messages = record["messages"]
            chars.append(sum(len(m["content"]) for m in messages))
            if tokenizer is not None:
                tokens.append(_token_count(tokenizer, messages))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    train_report = {
        "path": str(train_path),
        "selection": {
            "cap_per_db": args.cap_per_db,
            "hard_fraction": args.hard_fraction,
            "seed": args.seed,
        },
        "target": {
            "format": args.target,
            "n_with_unresolved_columns": n_unresolved,
            "mean_columns": round(sum(n_columns) / len(n_columns), 2) if n_columns else None,
        },
        "pool": _case_stats(all_cases),
        "chosen": _case_stats(chosen),
        "length": _length_stats(chars, tokens, args.max_tokens),
    }
    _, heldout_report = _write_eval(
        args.out_dir / "heldout.jsonl", heldout, store, tokenizer, args.max_tokens, args.target
    )
    _, val_report = _write_eval(
        args.out_dir / "val788.jsonl", val, store, tokenizer, args.max_tokens, args.target
    )

    def show(name: str, report: dict[str, object]) -> None:
        stats = report.get("chosen", report)
        length = report["length"]
        print(f"\n{name}: {stats['n']} questions, {stats['n_databases']} databases, "
              f"{stats['n_hard']} hard {dict(stats['hard_reasons'])}")
        print(f"  gold tables per question {stats['gold_table_count']}")
        line = (f"  chars p50/p99/max {length['chars_p50']}/{length['chars_p99']}"
                f"/{length['chars_max']}")
        if "tokens_p50" in length:
            line += (f"   tokens p50/p99/max {length['tokens_p50']}/{length['tokens_p99']}"
                     f"/{length['tokens_max']}   over {args.max_tokens}: "
                     f"{length['n_over_max_tokens']}")
        print(line)
        if "lexical_linker_baseline" in report:
            b = report["lexical_linker_baseline"]
            print(f"  lexical linker: all-gold {b['all_gold_retained_rate']:.4f}  "
                  f"precision {b['mean_precision']:.4f}  selected {b['mean_selected_tables']}"
                  f" of {b['mean_total_tables']}  gold {b['mean_gold_tables']}")

    show("train", train_report)
    if args.target == "columns":
        print(f"  columns target: mean {train_report['target']['mean_columns']} columns per "
              f"question, {n_unresolved} questions with unresolved column references")
    show("heldout", heldout_report)
    show("val788", val_report)
    print(f"\ntrain pool had {train_report['pool']['n']} candidates, "
          f"{train_report['pool']['n_hard']} hard")

    sha, dirty = git_state(ignore_paths=(Path("results/runs.jsonl"),))
    manifest = {
        "selector_split": str(args.split),
        "selector_split_sha256": file_sha256(args.split),
        "source_questions": str(args.questions),
        "source_questions_sha256": file_sha256(args.questions),
        "generator_val": str(args.val),
        "generator_val_sha256": file_sha256(args.val),
        "prompt": {
            "schema_style": args.schema_style,
            "descriptions": args.descriptions,
            "description_max_chars": args.description_max_chars if args.descriptions else 0,
            "target_format": args.target,
        },
        "schema_style": args.schema_style,
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "git_sha": sha,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
        "train": train_report,
        "heldout": heldout_report,
        "val788": val_report,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {train_path}")
    print(f"wrote {args.out_dir / 'heldout.jsonl'}")
    print(f"wrote {args.out_dir / 'val788.jsonl'}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
