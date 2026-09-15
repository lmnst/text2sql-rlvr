"""Choose which databases the schema selector trains on and which it is measured on.

    python scripts/build_selector_split.py

The generator's fixed val (3 databases, 59% of its questions on the 65-table
works_cycles) is too lopsided to judge a selector on its own. This holds out a
second set of databases from the generator's training portion, stratified by
table count, so selector metrics cover small, medium and large schemas. The
generator's val databases are excluded from both sides: the selector must not
train on them, or the two-stage score on val 788 would be contaminated.

Writes a small tracked manifest with the seed, buckets, database ids and
question ids. No data files: build_selector_data.py reads the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import discover_split, load_examples, load_schema
from text2sql_rlvr.data.selector import TABLE_COUNT_BUCKETS, bucket_of, plan_selector_split
from text2sql_rlvr.ledger import file_sha256, git_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--questions", type=Path,
                        default=Path("data/processed/train_filtered.json"),
                        help="the generator's filtered train questions")
    parser.add_argument("--generator-split", type=Path,
                        default=Path("configs/splits/train_val.json"),
                        help="manifest whose val_db_ids are excluded from both sides")
    parser.add_argument("--manifest", type=Path,
                        default=Path("configs/splits/selector_split.json"))
    parser.add_argument("--n-heldout", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-questions", type=int, default=40,
                        help="databases with fewer questions are never held out")
    parser.add_argument("--floor-per-bucket", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split = discover_split(args.root, "train")
    examples = load_examples(args.questions)
    generator_val_dbs = tuple(
        json.loads(args.generator_split.read_text(encoding="utf-8")).get("val_db_ids", ())
    )

    table_counts = {
        db_id: len(load_schema(split.db_path(db_id), db_id=db_id).tables)
        for db_id in sorted({e.db_id for e in examples})
    }
    plan = plan_selector_split(
        examples,
        table_counts,
        n_heldout=args.n_heldout,
        seed=args.seed,
        min_questions=args.min_questions,
        floor_per_bucket=args.floor_per_bucket,
        exclude_db_ids=generator_val_dbs,
    )

    questions_per_db = Counter(e.db_id for e in examples)
    leaked = sorted(set(generator_val_dbs) & set(questions_per_db))
    if leaked:
        print(f"note: generator val databases present in input and excluded: {leaked}")

    def describe(db_ids: tuple[str, ...]) -> list[dict[str, object]]:
        return [
            {
                "db_id": db_id,
                "n_tables": table_counts[db_id],
                "bucket": bucket_of(table_counts[db_id]),
                "n_questions": questions_per_db[db_id],
            }
            for db_id in db_ids
        ]

    print(f"input      {len(examples)} questions, {len(questions_per_db)} databases")
    print(f"buckets    {[list(b) for b in TABLE_COUNT_BUCKETS]} (by table count)")
    print(f"\nheld-out   {len(plan.heldout_db_ids)} databases, {len(plan.heldout_ids)} questions")
    for row in describe(plan.heldout_db_ids):
        print(f"  {row['db_id']:<30}{row['n_tables']:>4} tables  {row['n_questions']:>5} questions")
    print(f"\ntrain      {len(plan.train_db_ids)} databases, {len(plan.train_ids)} questions")
    train_buckets = Counter(bucket_of(table_counts[d]) for d in plan.train_db_ids)
    print(f"  databases per bucket: {[train_buckets[i] for i in range(len(TABLE_COUNT_BUCKETS))]}")

    sha, dirty = git_state(ignore_paths=(Path("results/runs.jsonl"),))
    manifest = {
        "source_questions": str(args.questions),
        "source_questions_sha256": file_sha256(args.questions),
        "generator_split": str(args.generator_split),
        "generator_val_db_ids": list(generator_val_dbs),
        "git_sha": sha,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
        **plan.summary(),
        "heldout_databases": describe(plan.heldout_db_ids),
        "train_databases": describe(plan.train_db_ids),
        "train_ids": list(plan.train_ids),
        "heldout_ids": list(plan.heldout_ids),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
