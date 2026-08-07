"""Build a filtered training set and a validation set from BIRD's train split.

    python scripts/build_splits.py --facts results/gold_facts_train.jsonl

Writes the question files under data/processed/ (regenerable, not tracked) and a
manifest under configs/splits/ (small, tracked) that records the seed, the
filters and the exact question ids, so any later result can be tied to the
split that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import discover_split, load_examples
from text2sql_rlvr.data.splits import VAL_STRATEGIES, plan_splits
from text2sql_rlvr.eval.surface import GoldFacts, summarise
from text2sql_rlvr.ledger import file_sha256, git_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--facts", type=Path, required=True,
                        help="gold facts jsonl from prepare_bird.py --facts-out")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("configs/splits/train_val.json"))
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--strategy", choices=VAL_STRATEGIES, default="db_disjoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-gold-seconds", type=float, default=10.0,
                        help="drop questions whose gold was slower than this; 0 keeps all")
    return parser.parse_args(argv)


def read_facts(path: Path) -> list[GoldFacts]:
    return [
        GoldFacts(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_questions(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")


def describe(name: str, ids: tuple[int, ...], facts_by_id: dict[int, GoldFacts]) -> None:
    surface = summarise([facts_by_id[i] for i in ids if i in facts_by_id])
    print(f"\n{name}: {len(ids)} questions")
    print(f"  gold empty            {surface.n_empty:>5} ({surface.rate(surface.n_empty):>5.1f}%)")
    print(f"  gold duplicate rows   {surface.n_with_duplicates:>5} "
          f"({surface.rate(surface.n_with_duplicates):>5.1f}%)")
    print(f"  exploitable           {surface.n_exploitable:>5} "
          f"({surface.rate(surface.n_exploitable):>5.1f}%)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    split = discover_split(args.root, "train")
    examples = load_examples(split.questions_path)
    facts = read_facts(args.facts)
    facts_by_id = {f.question_id: f for f in facts}

    plan = plan_splits(
        examples,
        facts,
        val_size=args.val_size,
        strategy=args.strategy,
        seed=args.seed,
        max_gold_seconds=args.max_gold_seconds,
    )

    raw = json.loads(split.questions_path.read_text(encoding="utf-8"))
    by_id = {int(item.get("question_id", i)): item for i, item in enumerate(raw)}
    train_path = args.out_dir / "train_filtered.json"
    val_path = args.out_dir / "val.json"
    write_questions(train_path, [by_id[i] for i in plan.train_ids])
    write_questions(val_path, [by_id[i] for i in plan.val_ids])

    summary = plan.summary()
    print(f"input                 {len(examples)} questions from {split.questions_path}")
    print(f"excluded              {summary['n_excluded']}")
    for reason, count in sorted(summary["excluded_by_reason"].items()):
        print(f"  {reason:<22}{count}")
    describe("train", plan.train_ids, facts_by_id)
    describe("val", plan.val_ids, facts_by_id)
    print(f"\nval databases         {len(plan.val_db_ids)}: {', '.join(plan.val_db_ids)}")
    print("  (train shares none of these, matching how dev relates to train)")

    sha, dirty = git_state()
    manifest = {
        "source_questions": str(split.questions_path),
        "source_facts": str(args.facts),
        "source_facts_sha256": file_sha256(args.facts),
        "git_sha": sha,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
        **summary,
        "val_db_ids": list(plan.val_db_ids),
        "train_ids": list(plan.train_ids),
        "val_ids": list(plan.val_ids),
        "excluded_ids": {k: list(v) for k, v in plan.excluded.items()},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"\nwrote {train_path}")
    print(f"wrote {val_path}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
