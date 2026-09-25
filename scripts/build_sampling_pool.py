"""Choose the train questions to sample k answers on for DPO pair mining.

    python scripts/build_sampling_pool.py --exclude data/processed/sft_4b_failure.jsonl \\
        --out data/processed/sample_pool_4b.json --manifest configs/dpo/sample_pool_4b.json

Questions the SFT adapter was trained on are excluded: the model has seen their
gold answer, answers them the same way eight times out of eight, and yields no
pair, so sampling them only costs time. Questions whose gold returns no rows are
dropped too, for the reason given in select_sft_examples. What is left is picked
round-robin over databases, so no single large database dominates the pairs.

The output is a questions file in the same format as train_filtered.json, ready
for generate.py --questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data.sft import select_sft_examples
from text2sql_rlvr.ledger import file_sha256, git_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path,
                        default=Path("data/processed/train_filtered.json"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("results/outcomes/train_linked_4b.jsonl"),
                        help="per-question outcomes on the same questions; only gold_empty "
                             "and db_id are read")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="SFT jsonl whose question_ids are left out; repeatable")
    parser.add_argument("--size", type=int, default=3000, help="questions; 0 keeps all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    excluded = {row["question_id"] for path in args.exclude for row in read_jsonl(path)}
    pool = [
        row for row in read_jsonl(args.outcomes) if row["question_id"] not in excluded
    ]
    chosen = set(
        select_sft_examples(pool, policy="all", size=args.size, drop_gold_empty=True,
                            seed=args.seed)
    )
    selected = [q for q in questions if q["question_id"] in chosen]
    if len(selected) != len(chosen):
        raise SystemExit(f"{len(chosen) - len(selected)} chosen ids are missing from "
                         f"{args.questions}; outcomes and questions do not match")

    per_db: dict[str, int] = {}
    for q in selected:
        per_db[q["db_id"]] = per_db.get(q["db_id"], 0) + 1
    base_failed = sum(1 for row in pool if row["question_id"] in chosen and not row["official"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selected, ensure_ascii=False, indent=1), encoding="utf-8")

    sha, dirty = git_state(ignore_paths=(Path("results/runs.jsonl"),))
    manifest = {
        "source_questions": str(args.questions),
        "source_questions_sha256": file_sha256(args.questions),
        "outcomes": str(args.outcomes),
        "outcomes_sha256": file_sha256(args.outcomes),
        "excluded": [{"path": str(p), "sha256": file_sha256(p)} for p in args.exclude],
        "n_excluded": len(excluded),
        "n_pool": len(pool),
        "size": args.size,
        "seed": args.seed,
        "output": str(args.out),
        "output_sha256": file_sha256(args.out),
        "n_selected": len(selected),
        "n_databases": len(per_db),
        "n_base_failed": base_failed,
        "questions_per_db": dict(sorted(per_db.items())),
        "git_sha": sha,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"excluded {len(excluded)} SFT questions, pool {len(pool)}")
    print(f"selected {len(selected)} questions over {len(per_db)} databases, "
          f"{base_failed} of them failed by the base model")
    print(f"wrote {args.out}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
