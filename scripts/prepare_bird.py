"""Check that a BIRD split is usable, and measure the ceiling before modelling.

Running every gold query first is not busywork. It tells you how many gold
queries error or time out (Execution Accuracy can never exceed that), and how
many return an empty result set -- the share of the benchmark that an
always-empty query would score on under the official set comparison.

    python scripts/prepare_bird.py --root data/bird --split mini_dev --check-gold
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import SPLITS, discover_split, format_schema, load_schema
from text2sql_rlvr.eval.surface import GoldFacts, gold_facts, summarise
from text2sql_rlvr.rewards.sandbox import SqlExecutor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--split", choices=SPLITS, default="mini_dev")
    parser.add_argument("--check-gold", action="store_true", help="execute every gold query")
    parser.add_argument("--timeout", type=float, default=30.0, help="gold execution timeout")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--show-schema", metavar="DB_ID", help="print one rendered schema")
    parser.add_argument("--facts-out", type=Path, default=None,
                        help="write per-question gold facts as jsonl")
    parser.add_argument("--from-facts", type=Path, default=None,
                        help="re-print the report from a previous --facts-out file, "
                             "executing nothing")
    return parser.parse_args(argv)


def read_facts(path: Path) -> list[GoldFacts]:
    """Load per-question gold facts written by an earlier run."""
    facts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            facts.append(GoldFacts(**json.loads(line)))
    return facts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    split = discover_split(args.root, args.split)
    examples = split.load()
    db_ids = sorted({e.db_id for e in examples})

    print(f"split           {split.name}")
    print(f"questions       {split.questions_path}")
    print(f"databases       {split.databases_dir}")
    print(f"examples        {len(examples)}")
    print(f"databases used  {len(db_ids)}")
    print(f"difficulty      {dict(Counter(e.difficulty or 'unknown' for e in examples))}")
    print(f"without gold    {sum(1 for e in examples if not e.gold_sql)}")

    missing = split.missing_databases(examples)
    if missing:
        print(f"\nMISSING DATABASE FILES ({len(missing)}): {', '.join(missing)}")
        return 1

    if args.show_schema:
        schema = load_schema(split.db_path(args.show_schema), db_id=args.show_schema)
        print(f"\n--- schema for {args.show_schema} ---")
        print(format_schema(schema, style="ddl"))

    if args.from_facts:
        # Re-print a report from a previous run. Executing nine thousand gold
        # queries takes half an hour; losing the terminal should not cost that.
        facts = read_facts(args.from_facts)
        print(f"\nreplaying {len(facts)} gold results from {args.from_facts}")
        print("(no queries executed)")
    elif args.check_gold:
        print("\nexecuting gold queries...")
        facts = []
        with SqlExecutor(timeout_s=args.timeout) as executor:
            from concurrent.futures import ThreadPoolExecutor

            def run(example):
                result = executor.execute(split.db_path(example.db_id), example.gold_sql)
                return gold_facts(example, result)

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for done, item in enumerate(pool.map(run, examples), start=1):
                    facts.append(item)
                    if done % 200 == 0:
                        print(f"  {done}/{len(examples)}", flush=True)
    else:
        return 0

    statuses = Counter(f.status for f in facts)
    surface = summarise(facts)
    total = surface.n

    print(f"\ngold status     {dict(statuses)}")
    print(f"gold executable {surface.n_executable}/{total} "
          f"({surface.rate(surface.n_executable):.1f}%)  <- ceiling on EX")
    if surface.failed_ids:
        shown = ", ".join(str(i) for i in surface.failed_ids[:20])
        more = "" if len(surface.failed_ids) <= 20 else f" (+{len(surface.failed_ids) - 20} more)"
        print(f"  failed ids    {shown}{more}")

    slowest = sorted(facts, key=lambda f: f.elapsed_s, reverse=True)[:5]
    print("  slowest       " + ", ".join(f"#{f.question_id} {f.elapsed_s:.1f}s" for f in slowest))

    print("\nexploitable surface under the official set comparison")
    print(f"  gold empty              {surface.n_empty:>5} ({surface.rate(surface.n_empty):>5.1f}%)"
          "  <- free credit for an always-empty query")
    print(f"  gold has duplicate rows {surface.n_with_duplicates:>5} "
          f"({surface.rate(surface.n_with_duplicates):>5.1f}%)"
          "  <- free credit for SELECT DISTINCT")
    print(f"  either                  {surface.n_exploitable:>5} "
          f"({surface.rate(surface.n_exploitable):>5.1f}%)"
          "  <- total room for result-shape tricks")

    print("\nresult shape (context for the numbers above)")
    print(f"  single row              {surface.n_single_row:>5} "
          f"({surface.rate(surface.n_single_row):>5.1f}%)")
    print(f"  single column           {surface.n_single_column:>5} "
          f"({surface.rate(surface.n_single_column):>5.1f}%)")
    print(f"  gold has ORDER BY       {surface.n_ordered:>5} "
          f"({surface.rate(surface.n_ordered):>5.1f}%)  <- order the official metric ignores")

    if args.facts_out:
        args.facts_out.parent.mkdir(parents=True, exist_ok=True)
        with args.facts_out.open("w", encoding="utf-8") as handle:
            for item in facts:
                handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        print(f"\nper-question gold facts written to {args.facts_out}")

    return 0 if surface.n_executable == total else 2


if __name__ == "__main__":
    sys.exit(main())
