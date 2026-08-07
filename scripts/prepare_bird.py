"""Check that a BIRD split is usable, and measure the ceiling before modelling.

Running every gold query first is not busywork. It tells you how many gold
queries error or time out (Execution Accuracy can never exceed that), and how
many return an empty result set -- the share of the benchmark that an
always-empty query would score on under the official set comparison.

    python scripts/prepare_bird.py --root data/bird --split mini_dev --check-gold
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import SPLITS, discover_split, format_schema, load_schema
from text2sql_rlvr.rewards.sandbox import OK, SqlExecutor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--split", choices=SPLITS, default="mini_dev")
    parser.add_argument("--check-gold", action="store_true", help="execute every gold query")
    parser.add_argument("--timeout", type=float, default=30.0, help="gold execution timeout")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--show-schema", metavar="DB_ID", help="print one rendered schema")
    return parser.parse_args(argv)


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

    if not args.check_gold:
        return 0

    print("\nexecuting gold queries...")
    statuses: Counter[str] = Counter()
    empty = 0
    with SqlExecutor(timeout_s=args.timeout) as executor:
        from concurrent.futures import ThreadPoolExecutor

        def run(example):
            return executor.execute(split.db_path(example.db_id), example.gold_sql)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for done, result in enumerate(pool.map(run, examples), start=1):
                statuses[result.status] += 1
                if result.status == OK and result.n_rows == 0:
                    empty += 1
                if done % 200 == 0:
                    print(f"  {done}/{len(examples)}", flush=True)

    total = len(examples)
    ok = statuses.get(OK, 0)
    print(f"\ngold status     {dict(statuses)}")
    print(f"gold executable {ok}/{total} ({100.0 * ok / total:.1f}%)  <- ceiling on EX")
    print(f"gold empty      {empty}/{total} ({100.0 * empty / total:.1f}%)  "
          f"<- free credit for an always-empty query under official EX")
    return 0 if ok == total else 2


if __name__ == "__main__":
    sys.exit(main())
