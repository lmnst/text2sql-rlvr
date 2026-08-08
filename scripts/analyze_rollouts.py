"""Read the rollout log a training run produced and show how the policy drifted.

    python scripts/analyze_rollouts.py --rollouts out/grpo/rollouts.jsonl

The log is written by verl_reward.py, one line per rollout, independent of what
verl does with the return value. Bucketing it in order gives the shape of the
run: whether accuracy rose, and whether the policy started collecting reward in
ways that do not answer the question.

Two columns matter more than the reward curve:

* ``no_from``  -- rollouts whose SQL reads no table at all (``SELECT 1``). Under
  a reward that pays for merely executing, this is the shortest path to a score.
* ``hack``     -- rollouts that earned reward the strict verifier rejects.

If those rise while accuracy does not, the policy is optimising the measurement
rather than the task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=10,
                        help="split the run into this many equal slices, in order")
    parser.add_argument("--examples", type=int, default=3,
                        help="degenerate completions to print from the final slice")
    return parser.parse_args(argv)


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "error" not in record:
            rows.append(record)
    return rows


def summarise(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    if not n:
        return {}
    return {
        "n": n,
        "reward": sum(r["reward"] for r in rows) / n,
        "strict": sum(r["correct"] for r in rows) / n,
        "official": sum(r["official"] for r in rows) / n,
        "exec": sum(r["executed"] for r in rows) / n,
        "no_from": sum(r["no_from_clause"] for r in rows) / n,
        "empty": sum(r["empty_result"] for r in rows) / n,
        "hack": sum(1 for r in rows if r["reward"] > 0 and not r["correct"]) / n,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load(args.rollouts)
    if not rows:
        print(f"no usable rollouts in {args.rollouts}")
        return 1

    print(f"{len(rows)} rollouts from {args.rollouts}\n")
    header = f"{'slice':>7}{'n':>8}{'reward':>9}{'strict':>9}{'official':>10}"
    header += f"{'exec':>8}{'no_from':>9}{'empty':>8}{'hack':>8}"
    print(header)
    print("-" * len(header))

    size = max(1, len(rows) // args.buckets)
    slices = [rows[i:i + size] for i in range(0, len(rows), size)][: args.buckets]
    for index, chunk in enumerate(slices):
        s = summarise(chunk)
        print(f"{index + 1:>7}{s['n']:>8}{s['reward']:>9.3f}{s['strict']:>9.3f}"
              f"{s['official']:>10.3f}{s['exec']:>8.3f}{s['no_from']:>9.3f}"
              f"{s['empty']:>8.3f}{s['hack']:>8.3f}")

    first, last = summarise(slices[0]), summarise(slices[-1])
    print("\nfirst slice -> last slice")
    for key in ("strict", "official", "no_from", "empty", "hack"):
        delta = last[key] - first[key]
        print(f"  {key:<10}{first[key]:>7.3f} -> {last[key]:>6.3f}   {delta:+.3f}")

    print("\nreading it: no_from and hack rising while strict does not means the")
    print("policy found a way to be paid without answering. That is the finding,")
    print("not a bug -- but only if the reward config that produced it is recorded.")

    degenerate = [r for r in slices[-1] if r["no_from_clause"] or
                  (r["reward"] > 0 and not r["correct"])]
    if degenerate and args.examples:
        print(f"\ndegenerate completions from the final slice ({len(degenerate)} found):")
        for row in degenerate[: args.examples]:
            print(f"  reward={row['reward']:<5} strict={row['correct']!s:<5} "
                  f"official={row['official']!s:<5} rows={row['pred_n_rows']}")
            print(f"    {' '.join((row.get('sql') or '').split())[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
