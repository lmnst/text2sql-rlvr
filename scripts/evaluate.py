"""Score a predictions file and append one line to the experiment ledger.

    python scripts/evaluate.py --root data/bird --split mini_dev \\
        --predictions results/preds/base_minidev.jsonl --stage baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import SPLITS, discover_split
from text2sql_rlvr.eval import evaluate
from text2sql_rlvr.ledger import DEFAULT_LEDGER, append_run
from text2sql_rlvr.rewards.compare import DEFAULT_ORDER_POLICY, ORDER_POLICIES
from text2sql_rlvr.rewards.sandbox import SqlExecutor

_STAGES = ("baseline", "sft", "grpo", "ablation", "smoke")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--split", choices=SPLITS, default="mini_dev")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--stage", choices=_STAGES, required=True)

    parser.add_argument("--order-policy", choices=ORDER_POLICIES, default=DEFAULT_ORDER_POLICY)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--checkpoint", default="", help="checkpoint path or model id")
    parser.add_argument("--config-path", default="", help="training config this run came from")
    parser.add_argument("--notes", default="")
    parser.add_argument("--outcomes", type=Path, default=None, help="per-example jsonl output")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true", help="print only, record nothing")
    return parser.parse_args(argv)


def load_predictions(path: Path) -> dict[int, str]:
    """Read a predictions jsonl. Uses ``completion`` when present, else ``sql``."""
    predictions: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = record.get("completion") or record.get("sql") or ""
        predictions[int(record["question_id"])] = text
    return predictions


def load_meta(path: Path) -> dict:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def print_report(report) -> None:
    print(f"\nsplit                {report.split}  (n={report.n})")
    print(f"order policy         {report.order_policy}")
    print(f"\nofficial EX          {report.official_ex:.2f}%   <- the BIRD metric, report this")
    print(f"strict EX            {report.strict_ex:.2f}%   <- our verifier")

    if report.by_difficulty:
        print("\nby difficulty")
        for name, stats in report.by_difficulty.items():
            print(f"  {name:<12} n={stats['n']:<6} official={stats['official_ex']:>6.2f}%"
                  f"  strict={stats['strict_ex']:>6.2f}%")

    print("\nverifier gap")
    print(f"  official pass / strict fail   {report.n_official_not_strict}")
    print(f"  strict pass / official fail   {report.n_strict_not_official}")
    print(f"  gold returns empty            {report.n_gold_empty}"
          f" (official credits {report.n_gold_empty_official_pass})")

    print("\nprediction status    " + str(report.pred_status_counts))
    print("gold status          " + str(report.gold_status_counts))
    print("strict fail reasons  " + str(report.strict_reason_counts))
    if report.n_missing_predictions or report.n_unparsed:
        print(f"\nmissing predictions  {report.n_missing_predictions}")
        print(f"unparsable output    {report.n_unparsed}")
    print("executor             " + str(report.executor_stats))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    split = discover_split(args.root, args.split)
    examples = split.load()
    predictions = load_predictions(args.predictions)
    meta = load_meta(args.predictions)

    with SqlExecutor(timeout_s=args.timeout) as executor:
        report = evaluate(
            examples,
            predictions,
            split,
            executor=executor,
            order_policy=args.order_policy,
            n_workers=args.workers,
        )

    print_report(report)

    outcomes_path = args.outcomes or Path("results/outcomes") / f"{args.predictions.stem}.jsonl"
    report.write_outcomes(outcomes_path)
    print(f"\nper-example outcomes {outcomes_path}")

    if args.no_ledger:
        print("ledger               skipped (--no-ledger)")
        return 0

    entry = append_run(
        {
            "stage": args.stage,
            "split": split.name,
            "n_samples": report.n,
            "seed": meta.get("decoding", {}).get("seed"),
            "decoding": meta.get("decoding", {}),
            "prompt_config": meta.get("prompt_config", {}),
            "model": meta.get("model", ""),
            "checkpoint": args.checkpoint or meta.get("model", ""),
            "config_path": args.config_path,
            "command": " ".join(sys.argv),
            "order_policy": args.order_policy,
            "metrics": report.metrics(),
            "log_path": str(outcomes_path),
            "notes": args.notes,
        },
        path=args.ledger,
    )
    print(f"ledger               {args.ledger} (run_id={entry['run_id']})")
    if entry["git_dirty"]:
        print("WARNING: working tree is dirty; this run is not reportable (see AGENTS.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
