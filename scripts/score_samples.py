"""Execute every sample of a k-sample generation run and bucket the questions.

    python scripts/score_samples.py --root data/bird --split train \\
        --questions data/processed/train_filtered.json \\
        --predictions results/preds/train_4b_sft_k8.jsonl \\
        --out results/outcomes/train_4b_sft_k8.jsonl

The headline of the printed report is not a score, it is the size of the
minable pool: how many questions the model got right *sometimes*. Those are the
ones that yield a preference pair made of the model's own answers.

Works on a single-sample predictions file too -- it is then the same thing
evaluate.py does, minus the split-level metric -- so the same command serves
both shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import SPLITS, discover_split
from text2sql_rlvr.eval import score_samples
from text2sql_rlvr.ledger import DEFAULT_LEDGER, append_run
from text2sql_rlvr.rewards.compare import DEFAULT_ORDER_POLICY, ORDER_POLICIES
from text2sql_rlvr.rewards.sandbox import SqlExecutor
from text2sql_rlvr.sql import extract_sql

_STAGES = ("baseline", "sft", "grpo", "ablation", "smoke")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--questions", type=Path, default=None,
                        help="custom BIRD-format question file, scored against --split's databases")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="generate.py output, normally written with --n-samples above 1")
    parser.add_argument("--out", type=Path, required=True,
                        help="per-sample outcomes jsonl, the input to build_dpo_data.py")

    parser.add_argument("--order-policy", choices=ORDER_POLICIES, default=DEFAULT_ORDER_POLICY)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N questions, 0 for all")

    parser.add_argument("--stage", choices=_STAGES, default="ablation")
    parser.add_argument("--checkpoint", default="", help="checkpoint path or model id")
    parser.add_argument("--notes", default="")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true", help="print only, record nothing")
    return parser.parse_args(argv)


def load_samples(path: Path) -> dict[int, list[str]]:
    """Read the SQL of every sample per question.

    A record written with --n-samples above 1 carries a ``samples`` list; one
    written without it carries a single answer under the old keys. Both are read
    here so a file generated either way can be scored by the same command.
    """
    samples: dict[int, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        question_id = int(record["question_id"])
        if record.get("samples"):
            sqls = [str(s.get("sql") or extract_sql(s.get("completion") or ""))
                    for s in record["samples"]]
        else:
            sqls = [str(record.get("sql") or extract_sql(record.get("completion") or ""))]
        samples[question_id] = sqls
    return samples


def load_meta(path: Path) -> dict:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}


def print_report(report, path: Path) -> None:
    print(f"\nquestions            {report.n_questions}")
    print(f"samples              {report.n_samples_total} (k up to {report.k_max})")
    print(f"\npass@1               {report.pass_at_1:.2f}%   <- first sample only")
    print(f"pass@k               {report.pass_at_k:.2f}%   <- solved at least once")
    print(f"mean pass rate       {report.mean_pass_rate:.2f}%   <- per question, across samples")

    minable = report.n_mixed + report.n_all_wrong
    print("\nquestion buckets")
    print(f"  all samples correct   {report.n_all_correct:>6}   no pair, nothing to prefer")
    print(f"  mixed                 {report.n_mixed:>6}   a pair from the model's own answers")
    print(f"  all samples wrong     {report.n_all_wrong:>6}   a pair only if gold stands in")
    print(f"\nminable questions    {minable} ({report.n_mixed} of them without gold)")
    print(f"per-sample outcomes  {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    split = discover_split(args.root, args.split)
    if args.questions:
        split = replace(split, questions_path=args.questions)
        print(f"questions overridden: {args.questions}")
    examples = split.load()
    samples = load_samples(args.predictions)
    meta = load_meta(args.predictions)

    examples = [e for e in examples if e.question_id in samples]
    if args.limit:
        examples = examples[: args.limit]
    print(f"{len(examples)} questions with samples from {args.predictions}")

    with SqlExecutor(timeout_s=args.timeout) as executor:
        report = score_samples(
            examples,
            samples,
            split,
            executor=executor,
            order_policy=args.order_policy,
            n_workers=args.workers,
            on_progress=lambda done, total: print(f"  sample {done}/{total}", flush=True),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for _question_id, outcomes in sorted(report.outcomes.items()):
            for index, outcome in enumerate(outcomes):
                row = {"sample_index": index, **asdict(outcome)}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_report(report, args.out)

    if args.no_ledger:
        print("ledger               skipped (--no-ledger)")
        return 0

    entry = append_run(
        {
            "stage": args.stage,
            "split": split.name,
            "n_samples": report.n_questions,
            "seed": meta.get("decoding", {}).get("seed"),
            "decoding": meta.get("decoding", {}),
            "prompt_config": meta.get("prompt_config", {}),
            "model": meta.get("model", ""),
            "checkpoint": args.checkpoint or meta.get("model", ""),
            "config_path": "",
            "command": " ".join(sys.argv),
            "order_policy": args.order_policy,
            "eval_subset": "sampling",
            "metrics": report.metrics(),
            "log_path": str(args.out),
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
