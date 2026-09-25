"""Mine DPO preference pairs from the model's own answers on the training split.

Preferred form -- several samples per question, scored by score_samples.py:

    python scripts/build_dpo_data.py \\
        --samples results/outcomes/train_4b_sft_k8.jsonl \\
        --out data/processed/dpo_4b.jsonl

Here both sides of a pair are queries the model wrote: the most frequent correct
one against the most frequent wrong one. Questions it never gets right fall back
to BIRD's gold as the chosen side, and that bucket is capped, because gold is
written in a house style the model does not use and a preference model would
rather learn the style than the semantics.

Older form -- one answer per question, from evaluate.py's outcomes:

    python scripts/build_dpo_data.py \\
        --outcomes results/outcomes/train_linked_4b.jsonl \\
        --out data/processed/dpo_4b.jsonl

Every pair is then gold against what the model wrote, which is the style problem
above applied to the whole file. Kept for the questions a sampling run has not
covered; not the way to build a training set.

Either input must come from the model you are about to train, on the prompt you
are about to train with. Pairs mined from a different checkpoint teach the model
to avoid mistakes it no longer makes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import (
    SCHEMA_MODES,
    SPLITS,
    PromptConfig,
    discover_split,
    load_schema,
    render_selected_schema,
)
from text2sql_rlvr.data.preference import (
    GOLD_FALLBACK,
    MinedPair,
    build_preference_record,
    cap_gold_fallback,
    mine_pair,
    mined_pair_report,
    normalise_sql,
    pair_is_usable,
    pair_report,
)
from text2sql_rlvr.data.sft import select_sft_examples
from text2sql_rlvr.ledger import file_sha256, git_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--questions", type=Path,
                        default=Path("data/processed/train_filtered.json"))
    parser.add_argument("--samples", type=Path, default=None,
                        help="per-sample outcomes from score_samples.py; pairs are mined from "
                             "the model's own answers wherever it got the question right at "
                             "least once")
    parser.add_argument("--outcomes", type=Path, default=None,
                        help="single-answer outcomes from evaluate.py; every pair is then gold "
                             "against the model's answer")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("configs/dpo/dataset.json"))

    parser.add_argument("--schema-mode", choices=tuple(m for m in SCHEMA_MODES if m != "oracle"),
                        default="linked", help="must match how the model is prompted")
    parser.add_argument("--schema-style", choices=("ddl", "compact"), default="ddl")
    parser.add_argument("--instruction-version", choices=("v1", "v2"), default="v1")

    parser.add_argument("--size", type=int, default=0, help="pairs to keep; 0 keeps all")
    parser.add_argument("--cap-per-db", type=int, default=0)
    parser.add_argument("--drop-gold-empty", action="store_true",
                        help="skip questions whose gold SQL returns no rows")
    parser.add_argument("--reasons", default="",
                        help="comma-separated strict-fail reasons to keep "
                             "(row_values, row_count, column_count, pred_failed); empty keeps all")
    parser.add_argument("--gold-fallback-fraction", type=float, default=0.25,
                        help="with --samples: the largest share of the result that may be pairs "
                             "whose chosen side is gold; 0 drops them entirely")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mine_from_samples(
    rows: list[dict], args: argparse.Namespace, wanted_reasons: set[str]
) -> tuple[list[MinedPair], dict, dict[int, dict]]:
    """One pair per question, from k scored samples.

    Also returns, per question, the scored sample the rejected query came from,
    so the manifest can still report *why* each rejected query was wrong.
    """
    by_question: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_question[int(row["question_id"])].append(row)

    pairs: list[MinedPair] = []
    rejection_rows: dict[int, dict] = {}
    for question_id, samples in sorted(by_question.items()):
        if args.drop_gold_empty and any(s.get("gold_empty") for s in samples):
            continue
        pair = mine_pair(
            [{"sql": s.get("pred_sql", ""), "official": bool(s["official"])} for s in samples],
            str(samples[0].get("gold_sql", "")),
            question_id=question_id,
            db_id=str(samples[0]["db_id"]),
        )
        if pair is None:
            continue
        rejected = normalise_sql(pair.rejected_sql)
        row = next(
            (s for s in samples if normalise_sql(str(s.get("pred_sql", ""))) == rejected),
            samples[0],
        )
        if wanted_reasons and str(row.get("reason")) not in wanted_reasons:
            continue
        pairs.append(pair)
        rejection_rows[question_id] = row

    counts = {
        "n_questions": len(by_question),
        "n_pairs_before_cap": len(pairs),
        "n_gold_fallback_before_cap": sum(1 for p in pairs if p.source == GOLD_FALLBACK),
    }
    return pairs, counts, rejection_rows


def pairs_from_single_answers(
    rows: list[dict], args: argparse.Namespace, wanted_reasons: set[str]
) -> tuple[list[MinedPair], dict]:
    """The older shape: gold as chosen, the model's one answer as rejected."""
    failures = [r for r in rows if not r["official"]]
    usable = [
        r for r in failures
        if pair_is_usable(r.get("gold_sql", ""), r.get("pred_sql", ""))
        and (not wanted_reasons or str(r.get("reason")) in wanted_reasons)
    ]
    pairs = [
        MinedPair(
            question_id=int(r["question_id"]),
            db_id=str(r["db_id"]),
            chosen_sql=str(r["gold_sql"]),
            rejected_sql=str(r["pred_sql"]),
            source=GOLD_FALLBACK,
            n_samples=1,
            n_correct=0,
            chosen_count=0,
            rejected_count=1,
        )
        for r in usable
        if not (args.drop_gold_empty and r.get("gold_empty"))
    ]
    counts = {
        "n_questions": len(rows),
        "n_failures": len(failures),
        "n_pairs_before_cap": len(pairs),
        "n_dropped_degenerate": len(failures) - len(usable),
    }
    return pairs, counts


def limit_size(pairs: list[MinedPair], args: argparse.Namespace) -> list[MinedPair]:
    """Round-robin over databases, so no single large database dominates."""
    if not args.size and not args.cap_per_db:
        return pairs
    keep = set(select_sft_examples(
        [{"question_id": p.question_id, "db_id": p.db_id, "official": False} for p in pairs],
        policy="all",
        size=args.size,
        cap_per_db=args.cap_per_db,
        seed=args.seed,
    ))
    return [p for p in pairs if p.question_id in keep]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.samples) == bool(args.outcomes):
        raise SystemExit("pass exactly one of --samples or --outcomes")

    wanted_reasons = {r.strip() for r in args.reasons.split(",") if r.strip()}
    source_path = args.samples or args.outcomes
    rows = load_rows(source_path)

    if args.samples:
        pairs, counts, rejection_rows = mine_from_samples(rows, args, wanted_reasons)
        print(f"questions  {counts['n_questions']} sampled, "
              f"{counts['n_pairs_before_cap']} produced a pair "
              f"({counts['n_gold_fallback_before_cap']} of them fall back to gold)")
        # Size first, then the cap: the cap is a property of the set that gets
        # written, so it has to be applied to the set that gets written.
        pairs = limit_size(pairs, args)
        before = len(pairs)
        pairs = cap_gold_fallback(pairs, max_fraction=args.gold_fallback_fraction, seed=args.seed)
        if before != len(pairs):
            print(f"capped     {before - len(pairs)} gold-fallback pairs dropped to keep them "
                  f"under {args.gold_fallback_fraction:.0%} of the set")
    else:
        pairs, counts = pairs_from_single_answers(rows, args, wanted_reasons)
        print(f"scored     {counts['n_questions']} questions, {counts['n_failures']} failed")
        print(f"usable     {counts['n_pairs_before_cap']} failures form a real pair "
              f"({counts['n_dropped_degenerate']} dropped: empty or identical to gold, "
              "or filtered by --reasons)")
        pairs = limit_size(pairs, args)
        rejection_rows = {int(r["question_id"]): r for r in rows}

    config = PromptConfig(
        schema_style=args.schema_style, instruction_version=args.instruction_version
    )
    split = replace(discover_split(args.root, args.split), questions_path=args.questions)
    keep = {p.question_id for p in pairs}
    examples = {e.question_id: e for e in split.load() if e.question_id in keep}
    pairs = [p for p in pairs if p.question_id in examples]
    print(f"kept       {len(pairs)} pairs from {len({p.db_id for p in pairs})} databases")

    schemas: dict[str, object] = {}
    records = []
    chars: list[int] = []
    for pair in pairs:
        example = examples[pair.question_id]
        if example.db_id not in schemas:
            schemas[example.db_id] = load_schema(
                split.db_path(example.db_id), db_id=example.db_id
            )
        schema_text, _ = render_selected_schema(
            schemas[example.db_id], example, mode=args.schema_mode, style=args.schema_style
        )
        record = build_preference_record(
            example,
            schema_text,
            pair.rejected_sql,
            config,
            chosen_sql=pair.chosen_sql,
        )
        record["pair_source"] = pair.source
        records.append(record)
        chars.append(
            sum(len(m["content"]) for m in record["prompt"])
            + max(len(record["chosen"][0]["content"]), len(record["rejected"][0]["content"]))
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = pair_report(records, rejection_rows)
    provenance = mined_pair_report(pairs)
    ordered = sorted(chars)
    lengths = {
        "chars_p50": ordered[len(ordered) // 2] if ordered else 0,
        "chars_p99": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] if ordered else 0,
        "chars_max": ordered[-1] if ordered else 0,
    }
    print(f"\npairs      {report['n']} over {report['n_databases']} databases, "
          f"{report['per_db_min']} to {report['per_db_max']} each")
    print(f"source     {provenance.get('by_source', {})}")
    print(f"rejected by execution status  {report['rejected_by_exec_status']}")
    print(f"rejected by mismatch reason   {report['rejected_by_reason']}")
    print(f"length     chars p50/p99/max {lengths['chars_p50']}/{lengths['chars_p99']}"
          f"/{lengths['chars_max']}")
    print(f"\nT1/T2 alias rate   chosen {provenance.get('alias_rate_chosen', 0):.1%}   "
          f"rejected {provenance.get('alias_rate_rejected', 0):.1%}")
    print("a large gap here means DPO can win by copying a writing style; check it again "
          "on what the trained model writes")

    sha, dirty = git_state(ignore_paths=(Path("results/runs.jsonl"),))
    manifest = {
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "source_kind": "samples" if args.samples else "single_answer",
        "source_questions": str(args.questions),
        "source_questions_sha256": file_sha256(args.questions),
        "output": str(args.out),
        "prompt_config": config.as_dict(),
        "schema_mode": args.schema_mode,
        "selection": {
            "size": args.size,
            "cap_per_db": args.cap_per_db,
            "drop_gold_empty": args.drop_gold_empty,
            "reasons": sorted(wanted_reasons),
            "gold_fallback_fraction": args.gold_fallback_fraction if args.samples else None,
            "seed": args.seed,
            **counts,
        },
        "report": report,
        "provenance": provenance,
        "length_stats": lengths,
        "git_sha": sha,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
