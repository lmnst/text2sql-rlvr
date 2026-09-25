"""Score a schema selector served through any OpenAI-compatible endpoint.

    python scripts/evaluate_selector.py \\
        --eval data/processed/selector/v1/heldout.jsonl \\
        --model selector --out results/selector/heldout_v1.jsonl

Like generate.py this is a plain HTTP client: the GPU box only runs
``vllm serve`` and the scoring stays local. Each question's prompt is taken
verbatim from the evaluation file written by build_selector_data.py, so the
selector is scored on exactly the input it was trained on.

Three selections are scored side by side on the same questions: the raw model
answer, that answer after the recall-oriented expansion (foreign-key
neighbours plus the best lexical matches, see ``--expand``), and the lexical
linker. Every gold table kept, precision and the number of tables kept have to
move together; a selector that keeps everything wins the first and loses the
other two. The expanded selection is written as ``expanded_tables`` so
generate.py can hand it to the SQL generator.

Re-running with ``--resume`` on a finished output file sends no requests and
just rescoring it, which is how a different ``--expand`` setting is tried
without the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
import httpx

from text2sql_rlvr.data import SPLITS, DatabaseSchema, discover_split, load_schema
from text2sql_rlvr.data.selector import (
    SelectorCase,
    expand_selection,
    parse_selector_output,
    selection_report,
)
from text2sql_rlvr.ledger import append_run, file_sha256

EXPAND_MODES = ("none", "fk", "fk+lex")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", type=Path, required=True,
                        help="heldout.jsonl or val788.jsonl from build_selector_data.py")
    parser.add_argument("--out", type=Path, required=True, help="per-question predictions")
    parser.add_argument("--split-name", default="",
                        help="ledger split label; defaults to selector-heldout or val by file name")
    parser.add_argument("--root", type=Path, default=Path("data/bird"),
                        help="BIRD root; schemas are needed for --expand")
    parser.add_argument("--split", choices=SPLITS, default="train",
                        help="which split's databases the evaluation questions belong to")

    parser.add_argument("--expand", choices=EXPAND_MODES, default="fk+lex",
                        help="recall-oriented expansion of the model answer: none, foreign-key "
                             "neighbours, or neighbours plus the best lexical matches")
    parser.add_argument("--fk-hops", type=int, default=1)
    parser.add_argument("--lex-top-k", type=int, default=2)
    parser.add_argument("--expand-cap", type=int, default=0,
                        help="stop adding foreign-key neighbours once this many tables are "
                             "kept; 0 keeps every neighbour")

    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)

    parser.add_argument("--n-samples", type=int, default=1,
                        help="sample this many answers per question and take the union of "
                             "their tables; needs --temperature > 0 to differ")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking", action="store_true",
                        help="enable Qwen3 thinking mode (off by default; see AGENTS.md)")

    parser.add_argument("--limit", type=int, default=0, help="first N questions, 0 for all")
    parser.add_argument("--resume", action="store_true",
                        help="skip ids already in --out; with all ids present this only rescores")
    parser.add_argument("--no-ledger", action="store_true",
                        help="do not append to results/runs.jsonl (smoke runs, rescoring)")
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def load_eval(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for row in rows:
        if "all_tables" not in row:
            raise SystemExit(f"{path} lacks all_tables; rebuild it with build_selector_data.py")
    return rows


def to_case(row: dict) -> SelectorCase:
    return SelectorCase(
        question_id=int(row["question_id"]),
        db_id=row["db_id"],
        gold_tables=tuple(row["gold_tables"]),
        n_tables_total=int(row["n_tables_total"]),
        linked_tables=tuple(row["linked_tables"]),
        lexical_worst_rank=int(row["lexical_worst_rank"]),
    )


def load_done(path: Path) -> dict[int, dict]:
    """Records already answered; a record whose request failed is not done."""
    if not path.is_file():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("error"):
                continue
            done[int(record["question_id"])] = record
    return done


def print_block(name: str, block: dict) -> None:
    def cells(m: dict) -> str:
        return (f"all-gold {m['all_gold_retained_rate']:.4f}  precision {m['mean_precision']:.4f}"
                f"  kept {m['mean_selected_tables']:>6.2f}")

    line = f"  {name:<28} n={block['model']['n']:>5}  model {cells(block['model'])}"
    if "expanded" in block:
        line += f"\n  {'':<28}         expanded {cells(block['expanded'])}"
    if "baseline" in block:
        line += f"\n  {'':<28}         linker {cells(block['baseline'])}"
    print(line)


class SchemaStore:
    def __init__(self, root: Path, split_name: str) -> None:
        self._split = None
        self._root, self._split_name = root, split_name
        self._schemas: dict[str, DatabaseSchema] = {}

    def get(self, db_id: str) -> DatabaseSchema:
        if db_id not in self._schemas:
            if self._split is None:
                self._split = discover_split(self._root, self._split_name)
            self._schemas[db_id] = load_schema(self._split.db_path(db_id), db_id=db_id)
        return self._schemas[db_id]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_eval(args.eval)
    if args.limit:
        rows = rows[: args.limit]
    if args.n_samples > 1 and args.temperature <= 0:
        raise SystemExit("--n-samples above 1 needs --temperature above 0")
    by_id = {int(row["question_id"]): row for row in rows}
    cases = [to_case(row) for row in rows]
    split_name = args.split_name or ("selector-heldout" if "heldout" in args.eval.name else "val")

    done = load_done(args.out) if args.resume else {}
    done = {qid: rec for qid, rec in done.items() if qid in by_id}
    todo = [row for row in rows if int(row["question_id"]) not in done]
    print(f"{len(todo)} to query ({len(done)} already present) from {args.eval}")

    body_extra: dict[str, object] = {}
    if not args.thinking:
        body_extra["chat_template_kwargs"] = {"enable_thinking": False}

    client = httpx.Client(
        base_url=args.base_url.rstrip("/"),
        timeout=args.request_timeout,
        headers={"Authorization": f"Bearer {args.api_key}"},
    )

    def query(row: dict) -> dict:
        payload = {
            "model": args.model,
            "messages": row["messages"],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "n": args.n_samples,
            **body_extra,
        }
        started = time.monotonic()
        completions: list[str] = []
        finish, usage, error = None, {}, ""
        for attempt in range(args.retries + 1):
            try:
                response = client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                completions = [c["message"]["content"] or "" for c in data["choices"]]
                finish = data["choices"][0].get("finish_reason")
                usage = data.get("usage", {})
                error = ""
                break
            except Exception as exc:  # noqa: BLE001 - retried, then recorded
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(2**attempt)
        samples = [
            list(parse_selector_output(text, row["all_tables"])) if text else []
            for text in completions
        ]
        union = {name for tables in samples for name in tables}
        predicted = tuple(name for name in row["all_tables"] if name in union)
        return {
            "question_id": int(row["question_id"]),
            "db_id": row["db_id"],
            "completion": completions[0] if completions else "",
            "n_samples": len(completions),
            "samples": samples,
            "finish_reason": finish,
            "usage": usage,
            "latency_s": round(time.monotonic() - started, 3),
            "error": error or None,
            "predicted_tables": list(predicted),
            "gold_tables": list(row["gold_tables"]),
            "linked_tables": list(row["linked_tables"]),
            "n_tables_total": row["n_tables_total"],
            "hard_reasons": list(row.get("hard_reasons", ())),
        }

    # Stream new answers to disk as they arrive (crash safety), then rewrite the
    # whole file once scoring fields are attached to every record.
    write_lock = threading.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    records = dict(done)
    if todo:
        handle = args.out.open("a" if args.resume else "w", encoding="utf-8")
        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for n, record in enumerate(pool.map(query, todo), start=1):
                    if record["error"]:
                        failures += 1
                    records[record["question_id"]] = record
                    with write_lock:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                    if n % 100 == 0:
                        print(f"  {n}/{len(todo)}", flush=True)
        finally:
            handle.close()
    client.close()

    schemas = SchemaStore(args.root, args.split)
    expansion = None
    if args.expand != "none":
        expansion = {
            "mode": args.expand,
            "fk_hops": args.fk_hops,
            "lex_top_k": args.lex_top_k if args.expand == "fk+lex" else 0,
            "cap": args.expand_cap or None,
        }
    for qid, record in records.items():
        row = by_id[qid]
        gold = {name.casefold() for name in row["gold_tables"]}
        predicted = tuple(record["predicted_tables"])
        if expansion is None:
            expanded = predicted
        else:
            expanded = expand_selection(
                schemas.get(row["db_id"]),
                predicted,
                question=row.get("question", ""),
                evidence=row.get("evidence", ""),
                fk_hops=expansion["fk_hops"],
                lex_top_k=expansion["lex_top_k"],
                cap=expansion["cap"],
            )
        kept = {name.casefold() for name in predicted}
        kept_expanded = {name.casefold() for name in expanded}
        record.update({
            "all_gold_retained": gold <= kept,
            "precision": round(len(gold & kept) / len(kept), 4) if kept else 0.0,
            "n_selected": len(predicted),
            "expanded_tables": list(expanded),
            "expanded_all_gold_retained": gold <= kept_expanded,
            "expanded_n_selected": len(expanded),
        })

    ordered = [
        records[int(row["question_id"])] for row in rows if int(row["question_id"]) in records
    ]
    args.out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )

    predicted_map = {qid: tuple(r["predicted_tables"]) for qid, r in records.items()}
    expanded_map = {qid: tuple(r["expanded_tables"]) for qid, r in records.items()}
    baseline = {c.question_id: c.linked_tables for c in cases}
    report = selection_report(
        cases, predicted_map, baseline=baseline,
        expanded=expanded_map if expansion is not None else None,
    )

    label = f"expansion {expansion}" if expansion else "no expansion"
    print(f"\nselector '{args.model}' vs lexical linker on {split_name} "
          f"({len(cases)} questions, {label})")
    print_block("overall", report["overall"])
    print_block("hard", report["hard"])
    print_block("easy", report["easy"])
    print("  per database:")
    for db_id, block in report["per_db"].items():
        print_block(f"    {db_id}", block)
    print(f"  empty predictions: {report['n_empty_predictions']}   request failures: {failures}")

    overall = report["overall"]["model"]
    base = report["overall"]["baseline"]
    metrics = {
        "all_gold_retained_rate": overall["all_gold_retained_rate"],
        "mean_precision": overall["mean_precision"],
        "mean_selected_tables": overall["mean_selected_tables"],
        "mean_table_recall": overall["mean_table_recall"],
        "hard_all_gold_retained_rate": report["hard"]["model"]["all_gold_retained_rate"],
        "hard_mean_precision": report["hard"]["model"]["mean_precision"],
        "baseline_all_gold_retained_rate": base["all_gold_retained_rate"],
        "baseline_mean_precision": base["mean_precision"],
        "baseline_mean_selected_tables": base["mean_selected_tables"],
        "n_empty_predictions": report["n_empty_predictions"],
        "n_request_failures": failures,
    }
    if expansion is not None:
        expanded_overall = report["overall"]["expanded"]
        metrics.update({
            "expanded_all_gold_retained_rate": expanded_overall["all_gold_retained_rate"],
            "expanded_mean_precision": expanded_overall["mean_precision"],
            "expanded_mean_selected_tables": expanded_overall["mean_selected_tables"],
            "expanded_hard_all_gold_retained_rate":
                report["hard"]["expanded"]["all_gold_retained_rate"],
        })
    summary = {
        "eval_path": str(args.eval),
        "eval_sha256": file_sha256(args.eval),
        "model": args.model,
        "base_url": args.base_url,
        "split": split_name,
        "n_samples": len(cases),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "thinking": args.thinking,
            "n_samples": args.n_samples,
        },
        "expansion": expansion,
        "metrics": metrics,
        "report": report,
    }
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"wrote {summary_path}")

    if not args.no_ledger:
        entry = append_run({
            "stage": "selector",
            "config_path": str(args.eval),
            "command": " ".join(sys.argv),
            "model": args.model,
            "checkpoint": args.model,
            "split": split_name,
            "n_samples": len(cases),
            "seed": args.seed,
            "decoding": {**summary["decoding"], "expansion": expansion},
            "metrics": metrics,
            "log_path": str(summary_path),
            "notes": args.notes,
        })
        print(f"ledger run_id {entry['run_id']} (git_dirty={entry['git_dirty']})")
    if failures:
        print(f"WARNING: {failures} requests failed after retries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
