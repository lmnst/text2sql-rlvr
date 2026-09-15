"""Answer BIRD questions with the execute-observe-revise agent loop.

    python scripts/run_agent.py --questions data/processed/val.json --split train \\
        --model Qwen3-1.7B --schema-mode linked --out results/preds/val788_agent.jsonl

Same client contract as generate.py: the GPU box runs ``vllm serve``, this
script runs locally, executes candidate SQL in the read-only sandbox and feeds
the result back to the model, up to ``--max-turns`` replies per question.

The output is a predictions file evaluate.py scores unchanged (its ``sql``
field is the final query); each record also carries the whole trajectory, so
the same file doubles as raw material for SFT or preference data.

    python scripts/evaluate.py --questions data/processed/val.json --split train \\
        --predictions results/preds/val788_agent.jsonl --stage ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import httpx

from text2sql_rlvr.agent import (
    DESCRIBE,
    EXECUTE,
    STOP_FINAL,
    AgentConfig,
    run_episode,
)
from text2sql_rlvr.data import (
    SCHEMA_MODES,
    SPLITS,
    discover_split,
    load_schema,
    render_selected_schema,
)
from text2sql_rlvr.data.selector import load_selected_tables
from text2sql_rlvr.ledger import file_sha256
from text2sql_rlvr.rewards.sandbox import OK, SqlExecutor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--split", choices=SPLITS, default="mini_dev")
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512, help="per reply")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking", action="store_true")

    parser.add_argument("--schema-style", choices=("ddl", "compact"), default="ddl")
    parser.add_argument("--schema-mode", choices=SCHEMA_MODES, default="full")
    parser.add_argument("--selected-tables", type=Path, default=None,
                        help="evaluate_selector.py output; tables not shown are listed by "
                             "name so the agent can DESCRIBE them")
    parser.add_argument("--selected-field", default="expanded_tables")
    parser.add_argument("--selected-fallback", choices=("full", "linked"), default="full")

    parser.add_argument("--max-turns", type=int, default=4, help="model replies per question")
    parser.add_argument("--max-rows-shown", type=int, default=5)
    parser.add_argument("--sample-rows", type=int, default=3, help="rows shown by DESCRIBE")
    parser.add_argument("--exec-timeout", type=float, default=10.0)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def load_done(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not record.get("error"):
                done.add(int(record["question_id"]))
    return done


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split = discover_split(args.root, args.split)
    if args.questions:
        split = replace(split, questions_path=args.questions)
    examples = split.load()
    if args.limit:
        examples = examples[: args.limit]
    already = load_done(args.out) if args.resume else set()
    todo = [e for e in examples if e.question_id not in already]
    print(f"{len(todo)} questions to run ({len(already)} already present)")

    selected = (
        load_selected_tables(args.selected_tables, args.selected_field)
        if args.selected_tables
        else None
    )
    config = AgentConfig(
        max_turns=args.max_turns,
        max_rows_shown=args.max_rows_shown,
        sample_rows=args.sample_rows,
    )

    schemas: dict[str, object] = {}
    schema_lock = threading.Lock()

    def cached_schema(db_id: str):
        with schema_lock:
            schema = schemas.get(db_id)
        if schema is None:
            schema = load_schema(split.db_path(db_id), db_id=db_id)
            with schema_lock:
                schemas[db_id] = schema
        return schema

    body_extra: dict[str, object] = {}
    if not args.thinking:
        body_extra["chat_template_kwargs"] = {"enable_thinking": False}
    client = httpx.Client(
        base_url=args.base_url.rstrip("/"),
        timeout=args.request_timeout,
        headers={"Authorization": f"Bearer {args.api_key}"},
    )

    def chat(messages: list[dict[str, str]]) -> str:
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            **body_extra,
        }
        last: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                response = client.post("/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # noqa: BLE001 - retried, then raised
                last = exc
                if attempt < args.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(str(last))

    executor = SqlExecutor(timeout_s=args.exec_timeout)

    def run(example):
        schema = cached_schema(example.db_id)
        mode, tables = args.schema_mode, None
        if selected is not None:
            tables = selected.get(example.question_id) or None
            if tables is None:
                mode = args.selected_fallback
        schema_text, selection = render_selected_schema(
            schema, example, mode=mode, style=args.schema_style, tables=tables
        )
        shown = {name.casefold() for name in selection.selected_tables}
        other = [t.name for t in schema.tables if t.name.casefold() not in shown]
        episode = run_episode(
            example, schema, schema_text, split.db_path(example.db_id), executor, chat,
            other_tables=other, config=config,
        )
        record = episode.as_dict()
        record["schema_mode"] = selection.mode
        record["selected_tables"] = list(selection.selected_tables)
        record["n_tables_total"] = selection.total_tables
        return record

    write_lock = threading.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    handle = args.out.open("a" if args.resume else "w", encoding="utf-8")
    records = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for done, record in enumerate(pool.map(run, todo), start=1):
                records.append(record)
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                if done % 50 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
    finally:
        handle.close()
        client.close()
        executor.close()

    n = len(records) or 1
    stops = Counter(r["stop_reason"] for r in records)
    first = Counter(r["first_exec_status"] for r in records)
    final = Counter(r["final_exec_status"] for r in records)
    recovered = sum(
        1 for r in records
        if r["first_exec_status"] not in (None, OK) and r["final_exec_status"] == OK
    )
    summary = {
        "n": len(records),
        "stop_reasons": dict(stops),
        "mean_turns": round(sum(r["n_turns"] for r in records) / n, 2),
        "mean_executions": round(sum(r["n_executions"] for r in records) / n, 2),
        "mean_describes": round(sum(r["n_describes"] for r in records) / n, 2),
        "first_exec_status": dict(first),
        "final_exec_status": dict(final),
        "n_recovered_from_first_error": recovered,
        "request_errors": sum(1 for r in records if r["error"]),
    }
    print("\nagent summary")
    for key, value in summary.items():
        print(f"  {key:<30} {value}")
    print(f"  ({EXECUTE}=ran a query, {DESCRIBE}=looked at a table, {STOP_FINAL}=committed)")

    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "split": split.name,
        "questions_path": str(split.questions_path),
        "n_requested": len(todo),
        "decoding": {
            "temperature": args.temperature, "top_p": args.top_p,
            "max_tokens": args.max_tokens, "seed": args.seed, "thinking": args.thinking,
        },
        "agent": {
            "max_turns": args.max_turns, "max_rows_shown": args.max_rows_shown,
            "sample_rows": args.sample_rows, "exec_timeout": args.exec_timeout,
        },
        "prompt_config": {"schema_style": args.schema_style, "instruction": "agent-v1"},
        "schema_selection": {
            "mode": "predicted" if selected is not None else args.schema_mode,
            "selected_tables_path": str(args.selected_tables) if selected else None,
            "selected_tables_sha256": file_sha256(args.selected_tables) if selected else None,
            "selected_field": args.selected_field if selected else None,
            "fallback_mode": args.selected_fallback if selected else None,
            "oracle_uses_gold_sql": args.schema_mode == "oracle" and selected is None,
        },
        "summary": summary,
        "executor_stats": executor.stats.as_dict(),
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"meta  {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
