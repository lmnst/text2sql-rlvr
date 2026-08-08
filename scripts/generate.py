"""Generate SQL for a BIRD split against any OpenAI-compatible endpoint.

Inference is deliberately a plain HTTP client rather than an in-process model:
this file stays runnable on a laptop, and the GPU box only has to run

    vllm serve Qwen/Qwen3-1.7B --port 8000

so the same script serves the base model, the SFT checkpoint and an RL
checkpoint without changing a line.

    python scripts/generate.py --root data/bird --split mini_dev \\
        --model Qwen/Qwen3-1.7B --out results/preds/base_minidev.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import httpx

from text2sql_rlvr.data import (
    SPLITS,
    PromptConfig,
    build_messages,
    discover_split,
    fetch_sample_rows,
    format_schema,
    load_schema,
)
from text2sql_rlvr.sql import extract_sql


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--questions", type=Path, default=None,
                        help="score a custom BIRD-format question file (e.g. the val "
                             "split built by build_splits.py) using --split's databases")
    parser.add_argument("--split", choices=SPLITS, default="mini_dev")
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable Qwen3 thinking mode (off by default; see AGENTS.md)",
    )

    parser.add_argument("--schema-style", choices=("ddl", "compact"), default="ddl")
    parser.add_argument("--descriptions", action="store_true", help="include column descriptions")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--instruction-version", choices=("v1", "v2"), default="v2",
                        help="v1 reproduces the first baseline; v2 adds the dialect "
                             "and pseudo-function rules")

    parser.add_argument("--limit", type=int, default=0, help="first N examples, 0 for all")
    parser.add_argument("--resume", action="store_true", help="skip ids already in --out")
    return parser.parse_args(argv)


def load_done(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["question_id"])
    return done


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = PromptConfig(
        schema_style=args.schema_style,
        include_descriptions=args.descriptions,
        include_evidence=not args.no_evidence,
        sample_rows=args.sample_rows,
        instruction_version=args.instruction_version,
    )

    split = discover_split(args.root, args.split)
    if args.questions:
        split = replace(split, questions_path=args.questions)
        print(f"questions overridden: {args.questions}")
    examples = split.load()
    if args.limit:
        examples = examples[: args.limit]

    already = load_done(args.out) if args.resume else set()
    todo = [e for e in examples if e.question_id not in already]
    print(f"{len(todo)} to generate ({len(already)} already present)")

    schema_cache: dict[str, str] = {}
    schema_lock = threading.Lock()

    def schema_text(db_id: str) -> str:
        with schema_lock:
            cached = schema_cache.get(db_id)
        if cached is not None:
            return cached
        db_path = split.db_path(db_id)
        schema = load_schema(db_path, db_id=db_id)
        samples = (
            fetch_sample_rows(db_path, schema, config.sample_rows)
            if config.sample_rows
            else None
        )
        text = format_schema(
            schema,
            style=config.schema_style,
            include_descriptions=config.include_descriptions,
            sample_rows=samples,
        )
        with schema_lock:
            schema_cache[db_id] = text
        return text

    body_extra: dict[str, object] = {}
    if not args.thinking:
        # vLLM forwards this to the chat template; Qwen3 uses it to skip the
        # <think> block, which otherwise dominates the sequence length.
        body_extra["chat_template_kwargs"] = {"enable_thinking": False}

    client = httpx.Client(
        base_url=args.base_url.rstrip("/"),
        timeout=args.request_timeout,
        headers={"Authorization": f"Bearer {args.api_key}"},
    )
    write_lock = threading.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    handle = args.out.open("a" if args.resume else "w", encoding="utf-8")

    def generate(example):
        messages = build_messages(example, schema_text(example.db_id), config)
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            **body_extra,
        }
        started = time.monotonic()
        last_error = ""
        for attempt in range(args.retries + 1):
            try:
                response = client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                return {
                    "question_id": example.question_id,
                    "db_id": example.db_id,
                    "completion": choice["message"]["content"] or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage", {}),
                    "latency_s": round(time.monotonic() - started, 3),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001 - retried, then recorded
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(2**attempt)
        return {
            "question_id": example.question_id,
            "db_id": example.db_id,
            "completion": "",
            "finish_reason": None,
            "usage": {},
            "latency_s": round(time.monotonic() - started, 3),
            "error": last_error,
        }

    failures = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for done, record in enumerate(pool.map(generate, todo), start=1):
                record["sql"] = extract_sql(record["completion"])
                if record["error"]:
                    failures += 1
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                if done % 50 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
    finally:
        handle.close()
        client.close()

    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "split": split.name,
        "questions_path": str(split.questions_path),
        "n_requested": len(todo),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "thinking": args.thinking,
        },
        "prompt_config": config.as_dict(),
        "request_failures": failures,
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {args.out}")
    print(f"meta  {meta_path}")
    if failures:
        print(f"WARNING: {failures} requests failed after retries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
