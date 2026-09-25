"""Freeze train prompts, collect paired predictions and compare residual failures.

No synthesis or training. Run --help for prepare, fingerprint, generate and compare.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
import httpx

from text2sql_rlvr.data import (
    PromptConfig,
    build_messages,
    load_examples,
    load_schema,
    oracle_table_names,
    render_selected_schema,
)
from text2sql_rlvr.eval.failure_analysis import stratified_sample, symptom
from text2sql_rlvr.eval.train_diagnosis import (
    choose_subset,
    json_hash,
    paired_summary,
    transition,
    verify_predictions,
)
from text2sql_rlvr.ledger import append_run, file_sha256
from text2sql_rlvr.rewards.compare import compare
from text2sql_rlvr.rewards.sandbox import SqlExecutor
from text2sql_rlvr.sql import extract_sql


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")


def code_hashes():
    return {str(p): file_sha256(p) for p in sorted(Path("src/text2sql_rlvr").rglob("*.py"))} | {
        "scripts/train_failure_diagnosis.py": file_sha256("scripts/train_failure_diagnosis.py")
    }


def asset_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(cfg, config_path):
    out = Path(cfg["output_dir"])
    if (out / "manifest.json").exists():
        raise ValueError("subset already frozen; use existing manifest, do not resample")
    train = read(cfg["train_questions"])
    original = read(cfg["original_questions"])
    split = read(cfg["split_manifest"])
    selected = choose_subset(train, split, cfg["n_samples"], cfg["seed"])
    for row in selected:
        source = original[row["question_id"]]
        keys = ("db_id", "question", "evidence", "SQL")
        if any(row.get(k, "") != source.get(k, "") for k in keys):
            raise ValueError("selected content is not identical to original train")
    out.mkdir(parents=True, exist_ok=True)
    write(out / "questions.json", selected)
    schemas, database_hashes, prompts, schema_audit = {}, {}, [], []
    config = PromptConfig(**cfg["prompt_config"])
    for example in load_examples(out / "questions.json"):
        db = example.db_id
        if db not in schemas:
            path = Path(cfg["database_root"]) / db / f"{db}.sqlite"
            schemas[db] = load_schema(path, db_id=db)
            database_hashes[db] = asset_hash(path)
        text, selection = render_selected_schema(
            schemas[db], example, mode=cfg["schema_mode"], style=config.schema_style,
            include_descriptions=False, sample_rows=None, max_chars=0,
        )
        messages = build_messages(example, text, config)
        prompts.append({
            "question_id": example.question_id, "db_id": db,
            "messages": messages, "prompt_sha256": json_hash(messages),
        })
        # Only the audit reads gold table names; it cannot feed back into prompts.
        required = oracle_table_names(schemas[db], example.gold_sql)
        schema_audit.append({
            "question_id": example.question_id, "db_id": db,
            "selected_tables": list(selection.selected_tables), "gold_tables": list(required),
            "missing_gold_tables": sorted(set(required) - set(selection.selected_tables)),
            "status": "lexical gold-table diagnostic, not causal attribution",
        })
    write_lines(out / "prompts.jsonl", prompts)
    write_lines(out / "schema_audit.jsonl", schema_audit)
    manifest = {
        "status": "prepared_no_inference", "config": cfg,
        "config_sha256": file_sha256(config_path), "code_sha256": code_hashes(),
        "source_sha256": {k: file_sha256(cfg[k]) for k in (
            "train_questions", "original_questions", "split_manifest")},
        "questions_sha256": file_sha256(out / "questions.json"),
        "prompts_sha256": file_sha256(out / "prompts.jsonl"),
        "database_sha256": database_hashes,
        "selected_ids": [r["question_id"] for r in selected],
        "n_samples": len(selected),
        "database_counts": dict(sorted(Counter(r["db_id"] for r in selected).items())),
        "synthetic_data": False,
        "exposure": "Subset of nominal SFT training partition; not held-out generalization.",
    }
    write(out / "manifest.json", manifest)
    entry = append_run({
        "stage": "ablation", "analysis_kind": "train_diagnosis_preparation",
        "model": "not_run", "checkpoint": "not_run", "split": "train",
        "n_samples": len(selected), "seed": cfg["seed"], "decoding": cfg["decoding"],
        "config_path": str(config_path), "command": " ".join(sys.argv),
        "metrics": {"n_selected": len(selected), "n_databases": len(schemas),
                    "n_prompts_missing_gold_table": sum(bool(r["missing_gold_tables"])
                                                        for r in schema_audit)},
        "log_path": str(out / "manifest.json"),
        "notes": "Subset preparation only. No model scores, no synthetic data, no val/dev reads.",
    })
    print(f"Prepared {len(selected)} original train questions; run_id={entry['run_id']}")


def frozen(cfg, config_path):
    out = Path(cfg["output_dir"])
    manifest = read(out / "manifest.json")
    if manifest["config_sha256"] != file_sha256(config_path):
        raise ValueError("frozen experiment config changed")
    if manifest["code_sha256"] != code_hashes():
        raise ValueError("diagnosis code changed after preparation; freeze a new experiment")
    for name in ("questions", "prompts"):
        suffix = ".json" if name == "questions" else ".jsonl"
        if file_sha256(out / (name + suffix)) != manifest[name + "_sha256"]:
            raise ValueError(f"frozen {name} changed")
    return out, manifest


def fingerprint(args):
    checkpoint, tokenizer = Path(args.checkpoint).resolve(), Path(args.tokenizer).resolve()
    weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights or not (checkpoint / "config.json").is_file():
        raise ValueError("expected a merged Hugging Face checkpoint with safetensors")
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                       "special_tokens_map.json", "vocab.json", "merges.txt")
    tokenizer_files = [p for name in tokenizer_names
                       if (p := tokenizer / name).is_file()]
    if not tokenizer_files:
        raise ValueError("tokenizer assets missing")
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "vllm", "transformers")}
    hardware = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    write(args.out, {
        "checkpoint": str(checkpoint), "tokenizer": str(tokenizer),
        "checkpoint_sha256": {p.name: asset_hash(p)
                              for p in weights + [checkpoint / "config.json"]},
        "tokenizer_sha256": {p.name: file_sha256(p) for p in tokenizer_files},
        "runtime": versions, "hardware": hardware, "platform": platform.platform(),
        "served_model": args.served_model,
        "note": "Fingerprint the local files used by the serving process; verify its launch path.",
    })


def generate(cfg, args):
    out, manifest = frozen(cfg, args.config)
    target = out / f"{args.arm}_predictions.jsonl"
    if target.exists():
        raise ValueError("prediction file already exists; preserve it rather than overwrite")
    provenance = read(args.provenance)
    expected = cfg["arms"][args.arm]
    if provenance["checkpoint"] != expected["checkpoint"]:
        raise ValueError("checkpoint fingerprint does not match frozen target")
    if provenance["served_model"] != expected["served_model"]:
        raise ValueError("served model mismatch")
    prompts = read_lines(out / "prompts.jsonl")
    decoding = {k: v for k, v in cfg["decoding"].items() if k != "thinking"}
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=300) as client:
        models = client.get("/models")
        models.raise_for_status()
        if expected["served_model"] not in {m["id"] for m in models.json()["data"]}:
            raise ValueError("requested model not listed by server")

        def query(row):
            response = client.post("/chat/completions", json={
                "model": expected["served_model"], "messages": row["messages"], **decoding,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            return {
                "question_id": row["question_id"], "db_id": row["db_id"],
                "prompt_sha256": row["prompt_sha256"], "model": expected["served_model"],
                "completion": choice["message"]["content"] or "",
                "finish_reason": choice["finish_reason"], "usage": result.get("usage", {}),
                "error": None,
            }

        # A failed HTTP request stops the run. It must not become a model SQL failure.
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = []
            for row in pool.map(query, prompts):
                rows.append(row)
                if len(rows) % 32 == 0:
                    print(f"{args.arm}: {len(rows)}/{len(prompts)}", flush=True)
    write_lines(target, rows)
    write(out / f"{args.arm}_provenance.json", {
        **provenance, "prompts_sha256": manifest["prompts_sha256"],
        "config_sha256": manifest["config_sha256"], "decoding": cfg["decoding"],
        "predictions_sha256": file_sha256(target), "server_models_response": models.json(),
    })
    print(f"Saved {args.arm} predictions; no evaluation performed yet")


def evaluate_pair(question, base, sft, executor, db_root):
    db = Path(db_root) / question["db_id"] / f"{question['db_id']}.sqlite"
    gold = executor.execute(db, question["SQL"])
    pair = {"question_id": question["question_id"], "db_id": question["db_id"],
            "question": question["question"], "evidence": question.get("evidence", ""),
            "gold_sql": question["SQL"]}
    for arm, prediction in (("base", base), ("sft", sft)):
        sql = extract_sql(prediction["completion"])
        result = executor.execute(db, sql)
        verdict = compare(result, gold, gold_sql=question["SQL"])
        comparable = gold.ok and not (gold.truncated or result.truncated)
        comparable = comparable and result.status != "timeout"
        row = {
            **asdict(verdict), "pred_sql": sql, "completion": prediction["completion"],
            "pred_status": result.status, "gold_status": gold.status, "pred_error": result.error,
            "pred_truncated": result.truncated, "gold_truncated": gold.truncated,
            "finish_reason": prediction["finish_reason"], "comparable": comparable,
        }
        row["bucket"] = symptom(row) if comparable else "unscorable"
        pair[arm] = row
    pair["transition"] = transition(pair["base"], pair["sft"])
    return pair


def compare_arms(cfg, args):
    out, manifest = frozen(cfg, args.config)
    prompts, questions = read_lines(out / "prompts.jsonl"), read(out / "questions.json")
    predictions, provenance = {}, {}
    for arm in ("base", "sft"):
        path = out / f"{arm}_predictions.jsonl"
        predictions[arm] = verify_predictions(read_lines(path), prompts, arm)
        p = read(out / f"{arm}_provenance.json")
        if p["checkpoint"] != cfg["arms"][arm]["checkpoint"]:
            raise ValueError("wrong checkpoint in paired predictions")
        if p["served_model"] != cfg["arms"][arm]["served_model"]:
            raise ValueError("wrong served model in paired predictions")
        if not p["checkpoint_sha256"]:
            raise ValueError("checkpoint fingerprint missing")
        if p["predictions_sha256"] != file_sha256(path):
            raise ValueError("prediction artifact hash mismatch")
        if p["config_sha256"] != manifest["config_sha256"] or p["decoding"] != cfg["decoding"]:
            raise ValueError("generation configuration mismatch")
        provenance[arm] = p
    for field in ("tokenizer_sha256", "runtime", "prompts_sha256"):
        if provenance["base"][field] != provenance["sft"][field]:
            raise ValueError(f"paired inference differs in {field}")
    for db, checksum in manifest["database_sha256"].items():
        if asset_hash(Path(cfg["database_root"]) / db / f"{db}.sqlite") != checksum:
            raise ValueError("database changed since subset preparation")
    with SqlExecutor(**cfg["execution"]) as executor, ThreadPoolExecutor(max_workers=8) as pool:
        pairs = list(pool.map(lambda q: evaluate_pair(
            q, predictions["base"][q["question_id"]], predictions["sft"][q["question_id"]],
            executor, cfg["database_root"],
        ), questions))
    summary = paired_summary(pairs)
    summary["by_database"] = {
        db: paired_summary([r for r in pairs if r["db_id"] == db])
        for db in sorted({r["db_id"] for r in pairs})
    }
    summary["length_limited"] = {
        arm: sum(r[arm]["finish_reason"] == "length" for r in pairs) for arm in ("base", "sft")
    }
    review = []
    for category in ("persistent_failure", "regression", "resolved", "both_correct"):
        candidates = [{**r, "bucket": r["sft"]["bucket"]} for r in pairs
                      if r["transition"] == category]
        review.extend(stratified_sample(candidates, cfg["review_per_transition"], cfg["seed"]))
    run_id = uuid.uuid4().hex[:12]
    archive = out / "runs" / run_id
    archive.mkdir(parents=True)
    write_lines(archive / "pairs.jsonl", pairs)
    write_lines(archive / "review_packet.jsonl", review)
    entry = append_run({
        "run_id": run_id, "stage": "ablation", "analysis_kind": "train_base_sft_paired_diagnosis",
        "model": "Qwen3-1.7B Base vs frozen strong SFT", "checkpoint": cfg["arms"],
        "split": "train", "n_samples": len(pairs), "seed": cfg["seed"],
        "decoding": cfg["decoding"], "config_path": str(args.config),
        "command": " ".join(sys.argv), "metrics": summary,
        "model_provenance": provenance, "code_sha256": code_hashes(),
        "subset_manifest_sha256": file_sha256(out / "manifest.json"),
        "log_path": str(archive / "summary.json"),
        "notes": "Train-side diagnosis, not generalization. No synthesis. Pairwise shared "
                 "complete-result denominator; semantic review still required.",
    })
    write(archive / "summary.json", {"run": entry, "summary": summary})
    write(out / "latest_comparison.json", {"run_id": run_id, "archive": str(archive)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fingerprint", "generate", "compare"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/analysis/train_base_vs_sft.json"))
    parser.add_argument("--arm", choices=("base", "sft"))
    parser.add_argument("--base-url")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument("--served-model")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "fingerprint":
        if not all((args.checkpoint, args.tokenizer, args.served_model, args.out)):
            parser.error("fingerprint needs --checkpoint, --tokenizer, --served-model, --out")
        fingerprint(args)
    else:
        cfg = read(args.config)
        if args.action == "prepare":
            prepare(cfg, args.config)
        elif args.action == "generate":
            if not all((args.arm, args.base_url, args.provenance)):
                parser.error("generate needs --arm, --base-url, --provenance")
            generate(cfg, args)
        else:
            compare_arms(cfg, args)


if __name__ == "__main__":
    main()
