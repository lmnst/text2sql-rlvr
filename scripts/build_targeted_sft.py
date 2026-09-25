"""Construct and audit a hand-authored train-only batch without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import PromptConfig, format_schema, load_schema
from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.sft import build_sft_record
from text2sql_rlvr.eval.failure_analysis import index_unique
from text2sql_rlvr.ledger import append_run, file_sha256
from text2sql_rlvr.rewards.compare import compare
from text2sql_rlvr.rewards.sandbox import ExecResult, SqlExecutor
from text2sql_rlvr.sql import extract_sql


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_rows(rows, spec):
    """Small Python reference computations over raw relational witnesses.

    These independently recompute projection, multiplicity and aggregation, but
    witness JOINs still require semantic/schema review. This is not a SQL prover.
    """
    for col, op, value in spec.get("filter", []):
        if op == "eq":
            rows = [r for r in rows if r[col] == value]
        elif op == "gt":
            rows = [r for r in rows if r[col] is not None and r[col] > value]
        else:
            raise ValueError(op)
    kind = spec["kind"]
    if kind == "project":
        def value(row, col):
            if isinstance(col, int):
                return row[col]
            parts = [row[i] for i in col["concat"]]
            return None if None in parts else col["separator"].join(str(x) for x in parts)
        result = [tuple(value(r, col) for col in spec["columns"]) for r in rows]
        return list(dict.fromkeys(result)) if spec.get("distinct") else result
    if kind == "count":
        return [(len(rows),)]
    if kind == "group_counts":
        groups = Counter(r[spec["key"]] for r in rows)
        groups = {k: v for k, v in groups.items() if v > spec["threshold"]}
        mode = spec["mode"]
        if mode == "keys":
            return [(k,) for k in groups]
        if mode == "key_sizes":
            return list(groups.items())
        if mode == "sizes":
            return [(n,) for n in groups.values()]
        if mode == "count_groups":
            return [(len(groups),)]
        if mode == "sum_sizes":
            return [(sum(groups.values()),)]
        raise ValueError(mode)
    if kind in ("mean", "group_sums_count"):
        groups = defaultdict(list)
        for r in rows:
            if r[spec["value"]] is not None:
                groups[r[spec["key"]]].append(r[spec["value"]])
        if kind == "group_sums_count":
            return [(sum(sum(g) > spec["threshold"] for g in groups.values()),)]
        values = ([sum(g) / len(g) for g in groups.values()]
                  if spec["mode"] == "group_means" else
                  [r[spec["value"]] for r in rows if r[spec["value"]] is not None])
        return [(sum(values) / len(values) if values else None,)]
    if kind == "two_counts":
        counts = [sum(r[spec["column"]] == v for r in rows) for v in spec["values"]]
        if spec["mode"] == "difference":
            return [(counts[0] - counts[1],)]
        return [tuple(counts)]
    raise ValueError(kind)


def result_evidence(result):
    encoded = json.dumps(result.rows, ensure_ascii=False, separators=(",", ":"))
    return {"status": result.status, "error": result.error, "n_rows": result.n_rows,
            "n_columns": len(result.columns), "columns": result.columns,
            "truncated": result.truncated, "elapsed_s": result.elapsed_s,
            "rows_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "preview": result.rows[:5], "contains_null": any(None in r for r in result.rows)}


def verify_pair(pair, executor, db):
    witness = executor.execute(db, pair["witness_sql"])
    checks = {"pair_id": pair["pair_id"], "family": pair["family"], "db_id": pair["db_id"],
              "witness": result_evidence(witness), "variants": [], "failures": []}
    if not witness.ok or witness.truncated or not witness.rows:
        checks["failures"].append("witness_unavailable_or_empty_or_capped")
        return checks
    results = []
    for variant in pair["variants"]:
        actual = executor.execute(db, variant["sql"])
        expected = reference_rows(witness.rows, variant["reference"])
        width = len(expected[0]) if expected else len(actual.columns)
        ref = ExecResult("ok", rows=tuple(expected), columns=tuple(str(i) for i in range(width)))
        matches = actual.ok and not actual.truncated and compare(actual, ref).strict
        if not matches or not actual.rows:
            checks["failures"].append(f"{variant['variant']}:reference_or_execution_or_empty")
        checks["variants"].append({"variant": variant["variant"],
                                   "result": result_evidence(actual),
                                   "python_reference_match": matches})
        results.append(actual)
    distinct = all(r.ok and not r.truncated for r in results) and not compare(*results).strict
    checks["counterpart_sql_rejected_by_strict_result"] = distinct
    checks["counterpart_sql_rejected_by_set_result"] = (
        all(r.ok for r in results) and not compare(*results).official
    )
    if not distinct:
        checks["failures"].append("pair_answers_not_distinguishable")
    checks["accepted"] = not checks["failures"]
    return checks


def validate_sources(cfg, questions, split):
    by_id = index_unique(questions, "question_id")
    if set(by_id) != set(split["train_ids"]):
        raise ValueError("Source must be the complete frozen filtered train split")
    index_unique(cfg["pairs"], "pair_id")
    for pair in cfg["pairs"]:
        if pair["db_id"] in split["val_db_ids"]:
            raise ValueError("Validation database used in construction")
        if pair["reviewer"] != "codex" or not pair["focus"]:
            raise ValueError("Missing semantic review")
        for qid in pair["source_question_ids"]:
            if qid in split["val_ids"] or qid not in by_id:
                raise ValueError(f"Source outside train: {qid}")
            if by_id[qid]["db_id"] != pair["db_id"]:
                raise ValueError("Source/database mismatch")
    return by_id


def review_document(cfg, checks, run_id):
    text = ["# Targeted SFT v1：逐对审核", "", f"运行：`{run_id}`。划分：train。", "",
            "每对两题均由 Codex 构造并审阅，未经独立人工复核。这里只展示结果前五行；",
            "完整执行的行数、列数与哈希在 audit.json，来源原题及 SQL 在 source_records.json。",
            "Python 核对独立计算投影、重复行和聚合，但共享原始关系查询，不是形式化正确性证明。",
            "full schema 提示词不含审核文字、参考计算或错误 SQL。没有模型推理或训练。", ""]
    for pair, check in zip(cfg["pairs"], checks, strict=True):
        text += [f"## {pair['pair_id']} · {pair['family']} · {pair['db_id']}", "",
                 f"来源题号：{pair['source_question_ids']}。", "", pair["focus"], "",
                 f"审核状态：{pair['review_status']}。",
                 f"执行审核通过：{check.get('accepted', False)}。",
                 f"原始核对记录数：{check['witness']['n_rows']}。", ""]
        for variant, result in zip(pair["variants"], check["variants"], strict=True):
            text += [f"### {variant['variant'].upper()}", "", variant["question"], ""]
            if pair["evidence"]:
                text += ["Evidence: " + pair["evidence"], ""]
            r = result["result"]
            text += ["```sql", variant["sql"], "```", "",
                     f"执行：{r['n_rows']} 行 × {r['n_columns']} 列；"
                     f"Python 核对：{result['python_reference_match']}。", "",
                     "结果预览：", "```json", json.dumps(r["preview"], ensure_ascii=False),
                     "```", ""]
        text += ["互换两题答案后，保留重复行的结果比较能拒绝："
                 + str(check.get("counterpart_sql_rejected_by_strict_result")), "",
                 "忽略重复行的集合比较也能拒绝："
                 + str(check.get("counterpart_sql_rejected_by_set_result")), ""]
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_construction/targeted_v1.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    cfg = read(args.config)
    questions = read(cfg["source_questions"])
    split = read(cfg["split_manifest"])
    by_id = validate_sources(cfg, questions, split)
    quarantine = read(cfg["quarantine_path"])
    selected_sources = {i for p in cfg["pairs"] for i in p["source_question_ids"]}
    if any(q["source_question_id"] in selected_sources for q in quarantine):
        raise ValueError("Quarantined source used in batch")
    run_id = uuid.uuid4().hex[:12]
    output = Path(cfg["output_root"]) / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    db_paths = {p["db_id"]: Path(cfg["database_root"]) / p["db_id"] / (p["db_id"] + ".sqlite")
                for p in cfg["pairs"]}
    db_hashes = {db: digest_file(path) for db, path in db_paths.items()}
    checks = []
    with SqlExecutor(timeout_s=cfg["timeout_s"], max_rows=cfg["max_rows"]) as executor:
        for pair in cfg["pairs"]:
            check = verify_pair(pair, executor, db_paths[pair["db_id"]])
            checks.append(check)
            print(pair["pair_id"], check["failures"] or "passed", flush=True)
    # No partial SFT export: fix or replace failing pairs, then build the whole batch.
    all_pass = all(c.get("accepted", False) for c in checks)
    exported = all_pass and not args.audit_only
    records, examples = [], []
    if exported:
        prompt_cfg = PromptConfig()
        schemas = {db: format_schema(load_schema(path, db_id=db), style="ddl")
                   for db, path in db_paths.items()}
        for pair in cfg["pairs"]:
            if pair["review_status"] != "codex_semantic_reviewed":
                raise ValueError("Semantic review must precede SFT export")
            for variant in pair["variants"]:
                targeted_id = f"{cfg['version']}_{pair['pair_id']}_{variant['variant']}"
                example = {"targeted_id": targeted_id,
                           "question_id": -100001 - len(examples), "db_id": pair["db_id"],
                           "pair_id": pair["pair_id"], "target_family": pair["family"],
                           "question": variant["question"], "evidence": pair["evidence"],
                           "SQL": variant["sql"],
                           "source_question_ids": pair["source_question_ids"],
                           "split": "train", "synthetic": True}
                record = build_sft_record(BirdExample(
                    example["question_id"], example["db_id"], example["question"],
                    evidence=example["evidence"], gold_sql=example["SQL"],
                ), schemas[pair["db_id"]], prompt_cfg).as_dict()
                if extract_sql(record["messages"][-1]["content"]) != variant["sql"].rstrip(";"):
                    raise ValueError("Target SQL changed during formatting")
                examples.append(example)
                records.append(record)
        if len({e["question"] for e in examples}) != len(examples):
            raise ValueError("Duplicate constructed questions")
        write(output / "examples.json", examples)
        (output / "sft.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
    write(output / "audit.json", checks)
    if all_pass:
        (output / "review.md").write_text(review_document(cfg, checks, run_id), encoding="utf-8")
    write(output / "quarantined_sources.json", quarantine)
    write(output / "construction_spec.json", cfg)
    source_ids = sorted({i for p in cfg["pairs"] for i in p["source_question_ids"]})
    write(output / "source_records.json", [by_id[i] for i in source_ids])
    metrics = {"candidate_examples": 2 * len(checks), "passed_pairs": sum(
        bool(c.get("accepted")) for c in checks), "exported_examples": len(records),
        "examples_by_family": dict(Counter(e["target_family"] for e in examples)),
        "examples_by_database": dict(Counter(e["db_id"] for e in examples)),
        "n_source_questions": len(source_ids), "n_quarantined_sources": len(quarantine),
        "strict_contrast_pairs": sum(bool(c.get("counterpart_sql_rejected_by_strict_result"))
                                     for c in checks),
        "set_contrast_pairs": sum(bool(c.get("counterpart_sql_rejected_by_set_result"))
                                  for c in checks)}
    manifest = {"run_id": run_id, "status": "audited_export" if exported else "audit_only",
                "split": "train", "seed": cfg["seed"], "metrics": metrics,
                "source_ids": source_ids, "database_sha256": db_hashes,
                "source_questions_sha256": file_sha256(cfg["source_questions"]),
                "split_manifest_sha256": file_sha256(cfg["split_manifest"]),
                "config_sha256": file_sha256(args.config), "script_sha256": file_sha256(__file__),
                "quarantine_sha256": file_sha256(cfg["quarantine_path"]),
                "prompt_config": PromptConfig().as_dict(), "schema_mode": "full",
                "thinking": False, "negative_id_namespace": "-100001 through -100060",
                "model_calls": 0, "training_runs": 0,
                "max_example_chars": max((sum(len(m["content"]) for m in r["messages"])
                                          for r in records), default=0),
                "comparison": "strict multiset, same column count, NULL preserved, "
                              "floats canonicalized to 6 significant digits; row order ignored",
                "review": "Codex-authored and Codex-reviewed; not independent human gold",
                "timeout": "SQLite progress-handler timeout; no process hard timeout",
                "limitations": ["No model or training efficacy measurement",
                                "Python reference shares witness relations; not formal equivalence",
                                "Pair contrast verified only on the current fixed database",
                                "Token lengths not measured with the model tokenizer",
                                "Keep paired variants and source families in the same train split"]}
    manifest["outputs_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
    write(output / "manifest.json", manifest)
    append_run({"run_id": run_id, "stage": "ablation",
                "analysis_kind": "targeted_data_construction",
                "split": "train", "n_samples": len(checks) * 2, "seed": cfg["seed"],
                "config_path": args.config, "command": " ".join(sys.argv),
                "model": "not_run", "checkpoint": "not_used", "decoding": {"inference": "not_run"},
                "metrics": metrics, "log_path": str(output / "manifest.json"),
                "notes": "Dataset audit only, not model accuracy. Codex semantic review plus "
                         "read-only execution, Python reference and counterpart contrast checks."})
    print(json.dumps({"output": str(output), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
