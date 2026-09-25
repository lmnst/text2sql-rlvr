"""Audit archived baseline outcomes; no model calls and no dev database execution.

    python scripts/analyze_baseline_failures.py --config configs/analysis/base_val_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.eval.failure_analysis import (
    align_records,
    index_unique,
    stratified_sample,
    summarize,
    training_seed_catalog,
)
from text2sql_rlvr.ledger import append_run, file_sha256, read_runs
from text2sql_rlvr.rewards.compare import compare
from text2sql_rlvr.rewards.sandbox import SqlExecutor


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, data):
    Path(path).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    source = [r for r in read_runs() if r["run_id"] == cfg["source_run_id"]]
    if len(source) != 1:
        raise ValueError("source run must exist exactly once")
    source = source[0]
    questions = read_json(cfg["questions"])
    split = read_json(cfg["split_manifest"])
    original = read_json(cfg["source_questions"])
    ids = {q["question_id"] for q in questions}
    if ids != set(split["val_ids"]) or ids.intersection(split["train_ids"]):
        raise ValueError("questions must match the fixed val partition")
    for q in questions:
        src = original[q["question_id"]]
        if any(q.get(k, "") != src.get(k, "") for k in ("db_id", "question", "evidence", "SQL")):
            raise ValueError(f"original train content mismatch: {q['question_id']}")
    rows = align_records(
        questions, read_jsonl(cfg["outcomes"]), read_jsonl(cfg["predictions"]),
        id_mode=cfg["id_mode"],
    )
    summary = summarize(rows)
    if summary["n_samples"] != source["n_samples"] or any(
        summary[k] != source["metrics"][k] for k in ("official_ex", "strict_ex")
    ):
        raise ValueError("archived outcomes do not reproduce the source ledger")
    failed = [r for r in rows if not r["official"]]
    gaps = [r for r in rows if r["official"] and not r["strict"]]
    sample = stratified_sample(failed, cfg["failure_review_size"], cfg["seed"])
    sample += stratified_sample(gaps, cfg["gap_review_size"], cfg["seed"])
    review_path = Path(cfg["annotations"])
    annotations = read_json(review_path) if review_path.exists() else []
    by_id = index_unique(annotations, "question_id")
    selected_ids = {r["question_id"] for r in sample}
    if set(by_id) - selected_ids:
        raise ValueError("annotations contain ids outside the locked review sample")
    for a in annotations:
        if not a.get("labels") or not a.get("evidence") or a.get("reviewer") != "codex":
            raise ValueError("each annotation needs labels, evidence and explicit reviewer")
    for row in sample:
        row["review"] = by_id.get(row["question_id"])
    summary["by_database"] = {
        db: summarize([r for r in rows if r["db_id"] == db])
        for db in sorted({r["db_id"] for r in rows})
    }
    summary["audit"] = {
        "n_selected_failures": len([r for r in sample if not r["official"]]),
        "n_selected_gaps": len([r for r in sample if r["official"]]),
        "n_reviewed": len(annotations),
        "reviewed_ids": sorted(by_id),
        "sampling": "equal-coverage database x symptom, hash order, round robin",
        "reviewer": "codex; not independent human annotation",
        "prevalence_estimate": False,
        "label_counts_in_review_only": dict(sorted(Counter(
            label for a in annotations for label in set(a["labels"])
        ).items())),
    }
    replay = []
    if cfg["replay_extraction_changes"]:
        with SqlExecutor(timeout_s=cfg["replay_timeout_s"],
                         max_rows=cfg["replay_max_rows"]) as executor:
            for row in rows:
                if not row["extraction_changed"]:
                    continue
                db = Path(cfg["databases_dir"]) / row["db_id"] / f"{row['db_id']}.sqlite"
                gold = executor.execute(db, row["gold_sql"])
                old = executor.execute(db, row["pred_sql"])
                new = executor.execute(db, row["reextracted_sql"])
                old_verdict = compare(old, gold, gold_sql=row["gold_sql"])
                new_verdict = compare(new, gold, gold_sql=row["gold_sql"])
                replay.append({
                    "question_id": row["question_id"], "db_id": row["db_id"],
                    "old_sql": row["pred_sql"], "new_sql": row["reextracted_sql"],
                    "gold_sql": row["gold_sql"],
                    "old": asdict(old_verdict), "new": asdict(new_verdict),
                    "old_status": old.status, "new_status": new.status,
                    "old_error": old.error, "new_error": new.error,
                    "capped": old.truncated or new.truncated or gold.truncated,
                    "gold_status": gold.status,
                    "old_matches_archive": old_verdict.official == row["official"]
                    and old_verdict.strict == row["strict"],
                })
    uncapped = [r for r in replay if not r["capped"] and r["gold_status"] == "ok"]
    summary["extraction_replay"] = {
        "n_changed": len(replay), "n_uncapped_gold_ok": len(uncapped),
        "n_old_verdict_matches_archive": sum(r["old_matches_archive"] for r in replay),
        "uncapped_official_gains": sum(not r["old"]["official"] and r["new"]["official"]
                                       for r in uncapped),
        "uncapped_official_losses": sum(r["old"]["official"] and not r["new"]["official"]
                                        for r in uncapped),
        "uncapped_strict_gains": sum(not r["old"]["strict"] and r["new"]["strict"]
                                     for r in uncapped),
        "uncapped_strict_losses": sum(r["old"]["strict"] and not r["new"]["strict"]
                                      for r in uncapped),
        "note": "Paired replay of changed extraction only, not a new full baseline.",
    }
    train = read_json(cfg["train_questions"])
    for q in train:
        src = original[q["question_id"]]
        if any(q.get(k, "") != src.get(k, "") for k in ("db_id", "question", "evidence", "SQL")):
            raise ValueError(f"train source content mismatch: {q['question_id']}")
    seeds = training_seed_catalog(
        train, questions, set(split["train_ids"]),
        per_family=cfg["train_seed_per_family"], seed=cfg["seed"],
    )
    summary["train_seed_catalog"] = {
        "split": "train", "n_source_questions": len(train), "n_candidates": len(seeds),
        "per_family": dict(sorted(Counter(r["target_family"] for r in seeds).items())),
        "n_overlap_val_ids": len({r["question_id"] for r in seeds} & ids),
        "n_overlap_val_databases": len(
            {r["db_id"] for r in seeds} & {r["db_id"] for r in questions}
        ),
        "n_synthetic": 0, "n_approved_for_training": 0,
        "note": "Structural retrieval only; these are not measured train failures.",
    }
    inputs = {k: {"path": cfg[k], "sha256": file_sha256(cfg[k])} for k in (
        "questions", "source_questions", "split_manifest", "outcomes", "predictions",
        "annotations", "train_questions",
    )}
    inputs["analysis_code"] = {
        p: file_sha256(p) for p in (
            "scripts/analyze_baseline_failures.py",
            "src/text2sql_rlvr/eval/failure_analysis.py",
            "src/text2sql_rlvr/sql/validate.py",
            "src/text2sql_rlvr/rewards/sandbox.py",
            "src/text2sql_rlvr/rewards/compare.py",
        )
    }
    inputs["replay_databases"] = {
        db: file_sha256(Path(cfg["databases_dir"]) / db / f"{db}.sqlite")
        for db in sorted({r["db_id"] for r in replay})
    }
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    archive = out / "runs" / run_id
    archive.mkdir(parents=True)
    write_jsonl(out / "joined.jsonl", rows)
    write_jsonl(out / "review_packet.jsonl", sample)
    write_jsonl(out / "extraction_replay.jsonl", replay)
    write_jsonl(out / "train_seed_candidates.jsonl", seeds)
    run = append_run({
        "run_id": run_id,
        "stage": "baseline", "analysis_kind": "offline_failure_audit",
        "source_run_id": source["run_id"], "split": "val", "seed": cfg["seed"],
        "n_samples": len(rows), "model": source["model"], "checkpoint": source["checkpoint"],
        "decoding": source["decoding"], "prompt_config": source["prompt_config"],
        "config_path": str(args.config), "command": " ".join(sys.argv),
        "metrics": summary, "input_artifacts": inputs,
        "log_path": str(archive / "summary.json"),
        "notes": "Offline reaggregation and paired extraction replay, no new inference. "
                 "Logical split=val, source "
                 "ledger split=train denotes DB source. Dirty audit is internal only. "
                 "Review labels are Codex qualitative judgments, not population causes.",
    })
    write_json(out / "summary.json", {"run": run, "summary": summary})
    lines = [
        "# Base model 固定验证集错误审计：自动统计", "",
        f"分析 run_id：`{run['run_id']}`；来源 run_id：`{source['run_id']}`。",
        f"分析工作区 git_dirty：`{run['git_dirty']}`，本次为内部诊断，不用于对外成绩。",
        "本文件由脚本生成。没有重新推理或读取 Mini-Dev/dev；仅对提取结果变化的题目重放 SQL。",
        "official/strict 为历史存档分数，包含旧提取器与结果上限的影响，未覆盖或更正原台账。", "",
        "## 总体", "", "| 项目 | 数值 |", "|---|---:|",
    ]
    for k in ("n_samples", "official_correct", "strict_correct", "official_ex", "strict_ex",
              "official_failures", "failed_execution", "executed_but_official_wrong",
              "length_limited", "gold_empty", "official_pass_gold_empty",
              "extraction_changed", "strict_truncated"):
        lines.append(f"| {k} | {summary[k]} |")
    lines += ["", "## 全量失败现象（互斥，包含正确/口径分歧）", "",
              "百分比分母分别为全部验证题、官方判错题。口径分歧不计为官方错题。", "",
              "| 现象 | 题数 | 占全部题 | 占官方错题 |", "|---|---:|---:|---:|"]
    for name, n in summary["symptom_counts"].items():
        share = "—" if name in ("correct_both", "official_only") else (
            f"{100 * n / len(failed):.2f}%"
        )
        lines.append(f"| {name} | {n} | {100 * n / len(rows):.2f}% | {share} |")
    lines += ["", "## 按数据库", "", "| 数据库 | n | official | strict | 执行失败 | 可执行但错 |",
              "|---|---:|---:|---:|---:|---:|"]
    for db, s in summary["by_database"].items():
        lines.append(f"| {db} | {s['n_samples']} | {s['official_ex']}% | {s['strict_ex']}% | "
                     f"{s['failed_execution']} | {s['executed_but_official_wrong']} |")
    lines += ["", "## 不存在的函数（按报错原文）", "", "```json",
              json.dumps(summary["missing_functions"], ensure_ascii=False, indent=2), "```", "",
              "## 提取修复配对重放", "", "只重放提取变化的题，固定同一数据库和执行配置。",
              "这些变化不能称为模型提升，也没有生成新的全量 baseline 成绩。", "", "```json",
              json.dumps(summary["extraction_replay"], ensure_ascii=False, indent=2), "```", "",
              "official-only 的 strict 原因：", "", "```json",
              json.dumps(summary["official_only_reasons"], ensure_ascii=False, indent=2), "```", "",
              "## 抽样语义审阅", "",
              "按数据库×现象均衡覆盖，不能将审阅标签频数外推为全量根因比例。",
              "由 Codex 结合题目、evidence 和 SQL 审阅；不是人工双人标注，也不是逐题修复实验。",
              f"已审阅 {len(annotations)}/{len(sample)} 题。多标签可以重叠。", "",
              "| 审阅标签 | 样本内次数 |", "|---|---:|"]
    for label, n in summary["audit"]["label_counts_in_review_only"].items():
        lines.append(f"| {label} | {n} |")
    lines += ["", "## 待核验的训练种子", "",
              "仅按训练 gold SQL 的结构筛选；不是已观测的训练错题，也不是已生成的新数据。",
              "全部 approved_for_training=false。先核验题意和 SQL，再据此合成。", "",
              "| 方向 | 原始训练题号 | 数据库 |", "|---|---:|---|"]
    for row in seeds:
        lines.append(f"| {row['target_family']} | {row['question_id']} | {row['db_id']} |")
    for row in sample:
        a = row["review"]
        lines += ["", f"### 原始题号 {row['question_id']} / 旧序号 "
                  f"{row['historical_question_id']} / {row['db_id']} / {row['bucket']}", "",
                  row["question"], "", f"Evidence：{row['evidence']}", "",
                  "预测：", "```sql", row["pred_sql"], "```", "标准：", "```sql",
                  row["gold_sql"], "```", f"执行报错：{row['pred_error']}", "",
                  "原始输出（用来区分模型与提取器问题）：", "```text",
                  row["completion"].replace("```", "[fence]"), "```", "",
                  f"审阅：{a['evidence'] if a else '待审阅'}", "",
                  f"标签：{', '.join(a['labels']) if a else '待审阅'}"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Keep this run's evidence even when the convenient latest-view files change.
    for name in ("summary.json", "report.md", "joined.jsonl", "review_packet.jsonl",
                 "extraction_replay.jsonl", "train_seed_candidates.jsonl"):
        (archive / name).write_bytes((out / name).read_bytes())
    snapshot_paths = [str(args.config), cfg["annotations"], *inputs["analysis_code"]]
    for name in snapshot_paths:
        path = Path(name)
        if path.is_file():
            target = archive / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    print(json.dumps({"run_id": run["run_id"], "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
