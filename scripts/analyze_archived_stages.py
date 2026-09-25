"""Analyze saved val outputs only: no training, inference, or SQL execution."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.eval.failure_analysis import align_records, index_unique, summarize
from text2sql_rlvr.ledger import append_run, file_sha256, read_runs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    config_path = Path("configs/analysis/archived_val_stages.json")
    cfg = read(config_path)
    questions = read(cfg["questions"])
    split = read(cfg["split_manifest"])
    ids = {q["question_id"] for q in questions}
    if ids != set(split["val_ids"]) or ids.intersection(split["train_ids"]):
        raise ValueError("Expected the exact fixed val partition")
    ledger = index_unique(read_runs(), "run_id")
    stages, summaries, sources = {}, {}, {}
    hashes = {p: file_sha256(p) for p in (cfg["questions"], cfg["split_manifest"], __file__)}
    for name, arm in cfg["stages"].items():
        pred = f"results/preds/{arm['stem']}.jsonl"
        outcome = f"results/outcomes/{arm['stem']}.jsonl"
        rows = align_records(questions, lines(outcome), lines(pred), id_mode=arm["id_mode"])
        summary = summarize(rows)
        source = ledger[arm["run_id"]]
        if len(rows) != source["n_samples"] or any(
            summary[k] != source["metrics"][k] for k in ("official_ex", "strict_ex")
        ):
            raise ValueError(f"{name}: saved outputs do not reproduce ledger")
        stages[name] = index_unique(rows, "question_id")
        summaries[name] = summary
        sources[name] = source
        hashes.update({p: file_sha256(p) for p in (pred, outcome)})
    pairs = []
    for qid in sorted(ids):
        arms = {name: rows[qid] for name, rows in stages.items()}
        base, sft = arms["base"], arms["early_sft"]
        transition = {
            (False, False): "persistent_failure", (False, True): "resolved",
            (True, False): "regression", (True, True): "both_correct",
        }[(bool(base["official"]), bool(sft["official"]))]
        pairs.append({"question_id": qid, "db_id": base["db_id"],
                      "base_early_sft_transition": transition, "arms": arms})
    paired = dict(Counter(p["base_early_sft_transition"] for p in pairs))
    persistent = [p for p in pairs if p["base_early_sft_transition"] == "persistent_failure"]
    persistent_symptoms = dict(Counter(p["arms"]["early_sft"]["bucket"] for p in persistent))
    failure_transitions = dict(Counter(
        ("executable" if p["arms"]["base"]["pred_status"] == "ok" else "execution_failed")
        + " -> "
        + ("executable" if p["arms"]["early_sft"]["pred_status"] == "ok"
           else "execution_failed")
        for p in persistent
    ))
    annotations = read(cfg["annotations"])
    annotated = index_unique(annotations, "question_id")
    if not set(annotated) <= ids or any(
        a.get("reviewer") != "codex" or not a.get("labels") or not a.get("evidence")
        for a in annotations
    ):
        raise ValueError("Invalid review annotation")
    hashes[cfg["annotations"]] = file_sha256(cfg["annotations"])
    review = [{**p, "review": annotated[p["question_id"]]}
              for p in pairs if p["question_id"] in annotated]
    by_database = {
        db: {name: summarize([r for r in rows.values() if r["db_id"] == db])
             for name, rows in stages.items()}
        for db in sorted({q["db_id"] for q in questions})
    }
    run_id = uuid.uuid4().hex[:12]
    output = Path(cfg["output_dir"]) / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "analysis_run_id": run_id, "split": "val", "n_samples": len(ids),
        "sources": sources, "file_sha256": hashes, "stages": summaries,
        "base_early_sft_pair": paired, "persistent_early_sft_symptoms": persistent_symptoms,
        "by_database": by_database,
        "persistent_execution_transitions": failure_transitions,
        "review": {"n": len(review), "question_ids": sorted(annotated),
                   "selection": "purposive examples across databases and transitions; biased",
                   "method": "Codex SQL/question/evidence review; "
                             "no repair or counterfactual replay"},
        "limitations": [
            "Historical scores preserved; no SQL replay or extractor correction applied.",
            "Early SFT is not current best SFT and is not the old GRPO parent checkpoint.",
            "Old GRPO is a historical diagnostic, not final official-reward GRPO.",
            "Val analysis is not train-side diagnosis; do not reuse val questions for training.",
            "Symptoms do not identify semantic root causes; capped/empty results need review.",
            "Dirty analysis run is internal only; source run metadata is preserved separately.",
        ],
    }
    write(output / "summary.json", result)
    write(output / "reviewed_cases.json", review)
    (output / "paired_records.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8"
    )
    report = ["# 已存档验证集结果：错误分布复核", "",
              f"分析运行 `{run_id}`，固定 val {len(ids)} 题，seed=0。仅读取旧文件。", "",
              "早期 SFT 不是当前 best SFT，旧 GRPO 也不是最终 GRPO；不能把三列当作同一训练链路。",
              "历史 official/strict 判分原样保留，不重新执行 SQL。",
              "本次 dirty 分析仅供内部诊断。", "",
              "| 阶段 | official 对 | strict 对 | 执行失败 | 可执行但 official 错 |",
              "|---|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        report.append(f"| {name} | {s['official_correct']} | {s['strict_correct']} | "
                      f"{s['failed_execution']} | {s['executed_but_official_wrong']} |")
    report += ["", "## 可观察错误类型", "", "| 类型 | Base | 早期 SFT | 旧 GRPO |",
               "|---|---:|---:|---:|"]
    buckets = sorted({b for s in summaries.values() for b in s["symptom_counts"]})
    for bucket in buckets:
        counts = [str(s["symptom_counts"].get(bucket, 0)) for s in summaries.values()]
        report.append(f"| {bucket} | " + " | ".join(counts) + " |")
    report += ["", "## Base 与早期 SFT 同题变化", "",
               "```json", json.dumps(paired, ensure_ascii=False, indent=2), "```", "",
               "两者都错的题中，早期 SFT 的症状：", "", "```json",
               json.dumps(persistent_symptoms, ensure_ascii=False, indent=2), "```", "",
               "全部配对 SQL、原始回答和判分见同目录 paired_records.jsonl。",
               "逐库统计见 summary.json，定向审阅例子见 reviewed_cases.json。",
               "这些验证题仅用于分析，不作为合成训练样本的原题或模板。"]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_run({
        "run_id": run_id, "stage": "ablation", "analysis_kind": "archived_val_failure_audit",
        "split": "val", "n_samples": len(ids), "seed": cfg["seed"],
        "config_path": str(config_path), "command": "python scripts/analyze_archived_stages.py",
        "model": "archived outputs only", "checkpoint": "see source_runs in summary",
        "decoding": {"inference": "not_run"},
        "source_run_ids": [s["run_id"] for s in sources.values()],
        "metrics": {"stages": summaries, "base_early_sft_pair": paired,
                    "persistent_early_sft_symptoms": persistent_symptoms,
                    "persistent_execution_transitions": failure_transitions,
                    "n_reviewed": len(review)},
        "log_path": str(output / "summary.json"), "notes": "; ".join(result["limitations"]),
    })
    print(json.dumps({"output": str(output), "stages": summaries, "paired": paired,
                      "persistent": persistent_symptoms}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
