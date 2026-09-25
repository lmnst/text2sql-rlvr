"""Audit/export the targeted expansion and an unchanged, reviewed replay set."""

from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from build_targeted_sft import (
    digest_file,
    read,
    result_evidence,
    review_document,
    validate_sources,
    verify_pair,
    write,
)

from text2sql_rlvr.data import PromptConfig, format_schema, load_schema
from text2sql_rlvr.data.bird import BirdExample
from text2sql_rlvr.data.prompt import build_messages
from text2sql_rlvr.eval.failure_analysis import index_unique
from text2sql_rlvr.ledger import append_run, file_sha256, git_state
from text2sql_rlvr.rewards.compare import compare
from text2sql_rlvr.rewards.sandbox import SqlExecutor
from text2sql_rlvr.sql import extract_sql

CFG = Path("configs/data_construction/targeted_v2.json")
REPLAY = Path("configs/data_construction/replay_v2_review.json")


def text_key(text):
    return " ".join(text.split()).casefold()


def sql_key(sql):
    # This catches formatting-only duplicates, not all semantic equivalents.
    tokens = re.findall(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|\w+|[^\s]", sql)
    return tuple(t if t.startswith(("'", '"')) else t.casefold() for t in tokens)


def make_chat(example, schema):
    bird = BirdExample(
        example["question_id"],
        example["db_id"],
        example["question"],
        evidence=example.get("evidence", ""),
        gold_sql=example["SQL"],
    )
    messages = list(build_messages(bird, schema, PromptConfig()))
    # Preserve whitespace inside SQL string literals; do not use whitespace flattening.
    sql = example["SQL"].strip().rstrip(";")
    messages.append({"role": "assistant", "content": "```sql\n" + sql + "\n```"})
    if extract_sql(messages[-1]["content"]) != sql:
        raise ValueError("SQL target did not survive extraction")
    return {
        "question_id": example["question_id"],
        "db_id": example["db_id"],
        "component": example["component"],
        "messages": messages,
    }


def choose_replay(approved, n, seed):
    groups = defaultdict(list)
    for q in approved:
        groups[q["db_id"]].append(q)
    for group in groups.values():
        group.sort(key=lambda q: hashlib.sha256(f"{seed}:{q['question_id']}".encode()).hexdigest())
    result = []
    while any(groups.values()) and len(result) < n:
        for db in sorted(groups):
            if groups[db] and len(result) < n:
                result.append(groups[db].pop(0))
    return result


def mailing_scope(executor, db, n=60000):
    stats = executor.execute(
        db, "SELECT COUNT(*),COUNT(DISTINCT REFID),MIN(REFID),MAX(REFID) FROM Mailings1_2"
    )
    difference = executor.execute(
        db,
        "SELECT COUNT(*) FROM (SELECT REFID FROM Mailings1_2 EXCEPT SELECT ID FROM "
        f"(SELECT ID FROM Customers ORDER BY ID LIMIT {n}))",
    )
    valid = (
        stats.ok and difference.ok and stats.rows[0][:2] == (n, n) and difference.rows == ((0,),)
    )
    return valid, {"stats": result_evidence(stats), "outside_first_n": result_evidence(difference)}


def main():
    cfg, replay = read(CFG), read(REPLAY)
    split = read(cfg["split_manifest"])
    original = read(cfg["source_questions"])
    by_id = validate_sources(cfg, original, split)
    index_unique(replay["reviewed_candidates"], "question_id")
    if len(cfg["pairs"]) != 200 or any(n != 4 for n in cfg["template_instances"].values()):
        raise ValueError("Expected 50 templates with four pairs each")
    source_ids = {i for p in cfg["pairs"] for i in p["source_question_ids"]}
    original60 = read("data/targeted_sft/v1/runs/88ce3420e942/examples.json")
    v1_by_content = {(q["question"], q["SQL"]): q for q in original60}
    target_examples, checks, replay_audit, approved = [], [], [], []
    dbroot = Path(cfg["database_root"])

    def dbpath(db):
        return dbroot / db / (db + ".sqlite")

    used_questions, used_sql = set(), set()
    duplicate_target_sql = 0
    with SqlExecutor(timeout_s=10, max_rows=100000) as executor:
        for pair in cfg["pairs"]:
            if pair["review_status"] != "template_reviewed_instance_verified":
                raise ValueError("Missing template/instance provenance")
            check = verify_pair(pair, executor, dbpath(pair["db_id"]))
            checks.append(check)
            if not check.get("accepted"):
                raise ValueError(f"Target pair failed: {pair['pair_id']}")
            for variant in pair["variants"]:
                old = v1_by_content.get((variant["question"], variant["sql"]))
                qid = old["question_id"] if old else -200001 - len(target_examples)
                example = {
                    "question_id": qid,
                    "db_id": pair["db_id"],
                    "question": variant["question"],
                    "evidence": pair["evidence"],
                    "SQL": variant["sql"],
                    "targeted_id": pair["pair_id"] + variant["variant"],
                    "component": "targeted",
                    "synthetic": True,
                    "split": "train",
                    "template_id": pair["template_id"],
                    "pair_id": pair["pair_id"],
                    "family": pair["family"],
                    "source_question_ids": pair["source_question_ids"],
                    "inherited_v1": bool(old),
                }
                key = (example["db_id"], sql_key(example["SQL"]))
                duplicate_target_sql += int(key in used_sql)
                used_sql.add(key)
                used_questions.add((example["db_id"], text_key(example["question"])))
                target_examples.append(example)
        if len(used_questions) != 400 or sum(q["inherited_v1"] for q in target_examples) != 60:
            raise ValueError("Duplicate questions or the original 60 were not retained exactly")

        special = {
            863: "SELECT COUNT(*) FROM registration r JOIN student s ON s.student_id=r.student_id "
            "JOIN course c ON c.course_id=r.course_id WHERE r.grade='B' AND s.gpa>3 "
            "AND c.name='Machine Learning Theory'",
            2277: "SELECT COUNT(DISTINCT m.movieid) FROM movies m JOIN movies2directors d "
            "ON d.movieid=m.movieid WHERE m.country='USA' AND m.isEnglish='F' "
            "AND d.genre='Action'",
        }
        for annotation in replay["reviewed_candidates"]:
            qid = annotation["question_id"]
            q = by_id[qid]
            if q["db_id"] in split["val_db_ids"] or q["db_id"] != annotation["db_id"]:
                raise ValueError("Replay source mismatch or validation database")
            record = {"question_id": qid, "db_id": q["db_id"], "review": annotation, "failures": []}
            key = (q["db_id"], sql_key(q["SQL"]))
            qkey = (q["db_id"], text_key(q["question"]))
            if qid in source_ids:
                record["failures"].append("targeted_source_overlap")
            if key in used_sql or qkey in used_questions:
                record["failures"].append("duplicate_sql_or_question")
            r = executor.execute(dbpath(q["db_id"]), q["SQL"])
            record["execution"] = result_evidence(r)
            if not r.ok or r.truncated or not r.rows:
                record["failures"].append("execution_empty_or_failed_or_capped")
            if qid in special:
                alt = executor.execute(dbpath(q["db_id"]), special[qid])
                record["alternative_sql"] = special[qid]
                record["alternative_result"] = result_evidence(alt)
                if not compare(r, alt).strict:
                    record["failures"].append("independent_aggregate_disagrees")
            if qid in {8513, 8516, 8522, 8523, 8527}:
                valid, mailing = mailing_scope(executor, dbpath(q["db_id"]))
                record["mailing_scope"] = mailing
                if not valid:
                    record["failures"].append("first_60000_scope_not_verified")
            record["approved"] = not record["failures"]
            replay_audit.append(record)
            if record["approved"]:
                approved.append({**q, "component": "replay", "synthetic": False, "split": "train"})
                used_sql.add(key)
                used_questions.add(qkey)

    chosen = choose_replay(approved, replay["n_replay"], replay["seed"])
    run_id = uuid.uuid4().hex[:12]
    output = Path(cfg["output_root"]) / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    export = len(chosen) == replay["n_replay"]
    all_dbs = sorted({q["db_id"] for q in target_examples + chosen})
    selected_ids = {q["question_id"] for q in chosen}
    for a in replay_audit:
        a["selected"] = a["question_id"] in selected_ids
    write(output / "targeted_audit.json", checks)
    write(output / "replay_audit.json", replay_audit)
    write(output / "targeted_examples.json", target_examples)
    write(output / "replay_examples.json", chosen)
    write(output / "construction_spec.json", cfg)
    write(output / "replay_review_spec.json", replay)
    write(output / "source_records.json", [by_id[i] for i in sorted(source_ids | selected_ids)])
    combined = []
    if export:
        schemas = {
            db: format_schema(load_schema(dbpath(db), db_id=db), style="ddl") for db in all_dbs
        }
        for label, examples in (("targeted", target_examples), ("replay", chosen)):
            chats = [make_chat(q, schemas[q["db_id"]]) for q in examples]
            (output / f"{label}_sft.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chats), encoding="utf-8"
            )
            combined.extend(chats)
        random.Random(replay["seed"]).shuffle(combined)
        (output / "mixed_sft.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8"
        )
        write(output / "mixed_order.json", [r["question_id"] for r in combined])
        # The v1 renderer is shared; correct its title and describe template-level review honestly.
        doc = review_document(cfg, checks, run_id).replace("Targeted SFT v1", "Targeted SFT v2")
        doc = doc.replace(
            "每对两题均由 Codex 构造并审阅，未经独立人工复核。这里只展示结果前五行；",
            "50 个对照模板经 Codex 设计审阅，每模板实例化四对并执行核对。"
            "不是逐条独立人工审核；这里只展示前五行。",
        )
        doc = doc.replace("audit.json", "targeted_audit.json")
        (output / "targeted_review.md").write_text(doc, encoding="utf-8")
        rdoc = [
            "# 原始 replay 审核",
            "",
            "问题、evidence、gold 保持原样；Codex 审阅，非独立人工金标。",
            "每条都做执行检查；部分聚合与数据范围另有复核，详见 replay_audit.json。",
            "",
        ]
        annotations = {a["question_id"]: a for a in replay_audit}
        for q in chosen:
            a = annotations[q["question_id"]]
            rdoc += [
                f"## {q['question_id']} · {q['db_id']}",
                "",
                q["question"],
                "",
                "Evidence: " + q.get("evidence", ""),
                "",
                "```sql",
                q["SQL"],
                "```",
                "",
                a["review"]["note"],
                "",
                "执行预览：",
                "```json",
                json.dumps(a["execution"]["preview"], ensure_ascii=False),
                "```",
                "",
            ]
        (output / "replay_review.md").write_text("\n".join(rdoc), encoding="utf-8")
    metrics = {
        "n_targeted": 400,
        "n_targeted_pairs": 200,
        "n_templates": 50,
        "n_replay_reviewed": len(replay_audit),
        "n_replay_approved": len(approved),
        "n_replay_selected": len(chosen),
        "n_exported": len(combined),
        "n_inherited_v1": 60,
        "targeted_repeated_sql": duplicate_target_sql,
        "targeted_by_family": dict(Counter(q["family"] for q in target_examples)),
        "targeted_by_database": dict(Counter(q["db_id"] for q in target_examples)),
        "replay_by_database": dict(Counter(q["db_id"] for q in chosen)),
        "strict_contrast_pairs": sum(
            c["counterpart_sql_rejected_by_strict_result"] for c in checks
        ),
        "set_contrast_pairs": sum(c["counterpart_sql_rejected_by_set_result"] for c in checks),
    }
    sha, dirty = git_state()
    manifest = {
        "run_id": run_id,
        "status": "exported" if export else "replay_pool_insufficient",
        "git_sha": sha,
        "git_dirty": dirty,
        "metrics": metrics,
        "split": "train",
        "seed": 0,
        "targeted_slots": "deterministic SQL ordered grounding, 4 instances per template",
        "mix": "one copy per example; shuffle seed 0; no oversampling or curriculum",
        "source_overlap": sorted(source_ids & selected_ids),
        "selected_replay_ids": sorted(selected_ids),
        "targeted_source_ids": sorted(source_ids),
        "prompt_config": PromptConfig().as_dict(),
        "schema_mode": "full",
        "thinking": False,
        "model_calls": 0,
        "training_runs": 0,
        "database_sha256": {db: digest_file(dbpath(db)) for db in all_dbs},
        "code_sha256": {
            str(p): file_sha256(p)
            for p in (
                Path(__file__),
                Path("scripts/expand_targeted_sft.py"),
                Path("scripts/build_targeted_sft.py"),
            )
        },
        "config_sha256": {str(p): file_sha256(p) for p in (CFG, REPLAY)},
        "source_sha256": file_sha256(cfg["source_questions"]),
        "split_sha256": file_sha256(cfg["split_manifest"]),
        "limitations": [
            "400 examples from 50 templates, not 400 independent task types",
            "Replay reviewed by Codex, not independently human-labelled",
            "Witness relation SQL is shared; not formal semantic equivalence",
            "No tokenizer length measurement or model/training effectiveness test",
            "Full schema export; no controlled comparison with linked best SFT",
        ],
    }
    manifest["outputs_sha256"] = {p.name: file_sha256(p) for p in output.iterdir()}
    write(output / "manifest.json", manifest)
    append_run(
        {
            "run_id": run_id,
            "stage": "ablation",
            "analysis_kind": "targeted_replay_dataset",
            "split": "train",
            "n_samples": 400 + len(chosen),
            "seed": 0,
            "config_path": str(CFG),
            "command": "python scripts/build_targeted_replay_v2.py",
            "model": "not_run",
            "checkpoint": "not_used",
            "decoding": {"inference": "not_run"},
            "metrics": metrics,
            "log_path": str(output / "manifest.json"),
            "notes": "Dataset audit only, not model accuracy; no training or inference.",
        }
    )
    print(
        json.dumps(
            {"output": str(output), "status": manifest["status"], **metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
