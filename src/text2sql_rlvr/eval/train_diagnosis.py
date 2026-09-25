"""Train-only paired diagnosis. Observable transitions are not semantic root causes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from text2sql_rlvr.eval.failure_analysis import index_unique


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def choose_subset(train: list[dict], manifest: dict, size: int, seed: int) -> list[dict]:
    by_id = index_unique(train, "question_id")
    if set(by_id) != set(manifest["train_ids"]):
        raise ValueError("train input does not match the fixed training partition")
    if set(by_id) & set(manifest["val_ids"]):
        raise ValueError("train/val question overlap")
    groups = defaultdict(list)
    for row in train:
        if row["db_id"] in manifest["val_db_ids"]:
            raise ValueError("train/val database overlap")
        groups[row["db_id"]].append(row)
    if not len(groups) <= size <= len(train):
        raise ValueError("subset size must cover every database and not exceed train")
    # Allocate one seat per database, then distribute remaining seats in
    # proportion to remaining rows. Exact integer remainders avoid float ties.
    remaining = size - len(groups)
    capacity = len(train) - len(groups)
    quotas = {db: 1 for db in groups}
    if remaining:
        for db, rows in groups.items():
            quotas[db] += remaining * (len(rows) - 1) // capacity
        ranked = sorted(groups, key=lambda db: (
            -(remaining * (len(groups[db]) - 1) % capacity), db
        ))
        for db in ranked[:size - sum(quotas.values())]:
            quotas[db] += 1
    selected = []
    for db in sorted(groups):
        ordered = sorted(groups[db], key=lambda r: json_hash([seed, db, r["question_id"]]))
        selected.extend(ordered[:quotas[db]])
    return sorted(selected, key=lambda r: r["question_id"])


def verify_predictions(rows: list[dict], prompts: list[dict], arm: str) -> dict:
    indexed = index_unique(rows, "question_id")
    if set(indexed) != {r["question_id"] for r in prompts}:
        raise ValueError(f"{arm}: predictions must cover the exact frozen subset")
    for prompt in prompts:
        row = indexed[prompt["question_id"]]
        if row["prompt_sha256"] != prompt["prompt_sha256"] or row["db_id"] != prompt["db_id"]:
            raise ValueError(f"{arm}: prompt or database mismatch")
        if row.get("error") or not isinstance(row.get("completion"), str):
            raise ValueError(f"{arm}: incomplete inference; retry before diagnosis")
    return indexed


def transition(base: dict, sft: dict) -> str:
    if not base["comparable"] or not sft["comparable"]:
        return "unscorable"
    return {
        (False, False): "persistent_failure", (False, True): "resolved",
        (True, False): "regression", (True, True): "both_correct",
    }[(base["official"], sft["official"])]


def paired_summary(pairs: list[dict]) -> dict:
    scored = [r for r in pairs if r["transition"] != "unscorable"]
    n = len(scored)
    counts = dict(sorted(Counter(r["transition"] for r in pairs).items()))
    arms = {}
    for arm in ("base", "sft"):
        rows = [r[arm] for r in scored]
        correct = sum(r["official"] for r in rows)
        strict = sum(r["strict"] for r in rows)
        arms[arm] = {
            "denominator": n, "official_correct": correct, "strict_correct": strict,
            "official_ex_comparable": round(100 * correct / n, 2) if n else None,
            "strict_ex_comparable": round(100 * strict / n, 2) if n else None,
            "symptoms_comparable": dict(sorted(Counter(r["bucket"] for r in rows).items())),
            "symptoms_all": dict(sorted(Counter(r[arm]["bucket"] for r in pairs).items())),
        }
    return {
        "split": "train", "n_samples": len(pairs), "n_pair_comparable": n,
        "n_pair_unscorable": len(pairs) - n, "transitions": counts, "arms": arms,
        "persistent_symptom_transitions": dict(sorted(Counter(
            f"{r['base']['bucket']} -> {r['sft']['bucket']}"
            for r in pairs if r["transition"] == "persistent_failure"
        ).items())),
        "note": "Train-side/resubstitution diagnosis, not generalization. Shared complete-result "
                "denominator excludes gold failure, any execution timeout or result truncation.",
    }
