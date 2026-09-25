"""Offline symptoms and reproducible audit sampling, without guessing SQL root causes."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from text2sql_rlvr.sql import extract_sql, iter_code_words


def index_unique(rows: list[dict], key: str) -> dict:
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def symptom(row: dict) -> str:
    """One mutually exclusive observable bucket; strict reason is not semantic cause."""
    if row["gold_status"] != "ok":
        return "gold_execution_failed"
    if row["official"]:
        return "correct_both" if row["strict"] else "official_only"
    if row["pred_status"] != "ok":
        error = (row.get("pred_error") or "").lower()
        for fragment, name in (
            ("no such column", "missing_column"),
            ("no such table", "missing_table"),
            ("no such function", "missing_function"),
            ("ambiguous column", "ambiguous_column"),
            ("syntax error", "syntax_error"),
        ):
            if fragment in error:
                return name
        return "execution_" + row["pred_status"]
    if row["strict"]:
        return "strict_only"
    return "result_" + str(row["reason"])


def align_records(
    questions: list[dict], outcomes: list[dict], predictions: list[dict], *, id_mode: str
) -> list[dict]:
    """Explicit historical position mode; verify every match, never silently fall back."""
    if id_mode not in ("position", "original"):
        raise ValueError("id_mode must be position or original")
    questions_by_id = index_unique(questions, "question_id")
    outcomes_by_id = index_unique(outcomes, "question_id")
    predictions_by_id = index_unique(predictions, "question_id")
    expected = set(range(len(questions))) if id_mode == "position" else set(questions_by_id)
    if set(outcomes_by_id) != expected or set(predictions_by_id) != expected:
        raise ValueError("outcomes/predictions must cover the exact full question set")
    joined = []
    for old_id in sorted(expected):
        q = questions[old_id] if id_mode == "position" else questions_by_id[old_id]
        o, p = outcomes_by_id[old_id], predictions_by_id[old_id]
        if not (q["db_id"] == o["db_id"] == p["db_id"]):
            raise ValueError(f"database mismatch at historical id {old_id}")
        if q["SQL"].strip() != o["gold_sql"].strip():
            raise ValueError(f"gold SQL mismatch at historical id {old_id}")
        if p["sql"].strip() != o["pred_sql"].strip():
            raise ValueError(f"prediction SQL mismatch at historical id {old_id}")
        joined.append({
            **o, "historical_question_id": old_id, "question_id": q["question_id"],
            "question": q["question"], "evidence": q.get("evidence", ""),
            "completion": p["completion"], "finish_reason": p.get("finish_reason"),
            "usage": p.get("usage", {}), "bucket": symptom(o),
            "reextracted_sql": extract_sql(p["completion"]),
            "extraction_changed": extract_sql(p["completion"]) != o["pred_sql"],
        })
    return joined


def stratified_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Round robin across database x symptom strata, hash-shuffled within each.

    Equal coverage is intentional. Root-cause frequencies in this audit are NOT
    estimates of population prevalence; small strata are oversampled.
    """
    strata = defaultdict(list)
    for row in rows:
        strata[(row["db_id"], row["bucket"])].append(row)

    def rank(row):
        key = f"{seed}:{row['db_id']}:{row['question_id']}".encode()
        return hashlib.sha256(key).hexdigest()

    queues = [sorted(strata[k], key=rank) for k in sorted(strata)]
    selected = []
    while queues and len(selected) < size:
        for queue in queues:
            if len(selected) == size:
                break
            selected.append(queue.pop(0))
        queues = [q for q in queues if q]
    return selected


def training_seed_catalog(
    train: list[dict], val: list[dict], allowed_ids: set[int], *, per_family: int, seed: int
) -> list[dict]:
    """Find structural candidates in train, not measured failures or synthetic data."""
    index_unique(train, "question_id")
    val_ids = {q["question_id"] for q in val}
    val_dbs = {q["db_id"] for q in val}
    if {q["question_id"] for q in train} != allowed_ids:
        raise ValueError("train candidates must match the fixed train partition")
    if any(q["question_id"] in val_ids or q["db_id"] in val_dbs for q in train):
        raise ValueError("train/val leakage in seed candidates")
    families = defaultdict(list)
    for q in train:
        words = {w for w, _d, _n in iter_code_words(q["SQL"])}
        rules = {
            "schema_grounding": "JOIN" in words,
            "entity_and_filters": {"JOIN", "WHERE"} <= words,
            "aggregation_grain": bool({"COUNT", "AVG", "SUM"} & words),
            "sqlite_dates_arithmetic": bool({"STRFTIME", "JULIANDAY"} & words),
            "ranking_and_scope": {"ORDER", "BY", "LIMIT"} <= words,
            "projection_cardinality": "DISTINCT" in words,
        }
        for family, matches in rules.items():
            if matches:
                families[family].append({**q, "bucket": family})
    candidates = []
    used = set()
    for family in sorted(families):
        pool = [r for r in families[family] if r["question_id"] not in used]
        for row in stratified_sample(pool, per_family, seed):
            used.add(row["question_id"])
            candidates.append({
                **row, "target_family": family, "split": "train",
                "selection_basis": "gold SQL lexical structure only; not an observed failure",
                "synthetic": False, "approved_for_training": False,
                "review_required": "Check question/evidence/gold agreement, schema, values, "
                                   "result multiplicity and counterexamples before synthesis.",
            })
    return candidates


def summarize(rows: list[dict]) -> dict:
    """All counts refer to the supplied set, with explicit denominators."""
    n = len(rows)
    official = sum(r["official"] for r in rows)
    strict = sum(r["strict"] for r in rows)
    failed = [r for r in rows if not r["official"]]
    return {
        "n_samples": n, "official_correct": official, "strict_correct": strict,
        "official_ex": round(100 * official / n, 2),
        "strict_ex": round(100 * strict / n, 2),
        "official_failures": len(failed),
        "failed_execution": sum(r["pred_status"] != "ok" for r in failed),
        "executed_but_official_wrong": sum(r["pred_status"] == "ok" for r in failed),
        "symptom_counts": dict(sorted(Counter(r["bucket"] for r in rows).items())),
        "length_limited": sum(r["finish_reason"] == "length" for r in rows),
        "gold_empty": sum(r["gold_empty"] for r in rows),
        "extraction_changed": sum(r["extraction_changed"] for r in rows),
        "strict_truncated": sum(r["reason"] == "truncated" for r in rows),
        "official_only_reasons": dict(sorted(Counter(
            r["reason"] for r in rows if r["official"] and not r["strict"]
        ).items())),
        "official_pass_gold_empty": sum(r["gold_empty"] and r["official"] for r in rows),
        "missing_functions": dict(sorted(Counter(
            r["pred_error"].rsplit(":", 1)[-1].strip().upper()
            for r in rows if r["bucket"] == "missing_function"
        ).items())),
    }
