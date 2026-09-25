"""Expand reviewed templates using grounded slots; no model inference or training."""

from __future__ import annotations

import copy
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from build_targeted_sft import read, validate_sources, verify_pair, write

from text2sql_rlvr.ledger import append_run
from text2sql_rlvr.rewards.sandbox import SqlExecutor


def literal(value):
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (int, float)):
        return str(value)
    raise ValueError("Only non-null text/numeric slots are supported")


def map_strings(obj, fn, key=""):
    if isinstance(obj, dict):
        return {k: map_strings(v, fn, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_strings(v, fn, key) for v in obj]
    return fn(obj, key)


def change_literal(pair, old, new):
    def change(value, key):
        if not isinstance(value, str):
            return value
        if key in ("sql", "witness_sql"):
            return value.replace(literal(old), literal(new))
        if value == old:
            return new
        if key in ("question", "evidence", "focus"):
            return value.replace(old, new)
        return value

    result = map_strings(pair, change)
    result.setdefault("slot_changes", []).append({"old": old, "new": new})
    return result


def edit_sql_question(pair, old_sql, new_sql, suffix):
    result = copy.deepcopy(pair)
    result["witness_sql"] = result["witness_sql"].replace(old_sql, new_sql)
    for v in result["variants"]:
        v["sql"] = v["sql"].replace(old_sql, new_sql)
        v["question"] += " " + suffix
    result.setdefault("slot_changes", []).append({"scope": suffix})
    return result


def change_range(pair, lo, hi, width):
    result = copy.deepcopy(pair)
    for key in ("witness_sql",):
        result[key] = result[key].replace(f"BETWEEN 1 AND {width}", f"BETWEEN {lo} AND {hi}")
    for v in result["variants"]:
        v["sql"] = v["sql"].replace(f"BETWEEN 1 AND {width}", f"BETWEEN {lo} AND {hi}")
        v["question"] = v["question"].replace(f"1 through {width}", f"{lo} through {hi}")
    result["slot_changes"] = [{"range": [lo, hi]}]
    return result


def change_threshold(pair, old, new, word):
    result = copy.deepcopy(pair)
    for v in result["variants"]:
        v["sql"] = v["sql"].replace(f">{old}", f">{new}")
        v["question"] = v["question"].replace(f"more than {word}", f"more than {new}")
        v["reference"]["threshold"] = new
    result["slot_changes"] = [{"threshold": new}]
    return result


def instantiate(template, values):
    args = dict(zip(template["slot_names"], values, strict=True))
    args.update({"sql_" + k: literal(v) for k, v in list(args.items())})

    def fill(value, key):
        if isinstance(value, str):
            if value.startswith("{") and value.endswith("}") and value[1:-1] in args:
                return args[value[1:-1]]
            return value.format_map(args)
        return value

    result = map_strings(template, fill)
    result["slot_values"] = dict(zip(template["slot_names"], values, strict=True))
    result["pair_id"] = template["template_id"]
    return result


def generate():
    base = read("configs/data_construction/targeted_v1.json")
    new = read("configs/data_construction/targeted_v2_new_templates.json")
    attempts, grounding, selected = [], [], []
    counts = defaultdict(int)
    seen_questions, seen_pair_sql = set(), set()
    root = Path(base["database_root"])
    with SqlExecutor(timeout_s=10, max_rows=100000) as executor:

        def dbpath(db):
            return root / db / (db + ".sqlite")

        def slots(db, sql):
            r = executor.execute(dbpath(db), sql)
            if not r.ok or r.truncated:
                raise ValueError(f"Grounding failed for {db}: {r.error}")
            grounding.append({"db_id": db, "sql": sql, "rows": r.rows})
            return r.rows

        def offer(pair, template_id, inherited=False):
            if counts[template_id] >= 4:
                return
            pair = copy.deepcopy(pair)
            pair["template_id"] = template_id
            pair["pair_id"] = f"{template_id}.v{counts[template_id]}"
            pair["review_status"] = "template_reviewed_instance_verified"
            pair["inherited_v1"] = inherited
            check = verify_pair(pair, executor, dbpath(pair["db_id"]))
            signature = (pair["db_id"], tuple(v["sql"] for v in pair["variants"]))
            duplicate = signature in seen_pair_sql or any(
                v["question"] in seen_questions for v in pair["variants"]
            )
            attempts.append(
                {"template_id": template_id, "pair": pair, "audit": check, "duplicate": duplicate}
            )
            if not check.get("accepted") or duplicate:
                return
            selected.append(pair)
            seen_pair_sql.add(signature)
            seen_questions.update(v["question"] for v in pair["variants"])
            counts[template_id] += 1

        for p in base["pairs"]:
            name, db = p["pair_id"], p["db_id"]
            offer(p, name, inherited=True)
            variants = []
            if name == "E01":
                titles = slots(
                    db, "SELECT DISTINCT positiontitle FROM position ORDER BY positionID"
                )
                cities = slots(db, "SELECT DISTINCT locationcity FROM location ORDER BY locationID")
                variants = [
                    change_literal(
                        change_literal(p, "Regional Manager", x[0]), "New York City", y[0]
                    )
                    for x, y in zip(titles, cities, strict=False)
                ]
            elif name == "E02":
                variants = [
                    change_literal(p, "Award", r[0])
                    for r in slots(
                        db, "SELECT criteria_name FROM ranking_criteria ORDER BY id LIMIT 24"
                    )
                ]
            elif name == "E03":
                variants = [
                    change_literal(p, "The Illuminati", r[0])
                    for r in slots(
                        db,
                        "SELECT b.title FROM book b WHERE EXISTS(SELECT 1 FROM book_author ba "
                        "WHERE ba.book_id=b.book_id) ORDER BY b.book_id LIMIT 24",
                    )
                ]
            elif name in ("E04", "O03"):
                variants = [
                    change_literal(p, "USA", r[0])
                    for r in slots(
                        "world",
                        "SELECT c.Code FROM Country c WHERE c.Capital IS NOT NULL AND "
                        "EXISTS(SELECT 1 FROM CountryLanguage l WHERE l.CountryCode=c.Code) "
                        "ORDER BY c.Code LIMIT 24",
                    )
                ]
            elif name == "E05":
                variants = [
                    change_literal(change_literal(p, "FL", short), "Florida", long)
                    for short, long in [
                        ("NY", "New York"),
                        ("NJ", "New Jersey"),
                        ("CA", "California"),
                        ("IL", "Illinois"),
                        ("TX", "Texas"),
                    ]
                ]
            elif name in ("E06", "O05"):
                original = "Paris" if name == "E06" else "Sydney"
                variants = [
                    change_literal(p, original, r[0])
                    for r in slots(db, "SELECT city FROM offices ORDER BY officeCode")
                ]
            elif name in ("E07", "A03", "O02"):
                width = 10 if name == "E07" else 100
                variants = [
                    change_range(p, 1 + width * n, width * (n + 1), width)
                    for n in range(1, 100 if name == "E07" else 8)
                ]
            elif name in ("E08", "A09", "O09"):
                for r in slots(
                    db, "SELECT DISTINCT FL_DATE FROM Airlines ORDER BY FL_DATE LIMIT 24"
                ):
                    v = change_literal(p, "2018/8/1", r[0])
                    pretty = datetime.strptime(r[0], "%Y/%m/%d").strftime("%B %d, %Y")
                    for a in v["variants"]:
                        a["question"] = a["question"].replace("August 1, 2018", pretty)
                    v["evidence"] = (
                        "Flight date: "
                        + r[0]
                        + ". "
                        + (
                            "CANCELLED=1 means cancelled."
                            if name == "O09"
                            else "DEP_DELAY>0 means a departure delay."
                            if name == "A09"
                            else ""
                        )
                    )
                    variants.append(v)
            elif name in ("E09", "O07"):
                variants = [
                    change_literal(p, "Brown Suga Diaries", r[0])
                    for r in slots(db, "SELECT title FROM podcasts ORDER BY podcast_id LIMIT 24")
                ]
            elif name in ("E10", "O04"):
                people = slots(
                    db, "SELECT first_name,last_name FROM driver ORDER BY driver_id LIMIT 24"
                )
                companies = slots(db, "SELECT cust_name FROM customer ORDER BY cust_id LIMIT 24")
                for person, company in zip(people, companies, strict=False):
                    v = change_literal(change_literal(p, "Sue", person[0]), "Newell", person[1])
                    if name == "E10":
                        v = change_literal(v, "S K L Enterprises Inc", company[0])
                    variants.append(v)
            elif name in ("A01", "A02", "A05", "A06", "A07", "A08"):
                old, word = {
                    "A01": (2, "two"),
                    "A02": (3, "three"),
                    "A05": (10, "ten"),
                    "A06": (10, "ten"),
                    "A07": (2, "two"),
                    "A08": (5, "five"),
                }[name]
                variants = [
                    change_threshold(p, old, n, word)
                    for n in ([0, 1, 3, 4, 5, 6] if old < 5 else [1, 2, 3, 4, 20, 30])
                    if n != old
                ]
            elif name == "A04":
                variants = [
                    change_literal(p, "%Life%", "%" + word + "%")
                    for word in ("Dream", "News", "Health", "Science", "History", "Love")
                ]
                for v, word in zip(
                    variants, ("Dream", "News", "Health", "Science", "History", "Love"), strict=True
                ):
                    for a in v["variants"]:
                        a["question"] = a["question"].replace("contains Life", "contains " + word)
            elif name == "A10":
                variants = [
                    edit_sql_question(
                        p,
                        "num_pages IS NOT NULL",
                        f"num_pages IS NOT NULL AND num_pages>={n}",
                        f"Only include books with at least {n} pages.",
                    )
                    for n in (50, 100, 200)
                ]
            elif name == "O01":
                variants = [change_literal(p, "Good", x) for x in ("Poor", "Average")]
                variants.append(
                    edit_sql_question(
                        p,
                        "performance='Good'",
                        "performance='Good' AND gender='M'",
                        "Only include male employees.",
                    )
                )
            elif name == "O06":
                variants = [
                    change_literal(p, "Thomas Nelson", r[0])
                    for r in slots(
                        db,
                        "SELECT p.publisher_name FROM publisher p WHERE EXISTS(SELECT 1 "
                        "FROM book b WHERE b.publisher_id=p.publisher_id) "
                        "ORDER BY p.publisher_id LIMIT 24",
                    )
                ]
            elif name == "O08":
                variants = [
                    edit_sql_question(
                        p,
                        "ranking_system_id IS NOT NULL",
                        f"ranking_system_id IS NOT NULL AND ranking_system_id IN ({values})",
                        f"Only include system IDs in ({values}).",
                    )
                    for values in ("1,2", "1,3", "2,3")
                ]
            elif name == "O10":
                variants = [
                    edit_sql_question(
                        p,
                        "FROM employee",
                        "FROM employee WHERE gender=" + literal(g),
                        "Only include " + word + " employees.",
                    )
                    for g, word in (("M", "male"), ("F", "female"))
                ]
                variants.append(
                    edit_sql_question(
                        p,
                        "FROM employee",
                        "FROM employee WHERE performance IN ('Good','Poor')",
                        "Exclude ratings other than Good and Poor.",
                    )
                )
            else:
                raise ValueError(name)
            for v in variants:
                offer(v, name)
            print(name, counts[name], flush=True)

        for template in new:
            for row in slots(template["db_id"], template["slots_sql"]):
                offer(instantiate(template, row), template["template_id"])
            print(template["template_id"], counts[template["template_id"]], flush=True)

    cfg = {
        **base,
        "version": "targeted_sft_v2",
        "output_root": "data/targeted_sft/v2",
        "pairs": selected,
        "template_instances": dict(counts),
        "construction_method": "50 reviewed templates, up to four grounded instances each",
    }
    validate_sources(cfg, read(base["source_questions"]), read(base["split_manifest"]))
    run_id = uuid.uuid4().hex[:12]
    evidence = Path("data/targeted_sft/v2/preparation/runs") / run_id
    evidence.mkdir(parents=True, exist_ok=True)
    write(evidence / "attempts.json", attempts)
    write(evidence / "grounding.json", grounding)
    write(evidence / "constructed_config.json", cfg)
    write(Path("data/targeted_sft/v2/preparation/latest.json"), {"path": str(evidence)})
    entry = append_run(
        {
            "run_id": run_id,
            "stage": "ablation",
            "analysis_kind": "targeted_expansion_preparation",
            "split": "train",
            "n_samples": 2 * len(selected),
            "seed": 0,
            "config_path": "configs/data_construction/targeted_v2_new_templates.json",
            "command": "python scripts/expand_targeted_sft.py",
            "model": "not_run",
            "checkpoint": "not_used",
            "decoding": {},
            "metrics": {
                "selected_pairs": len(selected),
                "attempted_pairs": len(attempts),
                "instances_per_template": dict(counts),
            },
            "log_path": str(evidence / "constructed_config.json"),
            "notes": "Preparation only, no SFT export, no model or training.",
        }
    )
    print(entry["run_id"], "selected", len(selected))


if __name__ == "__main__":
    generate()
