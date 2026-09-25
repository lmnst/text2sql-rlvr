"""Selector data: labels from gold SQL, size-stratified held-out, hard cases first."""

from __future__ import annotations

import pytest

from text2sql_rlvr.data import BirdExample, discover_split, format_schema, load_schema
from text2sql_rlvr.data.schema import Column, DatabaseSchema, ForeignKey, Table
from text2sql_rlvr.data.selector import (
    SelectorCase,
    annotate_case,
    build_selector_record,
    format_selector_target,
    parse_selector_output,
    plan_selector_split,
    select_training_cases,
    selection_metrics,
)


def _examples(db_ids: dict[str, int]) -> list[BirdExample]:
    out = []
    qid = 0
    for db_id, n in db_ids.items():
        for _ in range(n):
            out.append(BirdExample(qid, db_id, "q", "", "SELECT 1"))
            qid += 1
    return out


def test_split_is_database_disjoint_and_stratified():
    sizes = {f"small{i}": 3 for i in range(6)}
    sizes.update({f"mid{i}": 6 for i in range(6)})
    sizes.update({f"big{i}": 12 for i in range(4)})
    sizes.update({f"huge{i}": 30 for i in range(4)})
    examples = _examples({db: 50 for db in sizes})
    examples += _examples({"tiny": 5})
    sizes["tiny"] = 3

    plan = plan_selector_split(examples, sizes, n_heldout=8, seed=0, min_questions=40)

    assert len(plan.heldout_db_ids) == 8
    assert not set(plan.heldout_db_ids) & set(plan.train_db_ids)
    assert "tiny" in plan.train_db_ids
    for prefix in ("small", "mid", "big", "huge"):
        assert sum(db.startswith(prefix) for db in plan.heldout_db_ids) == 2
    assert len(plan.train_ids) + len(plan.heldout_ids) == len(examples)
    assert plan == plan_selector_split(examples, sizes, n_heldout=8, seed=0, min_questions=40)


def test_split_excludes_generator_val_databases_from_both_sides():
    sizes = {f"db{i}": 5 for i in range(8)}
    examples = _examples({db: 50 for db in sizes})
    plan = plan_selector_split(
        examples, sizes, n_heldout=2, seed=1, floor_per_bucket=0, exclude_db_ids=("db0",)
    )
    assert "db0" not in plan.train_db_ids
    assert "db0" not in plan.heldout_db_ids
    assert len(plan.train_ids) + len(plan.heldout_ids) == 350


def test_annotate_case_labels_from_gold_and_flags_hard_cases(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    example = split.load()[2]  # "Who works in the Research department?" -> staff

    case = annotate_case(example, schema)
    assert case.gold_tables == ("staff",)
    assert case.n_tables_total == 2
    assert case.linked_keeps_all
    assert case.lexical_worst_rank <= 2
    assert not case.is_hard


def test_hard_reasons_cover_multi_table_missed_and_deep():
    base = dict(question_id=1, db_id="d", n_tables_total=20)
    multi = SelectorCase(gold_tables=("a", "b", "c"), linked_tables=("a", "b", "c"),
                         lexical_worst_rank=3, **base)
    missed = SelectorCase(gold_tables=("a",), linked_tables=("b",), lexical_worst_rank=2, **base)
    deep = SelectorCase(gold_tables=("a",), linked_tables=("a",), lexical_worst_rank=15, **base)
    easy = SelectorCase(gold_tables=("a",), linked_tables=("a", "b"), lexical_worst_rank=1, **base)
    assert multi.hard_reasons == ("multi_table",)
    assert missed.hard_reasons == ("linker_misses_gold",)
    assert deep.hard_reasons == ("gold_ranked_deep",)
    assert not easy.is_hard


def _case(qid: int, db: str, hard: bool) -> SelectorCase:
    return SelectorCase(qid, db, ("a",), 5, ("a",), 15 if hard else 1)


def test_training_selection_caps_per_db_and_prefers_hard():
    cases = [_case(i, "x", hard=i < 8) for i in range(20)]  # 8 hard, 12 easy
    cases += [_case(100 + i, "y", hard=False) for i in range(3)]
    chosen = select_training_cases(cases, cap_per_db=6, hard_fraction=0.5, seed=0)
    x = [c for c in chosen if c.db_id == "x"]
    y = [c for c in chosen if c.db_id == "y"]
    assert len(x) == 6 and sum(c.is_hard for c in x) == 3
    assert len(y) == 3
    assert chosen == select_training_cases(cases, cap_per_db=6, hard_fraction=0.5, seed=0)

    only_hard = [_case(i, "z", hard=True) for i in range(10)]
    assert len(select_training_cases(only_hard, cap_per_db=4, hard_fraction=0.5)) == 4
    assert len(select_training_cases(cases, cap_per_db=0)) == len(cases)


def test_target_round_trips_and_parser_tolerates_noise():
    schema = DatabaseSchema("d", (
        Table("Product", (Column("id", "INTEGER"),)),
        Table("ProductCostHistory", (Column("id", "INTEGER"),)),
        Table("Vendor", (Column("id", "INTEGER"),)),
    ))
    target = format_selector_target(("Product", "ProductCostHistory"))
    assert target == '["Product", "ProductCostHistory"]'
    assert parse_selector_output(target, schema) == ("Product", "ProductCostHistory")
    assert parse_selector_output('Sure: ["productcosthistory", "product"]', schema) == (
        "Product", "ProductCostHistory")
    assert parse_selector_output("Vendor, Nope, Product", schema) == ("Product", "Vendor")
    assert parse_selector_output('```json\n["Vendor"]\n```', schema) == ("Vendor",)
    assert parse_selector_output("nothing here", schema) == ()


def test_record_contains_schema_question_and_parsable_answer(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    example = split.load()[2]
    record = build_selector_record(example, schema, format_schema(schema), ("staff",))

    system, user, assistant = record["messages"]
    assert system["role"] == "system"
    assert "CREATE TABLE" in user["content"]
    assert example.question in user["content"]
    assert example.evidence in user["content"]
    assert user["content"].rstrip().endswith("spelled exactly as in the schema.")
    assert assistant["content"] == '["staff"]'
    with pytest.raises(ValueError):
        build_selector_record(example, schema, format_schema(schema), ("ghost",))


def test_selection_metrics_reward_all_gold_kept_and_penalise_over_selection():
    cases = [
        SelectorCase(1, "d", ("a", "b"), 10, (), 1),
        SelectorCase(2, "d", ("c",), 10, (), 1),
    ]
    keep_everything = {1: tuple("abcdefghij"), 2: tuple("abcdefghij")}
    m = selection_metrics(cases, keep_everything)
    assert m["all_gold_retained_rate"] == 1.0
    assert m["mean_precision"] == pytest.approx(0.15)
    assert m["mean_selected_tables"] == 10.0

    partial = {1: ("a",), 2: ("c", "d")}
    m = selection_metrics(cases, partial)
    assert m["all_gold_retained_rate"] == 0.5
    assert m["mean_table_recall"] == 0.75
    assert m["mean_precision"] == 0.75
    assert selection_metrics(cases, {})["n_missing_predictions"] == 2


def test_parser_accepts_plain_table_names_in_schema_order():
    tables = ["Product", "ProductCostHistory", "Vendor"]
    assert parse_selector_output('["vendor", "PRODUCT"]', tables) == ("Product", "Vendor")
    assert parse_selector_output("[]", tables) == ()


def test_selection_report_breaks_down_hard_easy_and_per_db():
    from text2sql_rlvr.data.selector import selection_report

    cases = [
        SelectorCase(1, "x", ("a", "b"), 10, ("a", "b", "c"), 12),  # hard: ranked deep
        SelectorCase(2, "x", ("c",), 10, ("c", "d"), 1),
        SelectorCase(3, "y", ("e",), 4, ("e",), 1),
    ]
    predicted = {1: ("a",), 2: ("c",), 3: ()}
    baseline = {c.question_id: c.linked_tables for c in cases}
    report = selection_report(cases, predicted, baseline=baseline)

    assert report["overall"]["model"]["all_gold_retained_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["overall"]["baseline"]["all_gold_retained_rate"] == 1.0
    assert report["hard"]["model"]["n"] == 1 and report["easy"]["model"]["n"] == 2
    assert set(report["per_db"]) == {"x", "y"}
    assert report["per_db"]["y"]["model"]["mean_selected_tables"] == 0.0
    assert report["n_empty_predictions"] == 1


def test_load_selected_tables_reads_evaluator_output(tmp_path):
    import json

    from text2sql_rlvr.data.selector import load_selected_tables

    path = tmp_path / "preds.jsonl"
    path.write_text(
        json.dumps({"question_id": 7, "predicted_tables": ["staff"]}) + "\n"
        + json.dumps({"question_id": 8, "predicted_tables": []}) + "\n",
        encoding="utf-8",
    )
    assert load_selected_tables(path) == {7: ("staff",), 8: ()}


def test_select_schema_uses_predicted_tables_verbatim(bird_root):
    from text2sql_rlvr.data import render_selected_schema, select_schema

    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    example = split.load()[0]

    selection = select_schema(schema, example, mode="full", tables=("STAFF", "ghost"))
    assert selection.mode == "predicted"
    assert selection.selected_tables == ("staff",)

    text, rendered = render_selected_schema(schema, example, mode="linked", tables=("dept",))
    assert rendered.mode == "predicted"
    assert "CREATE TABLE dept" in text
    assert "CREATE TABLE staff" not in text


def test_expand_selection_adds_fk_neighbours_then_lexical_matches(bird_root):
    from text2sql_rlvr.data.selector import expand_selection

    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")

    assert expand_selection(schema, ("staff",), fk_hops=0) == ("staff",)
    assert expand_selection(schema, ("staff",), fk_hops=1) == ("dept", "staff")
    assert expand_selection(schema, ("DEPT",), fk_hops=1) == ("dept", "staff")
    assert expand_selection(schema, (), fk_hops=1) == ()
    assert expand_selection(
        schema, (), fk_hops=0, lex_top_k=1, question="What are the department names?"
    ) == ("dept",)


def test_load_selected_tables_can_read_the_expanded_field(tmp_path):
    import json

    from text2sql_rlvr.data.selector import load_selected_tables

    path = tmp_path / "preds.jsonl"
    path.write_text(
        json.dumps({"question_id": 7, "predicted_tables": ["staff"],
                    "expanded_tables": ["dept", "staff"]}) + "\n",
        encoding="utf-8",
    )
    assert load_selected_tables(path, "expanded_tables") == {7: ("dept", "staff")}
    with pytest.raises(KeyError):
        load_selected_tables(path, "missing_field")


def test_parser_reads_json_object_and_takes_tables_from_column_prefixes():
    tables = ["Employee", "Person", "Gender"]
    text = '{"columns": ["Person.FirstName", "Employee.JobTitle"], "tables": ["Employee"]}'
    assert parse_selector_output(text, tables) == ("Employee", "Person")
    assert parse_selector_output('{"tables": ["gender"]}', tables) == ("Gender",)
    assert parse_selector_output('{"columns": ["Nope.x"]}', tables) == ()
    # a plain list still works when the model ignores the object format
    assert parse_selector_output('["Person"]', tables) == ("Person",)


def test_columns_target_round_trips_through_record(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    example = split.load()[2]
    record = build_selector_record(
        example, schema, format_schema(schema), ("staff",),
        target_format="columns", gold_columns=("staff.name", "staff.dept_id"),
    )
    assert record["messages"][-1]["content"] == (
        '{"columns": ["staff.name", "staff.dept_id"], "tables": ["staff"]}'
    )
    assert "table.column" in record["messages"][1]["content"]
    assert 'form {"columns"' in record["messages"][1]["content"]


def test_gold_columns_resolves_aliases_bare_names_and_flags_ambiguity(bird_root):
    from text2sql_rlvr.data.selector import gold_columns

    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")

    sql = ("SELECT T1.name, T2.`head count` FROM staff AS T1 INNER JOIN dept AS T2 "
           "ON T1.dept_id = T2.dept_id WHERE T2.name = 'Sales' ORDER BY salary DESC")
    columns, unresolved = gold_columns(schema, sql)
    assert columns == (
        "dept.dept_id", "dept.name", "dept.head count",
        "staff.name", "staff.dept_id", "staff.salary",
    )
    assert unresolved == ()

    # `name` exists in both used tables: attributed to neither, reported instead
    columns, unresolved = gold_columns(schema, "SELECT name FROM staff JOIN dept")
    assert columns == ()
    assert unresolved == ("?name",)

    # COUNT(*) yields no column; a string literal is never a column
    assert gold_columns(schema, "SELECT count(*) FROM dept WHERE name = 'salary'") == (
        ("dept.name",), ())


def test_trim_descriptions_cuts_text_and_drops_value_notes():
    from text2sql_rlvr.data.selector import trim_descriptions

    schema = DatabaseSchema("d", (
        Table("t", (
            Column("a", "TEXT", description="x" * 100, value_description="junk"),
            Column("b", "TEXT"),
        )),
    ))
    trimmed = trim_descriptions(schema, 80)
    assert trimmed.tables[0].columns[0].description == "x" * 80
    assert trimmed.tables[0].columns[0].value_description is None
    assert trimmed.tables[0].columns[1].description is None
    assert trim_descriptions(schema, 0).tables[0].columns[0].description is None


def test_expand_selection_cap_adds_neighbours_by_lexical_rank():
    from text2sql_rlvr.data.selector import expand_selection

    # hub -> a, b, c via foreign keys; the question mentions c, so c ranks first
    schema = DatabaseSchema("d", (
        Table("hub", (Column("id", "INTEGER"),), (
            ForeignKey("a_id", "a", "id"), ForeignKey("b_id", "b", "id"),
            ForeignKey("c_id", "c", "id"),
        )),
        Table("a", (Column("id", "INTEGER"),)),
        Table("b", (Column("id", "INTEGER"),)),
        Table("c", (Column("id", "INTEGER"), Column("colour", "TEXT"))),
    ))
    question = "What colour is c?"
    assert expand_selection(schema, ("hub",), question=question) == ("hub", "a", "b", "c")
    assert expand_selection(schema, ("hub",), question=question, cap=2) == ("hub", "c")
    assert expand_selection(schema, ("hub",), question=question, cap=1) == ("hub",)
    # lexical tables are added before the cap is applied to neighbours
    assert expand_selection(schema, ("a",), question=question, lex_top_k=1, cap=2) == ("a", "c")
