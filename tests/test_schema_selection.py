"""Schema pruning must be deterministic and oracle leakage must stay explicit."""

from __future__ import annotations

from text2sql_rlvr.data import (
    discover_split,
    foreign_key_graph,
    linked_table_names,
    load_schema,
    oracle_table_names,
    render_selected_schema,
    required_tables_fk_connected,
)


def test_linked_selector_uses_question_and_fk_neighbour(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    example = split.load()[2]

    names = linked_table_names(schema, example.question, example.evidence, min_tables=1)
    assert "dept" in names
    assert "staff" in names


def test_oracle_selector_reads_tables_from_gold_sql(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")

    assert oracle_table_names(schema, "SELECT name FROM staff") == ("staff",)
    assert oracle_table_names(schema, "SELECT 'staff' FROM dept") == ("dept",)


def test_linker_splits_camel_case_identifiers():
    from text2sql_rlvr.data.schema import Column, DatabaseSchema, Table

    schema = DatabaseSchema(
        "d",
        (
            Table("UnitMeasure", (Column("UnitMeasureCode", "TEXT"),)),
            Table("Unrelated", (Column("value", "TEXT"),)),
        ),
    )
    assert linked_table_names(schema, "What is the unit measure code?", min_tables=1)[0] == (
        "UnitMeasure"
    )


def test_oracle_render_contains_only_gold_tables(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    example = split.load()[3]

    text, selection = render_selected_schema(schema, example, mode="oracle")
    assert selection.selected_tables == ("dept",)
    assert "CREATE TABLE dept" in text
    assert "CREATE TABLE staff" not in text


def test_linked_budget_keeps_complete_table_blocks(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    example = split.load()[2]

    text, selection = render_selected_schema(schema, example, mode="linked", max_chars=150)
    assert text.count("CREATE TABLE") == 1
    assert selection.exceeded_char_budget is True


def test_fk_graph_is_undirected_for_connectivity(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")

    graph = foreign_key_graph(schema)
    assert graph["staff"] == {"dept"}
    assert graph["dept"] == {"staff"}


def test_fk_connectivity_detects_missing_required_or_bridge_table():
    from text2sql_rlvr.data.schema import Column, DatabaseSchema, ForeignKey, Table

    schema = DatabaseSchema(
        "d",
        (
            Table("a", (Column("id", "INTEGER"),)),
            Table(
                "bridge",
                (Column("a_id", "INTEGER"), Column("c_id", "INTEGER")),
                (ForeignKey("a_id", "a", "id"), ForeignKey("c_id", "c", "id")),
            ),
            Table("c", (Column("id", "INTEGER"),)),
        ),
    )

    assert required_tables_fk_connected(schema, ("a", "c")) is True
    assert required_tables_fk_connected(
        schema, ("a", "c"), retained_tables=("a", "bridge", "c")
    ) is True
    assert required_tables_fk_connected(schema, ("a", "c"), retained_tables=("a", "c")) is False
    assert required_tables_fk_connected(schema, ("a", "c"), retained_tables=("a",)) is False
