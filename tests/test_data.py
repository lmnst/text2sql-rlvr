"""Split discovery, schema introspection and prompt construction."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from text2sql_rlvr.data import (
    PromptConfig,
    build_messages,
    build_user_prompt,
    discover_split,
    fetch_sample_rows,
    format_schema,
    load_examples,
    load_schema,
)
from text2sql_rlvr.data.schema import quote_identifier
from text2sql_rlvr.sql import extract_sql, validate_read_only


class TestSplitDiscovery:
    def test_finds_nested_release_directory(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        assert split.questions_path.name == "mini_dev_sqlite.json"
        assert split.databases_dir.name == "dev_databases"

    def test_db_path_points_at_a_real_file(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        assert split.db_path("company").is_file()

    def test_no_missing_databases(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        assert split.missing_databases(split.load()) == []

    def test_missing_database_is_reported(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        broken = [replace(split.load()[0], db_id="ghost")]
        assert split.missing_databases(broken) == ["ghost"]

    def test_unknown_split_name_raises(self, bird_root):
        with pytest.raises(ValueError):
            discover_split(bird_root, "test")

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_split(tmp_path / "absent", "mini_dev")

    def test_absent_question_file_names_what_it_looked_for(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="mini_dev_sqlite.json"):
            discover_split(tmp_path / "empty", "mini_dev")


class TestExampleLoading:
    def test_fields_are_parsed(self, bird_root):
        examples = discover_split(bird_root, "mini_dev").load()
        assert len(examples) == 5
        assert examples[2].evidence == "Research means dept_id = 1"
        assert examples[2].gold_sql == "SELECT name FROM staff WHERE dept_id = 1"
        assert examples[4].difficulty == "challenging"

    def test_accepts_query_instead_of_sql(self, tmp_path):
        path = tmp_path / "alt.json"
        path.write_text(
            json.dumps([{"db_id": "d", "question": "q?", "query": "SELECT 1"}]), encoding="utf-8"
        )
        examples = load_examples(path)
        assert examples[0].gold_sql == "SELECT 1"
        assert examples[0].question_id == 0

    def test_rejects_malformed_records(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"question": "no db"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="db_id"):
            load_examples(path)


class TestSchema:
    def test_tables_columns_and_keys(self, db_path):
        schema = load_schema(db_path, db_id="company")
        assert [t.name for t in schema.tables] == ["dept", "staff"]

        staff = schema.table("STAFF")
        assert staff is not None
        assert [c.name for c in staff.columns] == [
            "staff_id", "name", "dept_id", "salary", "note"
        ]
        assert staff.columns[0].primary_key is True
        assert staff.foreign_keys[0].to_table == "dept"
        assert staff.foreign_keys[0].to_column == "dept_id"

    def test_ddl_style_keeps_original_create_statements(self, db_path):
        text = format_schema(load_schema(db_path), style="ddl")
        assert "CREATE TABLE staff" in text
        assert "`head count`" in text

    def test_compact_style_lists_columns_and_links(self, db_path):
        text = format_schema(load_schema(db_path), style="compact")
        assert "# Table: staff" in text
        assert "dept_id -> dept.dept_id" in text

    def test_descriptions_are_merged_when_available(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        schema = load_schema(split.db_path("company"), db_id="company")
        text = format_schema(schema, style="compact", include_descriptions=True)
        assert "monthly salary in EUR" in text

    def test_descriptions_are_optional(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        schema = load_schema(split.db_path("company"), db_id="company")
        assert "monthly salary" not in format_schema(schema, style="compact")

    def test_sample_rows_are_readable(self, db_path):
        schema = load_schema(db_path)
        samples = fetch_sample_rows(db_path, schema, n=2)
        assert len(samples["staff"]) == 2
        assert samples["staff"][0][1] == "Ada"

    def test_unknown_style_raises(self, db_path):
        with pytest.raises(ValueError):
            format_schema(load_schema(db_path), style="yaml")

    def test_identifier_quoting_escapes_quotes(self):
        assert quote_identifier('we"ird') == '"we""ird"'


class TestPrompt:
    def test_contains_schema_question_and_instruction(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        example = split.load()[2]
        schema_text = format_schema(load_schema(split.db_path("company")), style="compact")

        prompt = build_user_prompt(example, schema_text)
        assert "# Table: staff" in prompt
        assert example.question in prompt
        assert "Research means dept_id = 1" in prompt
        assert "```sql" in prompt

    def test_evidence_can_be_withheld(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        example = split.load()[2]
        prompt = build_user_prompt(example, "schema", PromptConfig(include_evidence=False))
        assert "Research means" not in prompt

    def test_messages_have_system_and_user_turns(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        messages = build_messages(split.load()[0], "schema")
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_instruction_matches_what_the_extractor_expects(self, bird_root):
        """The prompt asks for a ```sql block; extract_sql must accept exactly that,
        and the result must survive read-only validation."""
        split = discover_split(bird_root, "mini_dev")
        example = split.load()[0]
        build_user_prompt(example, "schema")

        completion = "```sql\nSELECT count(*) FROM staff\n```"
        sql = extract_sql(completion)
        assert sql == "SELECT count(*) FROM staff"
        assert validate_read_only(sql).ok


class TestInstructionVersions:
    """v1 is the pinned prompt since milestone 7; v2 is kept for the negative result."""

    def test_default_is_v1(self):
        assert PromptConfig().instruction_version == "v1"

    def test_v1_is_the_original_short_instruction(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        config = PromptConfig(instruction_version="v1")
        prompt = build_user_prompt(split.load()[0], "schema", config)
        assert "Do not use functions from other databases" not in prompt
        assert "```sql" in prompt

    def test_v2_states_the_dialect_and_the_pseudo_functions(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        config = PromptConfig(instruction_version="v2")
        prompt = build_user_prompt(split.load()[0], "schema", config)
        assert "YEAR()" in prompt
        assert "DIVIDE(a, b)" in prompt
        assert "exactly one statement" in prompt

    def test_both_versions_still_ask_for_a_fenced_block(self, bird_root):
        split = discover_split(bird_root, "mini_dev")
        for version in ("v1", "v2"):
            prompt = build_user_prompt(
                split.load()[0], "schema", PromptConfig(instruction_version=version)
            )
            assert "```sql" in prompt, version

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="instruction_version"):
            PromptConfig(instruction_version="v3")

    def test_version_is_recorded_in_the_config_dict(self):
        assert PromptConfig(instruction_version="v1").as_dict()["instruction_version"] == "v1"
