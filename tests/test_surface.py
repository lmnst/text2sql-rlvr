"""The exploitable-surface measurement, which decides whether the reward-hacking
line of work has anything to bite on."""

from __future__ import annotations

from dataclasses import replace

from text2sql_rlvr.data import discover_split
from text2sql_rlvr.eval.surface import gold_facts, summarise
from text2sql_rlvr.rewards.sandbox import execute_sql


def facts_for(db_path, example, sql):
    return gold_facts(replace(example, gold_sql=sql), execute_sql(db_path, sql))


def test_duplicate_rows_are_counted(bird_root):
    split = discover_split(bird_root, "mini_dev")
    example = split.load()[0]
    db_path = split.db_path("company")

    # salaries are 100, 90, 90, 80, 80 -> two duplicate rows
    item = facts_for(db_path, example, "SELECT salary FROM staff")
    assert item.n_rows == 5
    assert item.n_duplicate_rows == 2
    assert item.has_duplicates is True
    assert item.exploitable is True


def test_distinct_result_is_not_exploitable(bird_root):
    split = discover_split(bird_root, "mini_dev")
    example = split.load()[0]
    item = facts_for(split.db_path("company"), example, "SELECT name FROM staff")
    assert item.n_duplicate_rows == 0
    assert item.exploitable is False


def test_empty_result_is_exploitable(bird_root):
    split = discover_split(bird_root, "mini_dev")
    example = split.load()[0]
    item = facts_for(split.db_path("company"), example, "SELECT name FROM staff WHERE 1 = 0")
    assert item.empty is True
    assert item.exploitable is True


def test_duplicates_use_canonical_values(bird_root):
    """1 and 1.0 are the same answer, so they count as a duplicate pair."""
    split = discover_split(bird_root, "mini_dev")
    example = split.load()[0]
    item = facts_for(split.db_path("company"), example, "SELECT 1 UNION ALL SELECT 1.0")
    assert item.n_duplicate_rows == 1


def test_order_by_is_detected(bird_root):
    split = discover_split(bird_root, "mini_dev")
    example = split.load()[0]
    db_path = split.db_path("company")

    ordered = facts_for(db_path, example, "SELECT name FROM staff ORDER BY name")
    plain = facts_for(db_path, example, "SELECT name FROM staff")
    assert ordered.has_top_level_order_by is True
    assert plain.has_top_level_order_by is False


def test_failed_gold_is_neither_ok_nor_exploitable(bird_root):
    split = discover_split(bird_root, "mini_dev")
    example = split.load()[0]
    item = facts_for(split.db_path("company"), example, "SELECT nope FROM staff")
    assert item.status == "error"
    assert item.exploitable is False


def test_summary_aggregates_and_lists_ids(bird_root):
    split = discover_split(bird_root, "mini_dev")
    examples = split.load()
    db_path = split.db_path("company")

    facts = [
        facts_for(db_path, examples[0], "SELECT salary FROM staff"),          # duplicates
        facts_for(db_path, examples[1], "SELECT name FROM staff WHERE 1 = 0"),  # empty
        facts_for(db_path, examples[2], "SELECT name FROM staff"),            # clean
        facts_for(db_path, examples[3], "SELECT nope FROM staff"),            # broken
    ]
    surface = summarise(facts)

    assert surface.n == 4
    assert surface.n_executable == 3
    assert surface.n_empty == 1
    assert surface.n_with_duplicates == 1
    assert surface.n_exploitable == 2
    assert surface.exploitable_ids == (0, 1)
    assert surface.failed_ids == (3,)
    assert surface.rate(surface.n_exploitable) == 50.0


def test_summary_of_nothing_does_not_divide_by_zero():
    surface = summarise([])
    assert surface.n == 0
    assert surface.rate(0) == 0.0
