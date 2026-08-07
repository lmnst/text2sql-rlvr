"""Sandbox behaviour: what it returns, what it refuses, and what it survives."""

from __future__ import annotations

import sqlite3

import pytest

from text2sql_rlvr.rewards.sandbox import (
    ERROR,
    OK,
    REJECTED,
    TIMEOUT,
    SqlExecutor,
    connect,
    execute_on,
    execute_sql,
)

_INFINITE = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
_COUNT_TO = (
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100) SELECT x FROM c"
)


def test_select_returns_rows_and_columns(db_path):
    result = execute_sql(db_path, "SELECT name, salary FROM staff ORDER BY staff_id")
    assert result.status == OK
    assert result.columns == ("name", "salary")
    assert result.rows[0] == ("Ada", 100.0)
    assert result.n_rows == 5


def test_empty_result_still_reports_columns(db_path):
    result = execute_sql(db_path, "SELECT name, salary FROM staff WHERE 1 = 0")
    assert result.status == OK
    assert result.rows == ()
    assert len(result.columns) == 2


def test_validation_rejects_before_touching_the_database(db_path):
    result = execute_sql(db_path, "DROP TABLE staff")
    assert result.status == REJECTED
    assert "DROP" in (result.error or "")


def test_authorizer_blocks_what_validation_lets_through(db_path):
    # validate=False simulates a hole in the text-level check; the authorizer is
    # the layer that has to hold in that case.
    conn = connect(db_path)
    try:
        result = execute_on(conn, "PRAGMA table_info(staff)", validate=False)
    finally:
        conn.close()
    assert result.status == ERROR
    assert "not authorized" in (result.error or "").lower()


def test_connection_is_read_only(db_path):
    conn = connect(db_path)
    try:
        result = execute_on(conn, "UPDATE staff SET salary = 0", validate=False)
    finally:
        conn.close()
    assert result.status == ERROR


def test_database_is_unchanged_after_write_attempts(db_path):
    before = execute_sql(db_path, "SELECT sum(salary) FROM staff").rows
    execute_sql(db_path, "UPDATE staff SET salary = 0")
    execute_sql(db_path, "SELECT 1; DROP TABLE staff")
    after = execute_sql(db_path, "SELECT sum(salary) FROM staff").rows
    assert before == after


def test_timeout_aborts_a_runaway_query(db_path):
    result = execute_sql(db_path, _INFINITE, timeout_s=0.25)
    assert result.status == TIMEOUT
    assert result.elapsed_s < 5.0


def test_connection_survives_a_timeout(db_path):
    conn = connect(db_path)
    try:
        assert execute_on(conn, _INFINITE, timeout_s=0.25).status == TIMEOUT
        recovered = execute_on(conn, "SELECT count(*) FROM staff")
    finally:
        conn.close()
    assert recovered.status == OK
    assert recovered.rows == ((5,),)


def test_row_cap_truncates_instead_of_exploding(db_path):
    result = execute_sql(db_path, _COUNT_TO, max_rows=10)
    assert result.status == OK
    assert result.truncated is True
    assert result.n_rows == 10


def test_result_under_the_cap_is_not_marked_truncated(db_path):
    result = execute_sql(db_path, "SELECT staff_id FROM staff", max_rows=10)
    assert result.truncated is False


def test_missing_database_is_an_error_not_an_exception(tmp_path):
    result = execute_sql(tmp_path / "nope.sqlite", "SELECT 1")
    assert result.status == ERROR


def test_invalid_utf8_text_does_not_break_execution(bad_text_db_path):
    # The default text factory raises sqlite3.OperationalError here.
    with pytest.raises(sqlite3.OperationalError):
        raw = sqlite3.connect(bad_text_db_path)
        try:
            raw.execute("SELECT v FROM t").fetchall()
        finally:
            raw.close()

    result = execute_sql(bad_text_db_path, "SELECT v FROM t")
    assert result.status == OK
    assert result.n_rows == 1


def test_executor_caches_repeated_queries(db_path):
    with SqlExecutor() as executor:
        first = executor.execute(db_path, "SELECT count(*) FROM staff")
        second = executor.execute(db_path, "SELECT count(*) FROM staff")
        assert first.rows == second.rows == ((5,),)
        assert executor.stats.calls == 2
        assert executor.stats.cache_hits == 1


def test_executor_does_not_cache_timeouts(db_path):
    with SqlExecutor(timeout_s=0.25) as executor:
        executor.execute(db_path, _INFINITE)
        executor.execute(db_path, _INFINITE)
        assert executor.stats.cache_hits == 0
        assert executor.stats.timeouts == 2


def test_executor_counts_outcomes(db_path):
    with SqlExecutor() as executor:
        executor.execute(db_path, "SELECT 1")
        executor.execute(db_path, "DROP TABLE staff")
        executor.execute(db_path, "SELECT nosuchcolumn FROM staff")
        assert executor.stats.ok == 1
        assert executor.stats.rejected == 1
        assert executor.stats.errors == 1
