"""Regression: baseline question 7093 lost its outer query before evaluation."""

import pytest

from text2sql_rlvr.sql import extract_sql, validate_read_only


@pytest.mark.parametrize("sql", [
    "SELECT e.JobTitle\nFROM Employee e\nWHERE e.BusinessEntityID IN (\n"
    "    SELECT BusinessEntityID\n    FROM EmployeeDepartmentHistory\n"
    "    WHERE DepartmentID = 12\n    ORDER BY StartDate DESC\n    LIMIT 1\n)",
    "WITH c AS (\n SELECT 1 AS x\n)\nSELECT x FROM c",
    "SELECT 1\nUNION ALL\nSELECT 2",
])
def test_bare_query_keeps_outer_select_cte_and_union(sql):
    assert extract_sql(sql + ";") == sql
    assert extract_sql("The query is:\n" + sql + ";") == sql


def test_bare_multiple_statements_are_not_hidden_by_extraction():
    extracted = extract_sql("SELECT 1;\nSELECT 2;")
    assert not validate_read_only(extracted).ok
    assert "found 2" in validate_read_only(extracted).reason
