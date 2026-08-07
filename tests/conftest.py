"""Shared fixtures. Every test builds its own SQLite file and its own BIRD-shaped
directory, so the suite runs on a clean checkout with no download and no GPU."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

_SCHEMA = """
CREATE TABLE dept (
    dept_id   INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    `head count` INTEGER
);

CREATE TABLE staff (
    staff_id  INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    dept_id   INTEGER,
    salary    REAL,
    note      TEXT,
    FOREIGN KEY (dept_id) REFERENCES dept(dept_id)
);
"""

_ROWS_DEPT = [
    (1, "Research", 3),
    (2, "Sales", 2),
    (3, "Empty", 0),
]

_ROWS_STAFF = [
    (1, "Ada", 1, 100.0, "lead"),
    (2, "Bob", 1, 90.0, None),
    (3, "Cy", 1, 90.0, ""),
    (4, "Dee", 2, 80.0, None),
    (5, "Eve", 2, 80.0, "part-time"),
]

_STAFF_DESCRIPTIONS = (
    "original_column_name,column_name,column_description,value_description\n"
    "staff_id,Staff ID,unique identifier of the staff member,\n"
    "salary,Salary,monthly salary in EUR,positive real number\n"
)

#: Mirrors the shape of BIRD's question files, including the fields we rely on.
_QUESTIONS = [
    {
        "question_id": 0,
        "db_id": "company",
        "question": "How many staff members are there?",
        "evidence": "",
        "SQL": "SELECT count(*) FROM staff",
        "difficulty": "simple",
    },
    {
        "question_id": 1,
        "db_id": "company",
        "question": "List the salary of every staff member.",
        "evidence": "",
        "SQL": "SELECT salary FROM staff",
        "difficulty": "simple",
    },
    {
        "question_id": 2,
        "db_id": "company",
        "question": "Who works in the Research department?",
        "evidence": "Research means dept_id = 1",
        "SQL": "SELECT name FROM staff WHERE dept_id = 1",
        "difficulty": "moderate",
    },
    {
        "question_id": 3,
        "db_id": "company",
        "question": "What are the department names?",
        "evidence": "",
        "SQL": "SELECT name FROM dept",
        "difficulty": "moderate",
    },
    {
        "question_id": 4,
        "db_id": "company",
        "question": "How many departments are there?",
        "evidence": "",
        "SQL": "SELECT count(*) FROM dept",
        "difficulty": "challenging",
    },
]


def _build_company_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.executemany("INSERT INTO dept VALUES (?, ?, ?)", _ROWS_DEPT)
        conn.executemany("INSERT INTO staff VALUES (?, ?, ?, ?, ?)", _ROWS_STAFF)
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small database exercising NULLs, duplicate values and quoted identifiers."""
    return _build_company_db(tmp_path_factory.mktemp("dbs") / "company.sqlite")


@pytest.fixture(scope="session")
def bad_text_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A database with a TEXT column holding bytes that are not valid UTF-8.

    Several real BIRD databases do this, and the default sqlite3 text factory
    raises on those rows -- which would turn a correct query into an execution
    error and silently depress every metric.
    """
    path = tmp_path_factory.mktemp("dbs") / "badtext.sqlite"
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE t (id INTEGER, v TEXT)")
        conn.execute("INSERT INTO t VALUES (1, CAST(X'FFFE' AS TEXT))")
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture(scope="session")
def bird_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory laid out the way a real BIRD mini-dev download is."""
    root = tmp_path_factory.mktemp("bird")
    release = root / "MINIDEV"
    db_dir = release / "dev_databases" / "company"
    _build_company_db(db_dir / "company.sqlite")

    descriptions = db_dir / "database_description"
    descriptions.mkdir(parents=True, exist_ok=True)
    (descriptions / "staff.csv").write_text(_STAFF_DESCRIPTIONS, encoding="utf-8")

    (release / "mini_dev_sqlite.json").write_text(
        json.dumps(_QUESTIONS, indent=2), encoding="utf-8"
    )
    return root
