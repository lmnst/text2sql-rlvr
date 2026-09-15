"""The execute-observe-revise loop: protocol parsing, observations, stopping."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from text2sql_rlvr.agent import (
    DESCRIBE,
    EXECUTE,
    FINAL,
    NONE,
    STOP_BUDGET,
    STOP_FINAL,
    STOP_NO_SQL,
    STOP_REQUEST_ERROR,
    AgentConfig,
    build_agent_messages,
    parse_action,
    run_episode,
)
from text2sql_rlvr.data import discover_split, format_schema, load_schema
from text2sql_rlvr.rewards.sandbox import ERROR, OK, SqlExecutor

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_parse_action_prefers_final_then_query_then_describe():
    assert parse_action("FINAL\n```sql\nSELECT 1\n```") == parse_action("final ```sql\nSELECT 1```")
    assert parse_action("FINAL\n```sql\nSELECT 1\n```").kind == FINAL
    assert parse_action("Let me check.\n```sql\nSELECT count(*) FROM staff\n```").kind == EXECUTE
    assert parse_action("```sql\nSELECT 1\n```").sql == "SELECT 1"
    assert parse_action("DESCRIBE staff").table == "staff"
    assert parse_action("describe `head count table`;").table == "head count table"
    assert parse_action("<think>FINAL?</think>DESCRIBE dept").kind == DESCRIBE
    assert parse_action("I am not sure what to do.").kind == NONE
    assert parse_action("FINAL").kind == NONE  # FINAL without a query is no action
    # a bare SELECT without a fence still counts as a query
    assert parse_action("SELECT name FROM dept").kind == EXECUTE


@pytest.fixture
def company(bird_root):
    split = discover_split(bird_root, "mini_dev")
    schema = load_schema(split.db_path("company"), db_id="company")
    return split, schema


def scripted(replies):
    """A chat function that returns the scripted replies in order."""
    calls = []

    def chat(messages):
        calls.append([dict(m) for m in messages])
        return replies[len(calls) - 1]

    chat.calls = calls
    return chat


def test_episode_describes_recovers_from_an_error_and_commits(company):
    split, schema = company
    example = split.load()[2]  # Who works in the Research department? (dept_id = 1)
    chat = scripted([
        "DESCRIBE dept",
        "```sql\nSELECT nam FROM staff WHERE dept_id = 1\n```",       # bad column
        "```sql\nSELECT name FROM staff WHERE dept_id = 1\n```",
        "FINAL\n```sql\nSELECT name FROM staff WHERE dept_id = 1\n```",
    ])
    with SqlExecutor() as executor:
        episode = run_episode(
            example, schema, format_schema(schema), split.db_path("company"), executor, chat,
            other_tables=(), config=AgentConfig(max_turns=4),
        )

    assert episode.stop_reason == STOP_FINAL
    assert episode.final_sql == "SELECT name FROM staff WHERE dept_id = 1"
    assert [s.action for s in episode.steps] == [DESCRIBE, EXECUTE, EXECUTE, FINAL]

    describe_obs = episode.steps[0].observation
    assert "CREATE TABLE dept" in describe_obs and "Research" in describe_obs
    assert episode.steps[1].exec_status == ERROR
    assert episode.steps[1].observation.startswith("Query error:")
    assert episode.steps[2].exec_status == OK and episode.steps[2].n_rows == 3
    assert episode.steps[2].observation.startswith("Query result: 3 row(s)")
    assert "Ada" in episode.steps[2].observation
    assert episode.first_exec_status == ERROR and episode.final_exec_status == OK

    # Observations were fed back as user turns, in order, and the record scores.
    roles = [m["role"] for m in episode.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user",
                     "assistant", "user", "assistant"]
    assert episode.messages[3]["content"] == describe_obs
    record = episode.as_dict()
    assert record["sql"] == episode.final_sql
    assert record["n_executions"] == 2 and record["n_describes"] == 1
    assert record["n_exec_errors"] == 1


def test_budget_exhausted_falls_back_to_last_successful_query(company):
    split, schema = company
    example = split.load()[0]
    chat = scripted([
        "```sql\nSELECT count(*) FROM staff\n```",
        "```sql\nSELECT count(*) FROM nowhere\n```",   # last reply is an error
    ])
    with SqlExecutor() as executor:
        episode = run_episode(
            example, schema, format_schema(schema), split.db_path("company"), executor, chat,
            config=AgentConfig(max_turns=2),
        )
    assert episode.stop_reason == STOP_BUDGET
    assert episode.final_sql == "SELECT count(*) FROM staff"
    assert episode.final_exec_status == OK
    # no observation is appended after the last allowed turn
    assert episode.messages[-1]["role"] == "assistant"


def test_no_action_replies_are_nudged_and_end_without_sql(company):
    split, schema = company
    example = split.load()[0]
    chat = scripted(["I need to think.", "Still thinking."])
    with SqlExecutor() as executor:
        episode = run_episode(
            example, schema, format_schema(schema), split.db_path("company"), executor, chat,
            config=AgentConfig(max_turns=2),
        )
    assert episode.stop_reason == STOP_NO_SQL
    assert episode.final_sql == "" and episode.final_exec_status is None
    assert "no action" in episode.steps[0].observation


def test_unknown_table_and_request_failure_are_reported(company):
    split, schema = company
    example = split.load()[0]

    def flaky(messages):
        if len(messages) == 2:
            return "DESCRIBE ghost"
        raise RuntimeError("connection refused")

    with SqlExecutor() as executor:
        episode = run_episode(
            example, schema, format_schema(schema), split.db_path("company"), executor, flaky,
            config=AgentConfig(max_turns=3),
        )
    assert "No table named 'ghost'" in episode.steps[0].observation
    assert "dept, staff" in episode.steps[0].observation
    assert episode.stop_reason == STOP_REQUEST_ERROR
    assert "connection refused" in episode.error


def test_prompt_lists_hidden_tables_for_describe(company):
    split, schema = company
    example = split.load()[2]
    messages = build_agent_messages(example, "CREATE TABLE staff (...)", other_tables=("dept",))
    user = messages[1]["content"]
    assert "not shown above" in user and "dept" in user
    assert example.evidence in user and example.question in user
    assert user.rstrip().endswith("no explanation.")
    assert "not shown above" not in build_agent_messages(example, "x")[1]["content"]


# ------------------------------------------------------------------ driver


@pytest.fixture(scope="module")
def run_agent_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("run_agent_script", SCRIPTS / "run_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stub(ThreadingHTTPServer):
    allow_reuse_address = True
    requests: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)
        n_assistant = sum(1 for m in body["messages"] if m["role"] == "assistant")
        # turn 1: run a query; turn 2: commit to it
        content = ("```sql\nSELECT count(*) FROM staff\n```" if n_assistant == 0
                   else "FINAL\n```sql\nSELECT count(*) FROM staff\n```")
        payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def server():
    httpd = _Stub(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_driver_writes_scorable_predictions_and_meta(run_agent_module, server, bird_root,
                                                     tmp_path):
    host, port = server.server_address[:2]
    selected = tmp_path / "selector.jsonl"
    selected.write_text(
        json.dumps({"question_id": 0, "expanded_tables": ["staff"]}) + "\n", encoding="utf-8"
    )
    out = tmp_path / "agent.jsonl"
    assert run_agent_module.main([
        "--root", str(bird_root), "--split", "mini_dev", "--out", str(out),
        "--base-url", f"http://{host}:{port}/v1", "--model", "stub",
        "--concurrency", "1", "--limit", "2", "--max-turns", "3",
        "--selected-tables", str(selected),
    ]) == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    by_id = {r["question_id"]: r for r in records}
    assert by_id[0]["sql"] == "SELECT count(*) FROM staff"
    assert by_id[0]["stop_reason"] == STOP_FINAL and by_id[0]["n_turns"] == 2
    assert by_id[0]["schema_mode"] == "predicted" and by_id[0]["selected_tables"] == ["staff"]
    assert by_id[1]["schema_mode"] == "full"  # no prediction -> fallback

    # The hidden table was offered for DESCRIBE, and the observation went back as a user turn.
    first_prompt = server.requests[0]["messages"][1]["content"]
    assert "not shown above" in first_prompt and "dept" in first_prompt
    assert server.requests[1]["messages"][3]["role"] == "user"
    assert server.requests[1]["messages"][3]["content"].startswith("Query result: 1 row(s)")
    assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}

    meta = json.loads(out.with_suffix(".jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["summary"]["stop_reasons"] == {STOP_FINAL: 2}
    assert meta["summary"]["final_exec_status"] == {OK: 2}
    assert meta["schema_selection"]["mode"] == "predicted"
    assert meta["agent"]["max_turns"] == 3
