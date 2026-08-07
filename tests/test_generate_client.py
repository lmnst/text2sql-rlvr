"""Drive scripts/generate.py against a stub OpenAI-compatible server.

The GPU box is the only place a real vLLM server exists, so the request shape,
the retry path and the resume path are pinned here instead. A stub also makes
the Qwen3 thinking-mode flag testable, which is easy to break silently.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def generate_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("generate_script", SCRIPTS / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stub(ThreadingHTTPServer):
    """Records every request body; fails the first N requests when asked to."""

    allow_reuse_address = True
    requests: list[dict] = []
    fail_first = 0
    reply_sql = "SELECT count(*) FROM staff"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # keep pytest output clean
        pass

    def do_POST(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)

        if len(self.server.requests) <= self.server.fail_first:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"{}")
            return

        payload = {
            "choices": [
                {
                    "message": {"content": f"```sql\n{self.server.reply_sql}\n```"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
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
    httpd.fail_first = 0
    httpd.reply_sql = "SELECT count(*) FROM staff"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def base_url(server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/v1"


def run(generate_module, server, bird_root, out, *extra):
    return generate_module.main(
        [
            "--root", str(bird_root),
            "--split", "mini_dev",
            "--out", str(out),
            "--base-url", base_url(server),
            "--model", "stub/model",
            "--concurrency", "2",
            *extra,
        ]
    )


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_writes_one_record_per_example(generate_module, server, bird_root, tmp_path):
    out = tmp_path / "preds.jsonl"
    assert run(generate_module, server, bird_root, out) == 0

    records = read(out)
    assert len(records) == 5
    assert {r["question_id"] for r in records} == {0, 1, 2, 3, 4}
    assert all(r["sql"] == "SELECT count(*) FROM staff" for r in records)
    assert all(r["error"] is None for r in records)


def test_prompt_carries_schema_and_question(generate_module, server, bird_root, tmp_path):
    run(generate_module, server, bird_root, tmp_path / "preds.jsonl", "--limit", "1")

    messages = server.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "CREATE TABLE staff" in messages[1]["content"]
    assert "How many staff members are there?" in messages[1]["content"]


def test_thinking_is_disabled_by_default(generate_module, server, bird_root, tmp_path):
    run(generate_module, server, bird_root, tmp_path / "preds.jsonl", "--limit", "1")
    assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_flag_removes_the_override(generate_module, server, bird_root, tmp_path):
    run(generate_module, server, bird_root, tmp_path / "preds.jsonl", "--limit", "1", "--thinking")
    assert "chat_template_kwargs" not in server.requests[0]


def test_decoding_params_are_sent(generate_module, server, bird_root, tmp_path):
    run(generate_module, server, bird_root, tmp_path / "preds.jsonl", "--limit", "1",
        "--temperature", "0.7", "--max-tokens", "128", "--seed", "7")

    body = server.requests[0]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 128
    assert body["seed"] == 7


def test_meta_sidecar_records_what_produced_the_file(generate_module, server, bird_root, tmp_path):
    out = tmp_path / "preds.jsonl"
    run(generate_module, server, bird_root, out, "--limit", "2", "--seed", "3")

    meta = json.loads(out.with_suffix(".jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "stub/model"
    assert meta["decoding"]["seed"] == 3
    assert meta["decoding"]["thinking"] is False
    assert meta["prompt_config"]["schema_style"] == "ddl"
    assert meta["request_failures"] == 0


def test_transient_failures_are_retried(generate_module, server, bird_root, tmp_path):
    server.fail_first = 1
    out = tmp_path / "preds.jsonl"
    run(generate_module, server, bird_root, out, "--limit", "1", "--concurrency", "1")

    assert len(server.requests) == 2
    assert read(out)[0]["error"] is None


def test_exhausted_retries_are_recorded_not_raised(generate_module, server, bird_root, tmp_path):
    server.fail_first = 99
    out = tmp_path / "preds.jsonl"
    assert run(generate_module, server, bird_root, out, "--limit", "1",
               "--concurrency", "1", "--retries", "0") == 0

    record = read(out)[0]
    assert record["completion"] == ""
    assert "HTTPStatusError" in record["error"]

    meta = json.loads(out.with_suffix(".jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["request_failures"] == 1


def test_resume_skips_already_generated_ids(generate_module, server, bird_root, tmp_path):
    out = tmp_path / "preds.jsonl"
    run(generate_module, server, bird_root, out, "--limit", "2")
    assert len(server.requests) == 2

    run(generate_module, server, bird_root, out, "--resume")
    assert len(server.requests) == 5  # 2 from before, 3 new
    assert len(read(out)) == 5
