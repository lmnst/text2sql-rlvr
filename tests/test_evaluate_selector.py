"""Drive scripts/evaluate_selector.py against a stub server that answers table lists."""

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
def evaluate_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "evaluate_selector_script", SCRIPTS / "evaluate_selector.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stub(ThreadingHTTPServer):
    allow_reuse_address = True
    requests: list[dict] = []
    replies: dict[str, str] = {}  # question text -> raw completion


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)
        user = body["messages"][1]["content"]
        content = next(
            (reply for key, reply in self.server.replies.items() if key in user), "[]"
        )
        payload = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
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
    httpd.replies = {}
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _row(qid: int, db: str, question: str, gold: list[str], tables: list[str]) -> dict:
    return {
        "question_id": qid,
        "db_id": db,
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": f"schema... Question: {question}"},
        ],
        "gold_tables": gold,
        "all_tables": tables,
        "n_tables_total": len(tables),
        "linked_tables": tables,  # the lexical linker kept everything
        "lexical_worst_rank": 1,
        "hard_reasons": [],
    }


@pytest.fixture
def eval_file(tmp_path):
    rows = [
        _row(0, "company", "How many staff?", ["staff"], ["dept", "staff"]),
        _row(1, "company", "Which departments?", ["dept"], ["dept", "staff"]),
        _row(2, "shop", "Orders per item?", ["orders", "items"], ["items", "orders", "log"]),
    ]
    path = tmp_path / "heldout.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def run(module, server, eval_file, out, *extra):
    host, port = server.server_address[:2]
    return module.main([
        "--eval", str(eval_file), "--out", str(out),
        "--base-url", f"http://{host}:{port}/v1", "--model", "stub",
        "--concurrency", "1", "--no-ledger", "--expand", "none", *extra,
    ])


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_scores_model_against_gold_and_lexical_baseline(
    evaluate_module, server, eval_file, tmp_path
):
    server.replies = {
        "How many staff?": '["staff"]',
        "Which departments?": 'Sure! ["dept", "staff"]',
        "Orders per item?": '["orders"]',
    }
    out = tmp_path / "preds.jsonl"
    assert run(evaluate_module, server, eval_file, out) == 0

    by_id = {r["question_id"]: r for r in read(out)}
    assert by_id[0]["predicted_tables"] == ["staff"] and by_id[0]["all_gold_retained"]
    assert by_id[1]["predicted_tables"] == ["dept", "staff"] and by_id[1]["precision"] == 0.5
    assert by_id[2]["predicted_tables"] == ["orders"] and not by_id[2]["all_gold_retained"]
    assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}

    summary = json.loads(out.with_suffix(".jsonl.summary.json").read_text(encoding="utf-8"))
    m = summary["metrics"]
    assert m["all_gold_retained_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mean_precision"] == pytest.approx((1 + 0.5 + 1) / 3, abs=1e-4)
    assert m["baseline_all_gold_retained_rate"] == 1.0
    assert m["baseline_mean_precision"] == pytest.approx((0.5 + 0.5 + 2 / 3) / 3, abs=1e-4)
    assert summary["split"] == "selector-heldout"
    assert set(summary["report"]["per_db"]) == {"company", "shop"}


def test_empty_answers_are_counted_and_resume_skips_done(
    evaluate_module, server, eval_file, tmp_path
):
    out = tmp_path / "preds.jsonl"
    run(evaluate_module, server, eval_file, out, "--limit", "2")
    assert len(server.requests) == 2
    summary = json.loads(out.with_suffix(".jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["n_empty_predictions"] == 2
    assert summary["metrics"]["all_gold_retained_rate"] == 0.0

    run(evaluate_module, server, eval_file, out, "--resume")
    assert len(server.requests) == 3
    assert len(read(out)) == 3


def test_fk_expansion_recovers_bridge_table_and_is_written_for_generate(
    evaluate_module, server, bird_root, tmp_path
):
    rows = [
        {
            **_row(0, "company", "Who works in Research?", ["dept", "staff"], ["dept", "staff"]),
            "question": "Who works in Research?",
            "evidence": "",
        },
        {
            **_row(1, "company", "How many staff?", ["staff"], ["dept", "staff"]),
            "question": "How many staff?",
            "evidence": "",
        },
    ]
    eval_file = tmp_path / "heldout.jsonl"
    eval_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    server.replies = {"Who works in Research?": '["staff"]', "How many staff?": '["staff"]'}

    out = tmp_path / "preds.jsonl"
    host, port = server.server_address[:2]
    assert evaluate_module.main([
        "--eval", str(eval_file), "--out", str(out),
        "--base-url", f"http://{host}:{port}/v1", "--model", "stub",
        "--concurrency", "1", "--no-ledger",
        "--root", str(bird_root), "--split", "mini_dev", "--expand", "fk", "--fk-hops", "1",
    ]) == 0

    by_id = {r["question_id"]: r for r in read(out)}
    assert by_id[0]["predicted_tables"] == ["staff"] and not by_id[0]["all_gold_retained"]
    assert by_id[0]["expanded_tables"] == ["dept", "staff"]
    assert by_id[0]["expanded_all_gold_retained"]
    assert by_id[1]["expanded_tables"] == ["dept", "staff"]  # extra table, still all gold

    summary = json.loads(out.with_suffix(".jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["all_gold_retained_rate"] == 0.5
    assert summary["metrics"]["expanded_all_gold_retained_rate"] == 1.0
    assert summary["metrics"]["expanded_mean_selected_tables"] == 2.0
    assert summary["expansion"] == {"mode": "fk", "fk_hops": 1, "lex_top_k": 0}

    # Rescoring a finished file sends no request and can change the expansion.
    n_before = len(server.requests)
    assert evaluate_module.main([
        "--eval", str(eval_file), "--out", str(out),
        "--base-url", f"http://{host}:{port}/v1", "--model", "stub",
        "--no-ledger", "--resume", "--expand", "none",
    ]) == 0
    assert len(server.requests) == n_before
    assert read(out)[0]["expanded_tables"] == ["staff"]


def test_sampled_answers_are_unioned_and_failed_requests_are_retried_on_resume(
    evaluate_module, server, eval_file, tmp_path
):
    # Each choice names a different table; the union must contain both.
    class _MultiHandler(_Handler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            self.server.requests.append(body)
            n = body.get("n", 1)
            choices = [
                {"message": {"content": '["staff"]' if k == 0 else '["dept"]'},
                 "finish_reason": "stop"}
                for k in range(n)
            ]
            encoded = json.dumps({"choices": choices, "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server.RequestHandlerClass = _MultiHandler
    out = tmp_path / "preds.jsonl"
    run(evaluate_module, server, eval_file, out, "--limit", "1",
        "--n-samples", "3", "--temperature", "0.8")
    record = read(out)[0]
    assert server.requests[0]["n"] == 3
    assert record["n_samples"] == 3
    assert record["samples"] == [["staff"], ["dept"], ["dept"]]
    assert record["predicted_tables"] == ["dept", "staff"]

    with pytest.raises(SystemExit):
        run(evaluate_module, server, eval_file, out, "--n-samples", "2")

    # A record whose request failed is re-queried on --resume.
    failed = dict(record, error="HTTPStatusError: 503", predicted_tables=[], samples=[])
    out.write_text(json.dumps(failed) + "\n", encoding="utf-8")
    n_before = len(server.requests)
    run(evaluate_module, server, eval_file, out, "--limit", "1", "--resume")
    assert len(server.requests) == n_before + 1
    assert read(out)[0]["error"] is None
