"""The ledger has to fill in provenance by itself, or it will not get filled in."""

from __future__ import annotations

import pytest

from text2sql_rlvr.ledger import append_run, file_sha256, read_runs

MINIMAL = {
    "stage": "baseline",
    "split": "mini_dev",
    "n_samples": 5,
    "metrics": {"official_ex": 40.0},
}


def test_provenance_is_filled_in_automatically(tmp_path):
    entry = append_run(dict(MINIMAL), path=tmp_path / "runs.jsonl")
    assert entry["run_id"]
    assert entry["timestamp"].endswith("+00:00")
    assert "git_sha" in entry
    assert isinstance(entry["git_dirty"], bool)
    assert entry["hardware"]


def test_records_round_trip(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_run(dict(MINIMAL), path=path)
    append_run({**MINIMAL, "stage": "sft"}, path=path)
    runs = read_runs(path)
    assert [r["stage"] for r in runs] == ["baseline", "sft"]
    assert len({r["run_id"] for r in runs}) == 2


def test_reading_an_absent_ledger_is_empty_not_an_error(tmp_path):
    assert read_runs(tmp_path / "nothing.jsonl") == []


@pytest.mark.parametrize("field", ["stage", "split", "n_samples", "metrics"])
def test_required_fields_are_enforced(tmp_path, field):
    record = dict(MINIMAL)
    record.pop(field)
    with pytest.raises(ValueError, match=field):
        append_run(record, path=tmp_path / "runs.jsonl")


def test_unknown_stage_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="stage"):
        append_run({**MINIMAL, "stage": "final-results"}, path=tmp_path / "runs.jsonl")


def test_config_hash_is_recorded(tmp_path):
    config = tmp_path / "grpo.yaml"
    config.write_text("lr: 1e-6\n", encoding="utf-8")
    entry = append_run({**MINIMAL, "config_path": str(config)}, path=tmp_path / "runs.jsonl")
    assert entry["config_sha256"] == file_sha256(config)
    assert len(entry["config_sha256"]) == 64


def test_absent_config_hashes_to_empty(tmp_path):
    entry = append_run({**MINIMAL, "config_path": "nope.yaml"}, path=tmp_path / "runs.jsonl")
    assert entry["config_sha256"] == ""
