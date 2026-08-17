"""Reward entry point for verl.

Point verl at this file:

    custom_reward_function.path=/root/autodl-tmp/verl_reward.py
    custom_reward_function.name=compute_score

Configuration is read from the environment rather than baked in, so the same
file serves the honest run and the deliberate reward-hacking run without an edit
that would have to be remembered and undone:

    TEXT2SQL_DB_ROOT          required, the *_databases directory
    TEXT2SQL_REWARD_FORMAT    format bonus, default 0
    TEXT2SQL_REWARD_EXEC      bonus for SQL that merely runs, default 0
    TEXT2SQL_REWARD_OFFICIAL  "1" to score with BIRD's set comparison (default)
    TEXT2SQL_REWARD_TIMEOUT   seconds, default 10
    TEXT2SQL_ROLLOUT_LOG      jsonl path; every rollout's breakdown is appended

The rollout log is the important one. It makes the hacking measurement
independent of what verl does or does not do with the return value: whatever
version is installed, the per-rollout record lands in a file we control and can
analyse afterwards.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path


def _locate_package() -> Path:
    """Find `text2sql_rlvr` whether this file sits in the repo or was copied out.

    verl is pointed at this file by absolute path, so it commonly ends up on its
    own in a working directory with no repo around it. Guessing one layout and
    failing with a bare ImportError wastes a GPU-hour of confusion.
    """
    candidates = []
    override = os.environ.get("TEXT2SQL_SRC")
    if override:
        candidates.append(Path(override))
    here = Path(__file__).resolve()
    candidates += [
        here.parents[1] / "src",  # in the repo: <repo>/scripts/verl_reward.py
        here.parent / "src",  # copied next to a src/ directory
        here.parent,  # copied next to the package itself
    ]
    for candidate in candidates:
        if (candidate / "text2sql_rlvr" / "__init__.py").is_file():
            return candidate
    tried = "\n  ".join(str(c) for c in candidates)
    raise ModuleNotFoundError(
        "cannot find the text2sql_rlvr package. Copy the repo's src/ directory to "
        "the training box, or set TEXT2SQL_SRC to wherever it lives.\nLooked in:\n  "
        + tried
    )


sys.path.insert(0, str(_locate_package()))

from text2sql_rlvr.rewards.reward import (  # noqa: E402
    RewardConfig,
    RewardStats,
    compute_reward,
)
from text2sql_rlvr.rewards.sandbox import SqlExecutor  # noqa: E402


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


_CONFIG = RewardConfig(
    format_bonus=float(os.environ.get("TEXT2SQL_REWARD_FORMAT", "0")),
    execution_bonus=float(os.environ.get("TEXT2SQL_REWARD_EXEC", "0")),
    use_official=_flag("TEXT2SQL_REWARD_OFFICIAL", "1"),
    order_policy=os.environ.get("TEXT2SQL_ORDER_POLICY", "ignore"),
)

# One executor per process: connections are per-thread and results are cached,
# which is what keeps reward computation from becoming the training bottleneck.
_EXECUTOR = SqlExecutor(timeout_s=float(os.environ.get("TEXT2SQL_REWARD_TIMEOUT", "10")))

_STATS = RewardStats()
_LOG_PATH = os.environ.get("TEXT2SQL_ROLLOUT_LOG", "")
_LOG_LOCK = threading.Lock()


def _db_path(extra_info: dict | None) -> Path:
    root = os.environ.get("TEXT2SQL_DB_ROOT")
    if not root:
        raise RuntimeError("TEXT2SQL_DB_ROOT is not set; the reward cannot find the databases")
    db_id = (extra_info or {}).get("db_id")
    if not db_id:
        raise RuntimeError(f"extra_info has no db_id: {extra_info!r}")
    return Path(root) / db_id / f"{db_id}.sqlite"


def _record(payload: dict) -> None:
    """Append one rollout record to the log immediately.

    Each rollout is written as its own line rather than buffered, so a short
    smoke run (a handful of rollouts) still produces a readable log -- buffering
    until N records silently dropped the entire smoke log.
    """
    if not _LOG_PATH:
        return
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_kwargs: object,
) -> float:
    """Score one rollout. Returns a plain float, which every verl version accepts."""
    try:
        breakdown = compute_reward(
            solution_str,
            ground_truth,
            _db_path(extra_info),
            executor=_EXECUTOR,
            config=_CONFIG,
        )
    except Exception as exc:  # noqa: BLE001 - a broken reward must not kill training
        _record({"error": f"{type(exc).__name__}: {exc}", "extra_info": extra_info})
        return 0.0

    with _LOG_LOCK:
        _STATS.update(breakdown)

    payload = breakdown.as_dict()
    payload["question_id"] = (extra_info or {}).get("question_id")
    payload["db_id"] = (extra_info or {}).get("db_id")
    payload["split"] = (extra_info or {}).get("split")
    payload["sample_index"] = (extra_info or {}).get("index")
    payload["data_source"] = data_source
    _record(payload)

    return float(breakdown.reward)


def flush() -> dict[str, float]:
    """Return the running reward statistics (logging is no longer buffered)."""
    with _LOG_LOCK:
        return _STATS.as_dict()


if __name__ == "__main__":
    # Self-check: run this on the training box before starting a real run.
    if len(sys.argv) != 4:
        print('usage: python verl_reward.py <db_root> <db_id> "<gold sql>"')
        print("example:")
        print("  python verl_reward.py /root/autodl-tmp/bird/train/train_databases \\")
        print('      california_schools "SELECT COUNT(*) FROM schools"')
        sys.exit(2)
    root, db_id, gold = sys.argv[1], sys.argv[2], sys.argv[3]
    os.environ["TEXT2SQL_DB_ROOT"] = root
    print("config:", _CONFIG.as_dict())
    for label, completion in (
        ("perfect", f"```sql\n{gold}\n```"),
        ("degenerate SELECT 1", "```sql\nSELECT 1\n```"),
        ("not sql", "I cannot answer that."),
        ("write attempt", "```sql\nDROP TABLE x\n```"),
    ):
        score = compute_score("bird", completion, gold, {"db_id": db_id, "question_id": -1})
        print(f"  {label:<22} reward={score}")
    print("\nexpected: perfect > 0, everything else 0 under the default config")
