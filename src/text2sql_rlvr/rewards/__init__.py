"""SQL execution and verifiable reward functions."""

from text2sql_rlvr.rewards.canonical import canon_row, canon_rows, canon_value
from text2sql_rlvr.rewards.compare import Comparison, compare, compare_official
from text2sql_rlvr.rewards.reward import (
    RewardBreakdown,
    RewardConfig,
    RewardStats,
    compute_reward,
)
from text2sql_rlvr.rewards.sandbox import (
    ERROR,
    OK,
    REJECTED,
    TIMEOUT,
    ExecResult,
    SqlExecutor,
    connect,
    execute_on,
    execute_sql,
)

__all__ = [
    "ERROR",
    "OK",
    "REJECTED",
    "TIMEOUT",
    "Comparison",
    "RewardBreakdown",
    "RewardConfig",
    "RewardStats",
    "ExecResult",
    "SqlExecutor",
    "canon_row",
    "canon_rows",
    "canon_value",
    "compare",
    "compare_official",
    "compute_reward",
    "connect",
    "execute_on",
    "execute_sql",
]
