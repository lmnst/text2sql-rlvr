"""The reward signal for GRPO.

Design follows from what was measured, not from what sounded good:

**The strict verifier is the reward, not BIRD's official one.** Milestone 9 found
25 of 500 SFT answers that the official set comparison credits and the strict one
rejects, and every one is the same shape: gold says ``SELECT DISTINCT`` and the
model omits it, returning 933 rows where 21 were wanted. Under the official
metric those score identically, so a policy trained on it faces no pressure to
ever deduplicate. Reinforcement learning amplifies whatever it is not pushed
against.

**Partial credit is off by default.** ``execution_bonus`` pays out for SQL that
merely runs, which ``SELECT 1`` satisfies without reading the database. That is
not an oversight to be avoided -- it is the experiment: turn it on, measure how
fast the policy finds it, turn it off, measure again. Both settings are
first-class and both are recorded.

Every call reports the official verdict alongside the strict one and flags the
known degenerate shapes, so the hacking rate is *measured during training*
rather than reconstructed from checkpoints afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from text2sql_rlvr.rewards.compare import DEFAULT_ORDER_POLICY, compare
from text2sql_rlvr.rewards.sandbox import OK, REJECTED, ExecResult, SqlExecutor
from text2sql_rlvr.sql import extract_sql, has_from_clause, validate_read_only


@dataclass(frozen=True)
class RewardConfig:
    """What the policy is paid for.

    Defaults are the honest setting: correctness only, judged strictly.
    """

    correct: float = 1.0
    #: Paid when the completion contains one parsable read-only statement.
    #: Cheap to satisfy without answering; keep at 0 unless studying that.
    format_bonus: float = 0.0
    #: Paid when the SQL executes without error. ``SELECT 1`` collects this.
    #: This is the deliberate hack lever.
    execution_bonus: float = 0.0
    #: Score against BIRD's set comparison instead of the strict verifier.
    #: Only for the ablation that shows why we do not.
    use_official: bool = False
    order_policy: str = DEFAULT_ORDER_POLICY

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def max_reward(self) -> float:
        return self.correct + self.format_bonus + self.execution_bonus


@dataclass(frozen=True)
class RewardBreakdown:
    """The score, and everything needed to explain it."""

    reward: float
    correct: bool
    official: bool
    parsed: bool
    executed: bool
    pred_status: str
    gold_status: str
    reason: str | None
    pred_n_rows: int
    gold_n_rows: int
    #: Degenerate shapes worth counting while training runs.
    no_from_clause: bool
    empty_result: bool
    sql: str

    @property
    def credited_without_answering(self) -> bool:
        """Reward collected by a completion the strict verifier rejects."""
        return self.reward > 0.0 and not self.correct

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _zero(sql: str, reason: str, parsed: bool = False) -> RewardBreakdown:
    return RewardBreakdown(
        reward=0.0,
        correct=False,
        official=False,
        parsed=parsed,
        executed=False,
        pred_status=REJECTED,
        gold_status=REJECTED,
        reason=reason,
        pred_n_rows=0,
        gold_n_rows=0,
        no_from_clause=not has_from_clause(sql) if sql else True,
        empty_result=False,
        sql=sql,
    )


def compute_reward(
    completion: str,
    gold_sql: str,
    db_path: str | Path,
    *,
    executor: SqlExecutor,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score one rollout."""
    config = config or RewardConfig()

    sql = extract_sql(completion or "")
    if not sql:
        return _zero("", "no_sql_in_completion")

    validation = validate_read_only(sql)
    if not validation.ok:
        return _zero(sql, validation.reason or "rejected")

    sql = validation.statement
    reward = config.format_bonus

    pred = executor.execute(db_path, sql)
    gold = (
        executor.execute(db_path, gold_sql)
        if gold_sql
        else ExecResult(REJECTED, error="no gold SQL")
    )

    if pred.status == OK:
        reward += config.execution_bonus

    verdict = compare(pred, gold, gold_sql=gold_sql, order_policy=config.order_policy)
    accepted = verdict.official if config.use_official else verdict.strict
    if accepted:
        reward += config.correct

    return RewardBreakdown(
        reward=reward,
        correct=verdict.strict,
        official=verdict.official,
        parsed=True,
        executed=pred.status == OK,
        pred_status=pred.status,
        gold_status=gold.status,
        reason=verdict.reason,
        pred_n_rows=pred.n_rows,
        gold_n_rows=gold.n_rows,
        no_from_clause=not has_from_clause(sql),
        empty_result=pred.status == OK and pred.n_rows == 0,
        sql=sql,
    )


@dataclass
class RewardStats:
    """Running counts, for watching the policy during training."""

    n: int = 0
    total_reward: float = 0.0
    n_correct: int = 0
    n_official: int = 0
    n_parsed: int = 0
    n_executed: int = 0
    n_no_from: int = 0
    n_empty_result: int = 0
    n_credited_without_answering: int = 0

    def update(self, breakdown: RewardBreakdown) -> None:
        self.n += 1
        self.total_reward += breakdown.reward
        self.n_correct += breakdown.correct
        self.n_official += breakdown.official
        self.n_parsed += breakdown.parsed
        self.n_executed += breakdown.executed
        self.n_no_from += breakdown.no_from_clause
        self.n_empty_result += breakdown.empty_result
        self.n_credited_without_answering += breakdown.credited_without_answering

    def as_dict(self) -> dict[str, float]:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "mean_reward": round(self.total_reward / self.n, 4),
            "strict_acc": round(self.n_correct / self.n, 4),
            "official_acc": round(self.n_official / self.n, 4),
            # The gap the whole project is about. If this grows during training,
            # the policy is learning to satisfy the loose metric.
            "official_minus_strict": round((self.n_official - self.n_correct) / self.n, 4),
            "parse_rate": round(self.n_parsed / self.n, 4),
            "exec_rate": round(self.n_executed / self.n, 4),
            "no_from_rate": round(self.n_no_from / self.n, 4),
            "empty_result_rate": round(self.n_empty_result / self.n, 4),
            "hack_rate": round(self.n_credited_without_answering / self.n, 4),
        }
