# Text2SQL-RLVR

Execution-grounded reinforcement learning for Text-to-SQL with Qwen3, verl, and vLLM.

## Goal

Build a reproducible SFT-to-GRPO pipeline on BIRD in which generated SQL is executed against
read-only databases and scored with verifiable rewards.

## Status

The data, evaluation, read-only execution, verifier, baseline, and LoRA SFT stages are implemented
and recorded. The tracked experiment ledger currently reports:

- Qwen3-1.7B baseline on BIRD Mini-Dev (500 examples): 19.80% official EX, 17.40% strict EX.
- One-epoch LoRA SFT: 34.60% official EX, 30.00% strict EX.

The current gate is the first GRPO smoke test on AutoDL. It has not completed, so there are no
GRPO or reward-hacking results. The recent commits document environment failures and candidate
configuration fixes, not a successful training run.

Start with [HANDOFF.md](HANDOFF.md) for the exact continuation point and
[docs/PROGRESS.md](docs/PROGRESS.md) for the evidence-backed milestone history. Project rules and
data-split discipline are in [AGENTS.md](AGENTS.md).

## Two verifiers, deliberately

Every evaluation reports two numbers over the same run:

- **`official_ex`** reproduces BIRD's Execution Accuracy exactly: `set(pred) == set(gold)` over
  the raw result tuples. This is the number that goes in any table claiming to report EX, and
  the wrapper does not "improve" it.
- **`strict_ex`** is this project's verifier: same column count, same rows *with multiplicity*,
  numbers compared after canonicalisation.

The official set comparison can be satisfied without answering the question — dropping duplicate
rows with `DISTINCT`, or returning an empty result set when the gold answer happens to be empty.
Those are exactly the behaviours an RL policy is free to discover. Reporting both numbers costs
nothing (gold and prediction are already executed) and the gap between them is the measurement
that the reward-hacking analysis is built on, not a nuisance to be hidden.

## Getting started

No GPU and no dataset needed for the tests:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Then fetch BIRD as described in [docs/data.md](docs/data.md) and check it:

```bash
python scripts/prepare_bird.py --root data/bird --split mini_dev --check-gold
```

Generation talks to any OpenAI-compatible endpoint, so the GPU box only runs
`vllm serve Qwen/Qwen3-1.7B` and this script stays the same for base, SFT and RL checkpoints:

```bash
python scripts/generate.py --root data/bird --split mini_dev --model Qwen/Qwen3-1.7B --out results/preds/base_minidev.jsonl
```

```bash
python scripts/evaluate.py --root data/bird --split mini_dev --predictions results/preds/base_minidev.jsonl --stage baseline
```

`evaluate.py` appends one line to `results/runs.jsonl` and warns when the working tree is dirty,
because a run that cannot be pinned to a commit is not reportable.

## Repository layout

```text
configs/                      Training and evaluation configurations
scripts/                      Reproducible entry points: prepare_bird, generate, evaluate
src/text2sql_rlvr/sql/        SQLite-aware scanner, SQL extraction, read-only validation
src/text2sql_rlvr/data/       BIRD loading, schema introspection, prompt construction
src/text2sql_rlvr/rewards/    Execution sandbox, value canonicalisation, verifiers
src/text2sql_rlvr/eval/       Execution Accuracy and per-example outcomes
src/text2sql_rlvr/ledger.py   Append-only experiment ledger
tests/                        Unit and adversarial tests; build their own databases
docs/                         Data setup and design notes
results/                      runs.jsonl is tracked; everything else is ignored
```

## Remaining plan

1. Pass the three-step GRPO smoke test on the pinned AutoDL environment.
2. Run GRPO with the strict verifier as reward and evaluate on validation before Mini-Dev.
3. Run the matched naive-reward control with execution partial credit.
4. Compare rollout hack rates, then decide which ablations and error analyses are justified.

The operational continuation point is in `HANDOFF.md`. Conventions, data-split discipline and
reporting rules are in `AGENTS.md`.
