# Text2SQL-RLVR

Execution-grounded reinforcement learning for Text-to-SQL with Qwen3, verl, and vLLM.

## Goal

Build a reproducible SFT-to-GRPO pipeline on BIRD in which generated SQL is executed against read-only databases and scored with verifiable rewards.

## MVP

1. Prepare BIRD data and database schemas.
2. Reproduce a Qwen3-1.7B baseline with the official execution-accuracy evaluator.
3. Run LoRA SFT and record a reproducible checkpoint and evaluation result.
4. Implement sandboxed SQL execution and exact result-set rewards.
5. Run GRPO with verl and vLLM rollout.
6. Compare SFT, GRPO, and GRPO with dynamic sampling using measured results only.

## Repository layout

```text
configs/                    Training and evaluation configurations
scripts/                    Reproducible entry-point scripts
src/text2sql_rlvr/data/     BIRD preprocessing and prompt construction
src/text2sql_rlvr/rewards/  SQL sandbox, canonicalization, and reward functions
src/text2sql_rlvr/eval/     Official evaluation wrappers and error analysis
tests/                      Unit and adversarial verifier tests
docs/                       Experiment notes and design decisions
results/                    Small tracked summaries; large artifacts are ignored
```

## Status

Project scaffold only. No experiment results have been produced yet.

