# Text2SQL-RLVR

An execution-grounded Text-to-SQL pipeline built with Qwen3-1.7B, LoRA SFT, verl GRPO,
vLLM, and SQLite. Generated SQL is executed against read-only BIRD databases, and the
execution result is used for both evaluation and reinforcement-learning reward.

## Final status

The complete path is implemented and tested:

```text
BIRD data -> schema-aware prompt -> LoRA SFT -> vLLM rollout
          -> read-only SQL execution -> official/strict scoring -> GRPO
```

The main controlled result is schema selection, not GRPO. All reportable rows below come
from a clean Git commit and have a unique entry in `results/runs.jsonl`.

| Model / prompt | Split | n | BIRD official EX | Strict EX | run_id |
|---|---|---:|---:|---:|---|
| Qwen3-1.7B baseline | Mini-Dev | 500 | 19.80% | 17.40% | `43bb66bbe1e8` |
| LoRA SFT | Mini-Dev | 500 | 34.60% | 30.00% | `f1ef9b247255` |
| Strong SFT, full schema | fixed train-val | 788 | 32.87% | 28.68% | `74d4d83c087c` |
| Strong SFT, linked schema | fixed train-val | 788 | 37.94% | 33.63% | `3b9e91f891d8` |
| Official-reward GRPO, linked schema | fixed train-val | 788 | 38.20% | 33.88% | `53d1586c7d33` |

Only rows on the same split are direct comparisons:

- Linked schema improves the strong SFT checkpoint from 32.87% to 37.94% official EX.
- GRPO changes linked-schema official EX from 37.94% to 38.20%. This is treated as no
  clear additional improvement, not as a successful RL gain.
- The 788-example set is a fixed validation split derived from BIRD train. BIRD dev was
  not used for tuning, checkpoint selection, or the numbers above.

See [the final experiment report](docs/FINAL_REPORT.md) for the design, paired analysis,
limitations, and interpretation. The chronological record, including failed and superseded
experiments, is kept in [PROGRESS.md](docs/PROGRESS.md).

## Why two execution metrics

Every evaluation reports two scores over the same predictions:

- `official_ex` reproduces BIRD's set-based Execution Accuracy and is the main reward and
  reportable benchmark metric.
- `strict_ex` additionally preserves duplicate rows and checks column count, so it remains
  a monitoring metric for cases that pass the official scorer without matching the full
  result semantics.

The two metrics are deliberately not collapsed into one. Training is aligned with the final
BIRD scorer, while the stricter result exposes possible metric exploitation.

## Schema linking

`generate.py` and the RL data builder support three schema modes:

- `full`: include the complete database schema;
- `linked`: select tables using question/evidence terms and retain foreign-key neighbours;
- `oracle`: include gold-SQL tables for diagnosis only, never for training or deployment.

Exact tokenizer diagnostics showed that the full prompt was not being truncated. The linked
prompt helped because it removed irrelevant schema context, although the lightweight linker
still misses some required tables and retains many distractors. It is a measured baseline,
not a claim that schema linking is solved.

## Safety and reproducibility

Model-generated SQL is handled by a SQLite-only sandbox with:

- immutable/read-only database access;
- a SQLite authorizer that denies writes and unsafe operations;
- single-statement validation and DDL/DML rejection;
- hard execution timeout, bounded results, connection reuse, and result caching.

`scripts/evaluate.py` appends experiment provenance to `results/runs.jsonl`, including the Git
SHA and dirty state, split, sample count, decoding settings, checkpoint, config hash, metrics,
and output path. Dirty runs are retained only as diagnostics and are not used in the table above.

## Local setup

The local development environment needs no GPU and the tests build their own temporary databases:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Download BIRD separately as described in [docs/data.md](docs/data.md). A deterministic generation
and evaluation run against an OpenAI-compatible endpoint looks like this:

```bash
python scripts/generate.py \
  --root data/bird \
  --questions data/processed/val.json \
  --split train \
  --schema-mode linked \
  --model text2sql-model \
  --temperature 0 \
  --top-p 1 \
  --seed 0 \
  --out results/preds/linked.jsonl

python scripts/evaluate.py \
  --root data/bird \
  --questions data/processed/val.json \
  --split train \
  --predictions results/preds/linked.jsonl \
  --stage ablation
```

GPU dependencies are isolated in `requirements-train.txt`. The validated GRPO configuration and
the observed Blackwell workarounds are documented in [docs/grpo-runbook.md](docs/grpo-runbook.md).

## Repository layout

```text
configs/                      SFT and GRPO configurations
scripts/                      Data, generation, evaluation, diagnostics, checkpoint export
src/text2sql_rlvr/sql/        SQL extraction and read-only validation
src/text2sql_rlvr/data/       BIRD loading, schema introspection, prompts, schema linking
src/text2sql_rlvr/rewards/    SQLite sandbox and execution verifiers
src/text2sql_rlvr/eval/       Execution Accuracy and per-example outcomes
src/text2sql_rlvr/ledger.py   Append-only experiment ledger
tests/                        Unit and adversarial tests using temporary SQLite databases
docs/                         Runbooks, final report, resume draft, milestone history
results/runs.jsonl            Tracked, append-only metric ledger
```

## Scope of the claim

This repository demonstrates a reproducible Text-to-SQL RLVR system and a controlled schema-linking
improvement. It does **not** claim that the ordinary GRPO run produced a meaningful accuracy gain,
that the lightweight linker is optimal, or that the reported train-val results are BIRD dev scores.
