# Text2SQL-RLVR

An execution-grounded Text-to-SQL post-training system built on BIRD, Qwen3, LoRA, vLLM
and SQLite. Every query the model writes is executed against a read-only database, and
that execution is used three ways: as the reward during reinforcement learning, as the
filter that decides which training data is worth keeping, and as a tool the model itself
calls before committing to an answer.

## Status

The infrastructure is complete and tested. The generator model is being rebuilt on
Qwen3-4B after the earlier 1.7B checkpoints were lost, so the headline accuracy numbers
below are still the August 1.7B ones.

Results so far, each traceable to a unique line in `results/runs.jsonl`. Four of the five
come from a clean commit and are reportable; the baseline row does not, see the note below:

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
- The 788-example set is a fixed validation split derived from BIRD train, chosen so its
  three databases appear nowhere in training. BIRD dev has never been read, for tuning or
  for anything else.

**The baseline row is flawed and is marked as such.** `43bb66bbe1e8` carries
`git_dirty=true` in the ledger, so by this project's own rule it is not a reportable
number. It had been quoted as clean since August; the ledger review on 2026-09-16 caught
it. The remedy is cheap and needs no GPU: `results/preds/base_minidev.jsonl` still exists,
so re-scoring it on a clean tree produces a clean row. Until then the "19.80% to 34.60%"
claim rests on a dirty starting point.

Work since then runs on a dirty working tree and is therefore internal evidence only, not
reportable: the trained schema selector, the move to Qwen3-4B, and the agent loop. It is
all written up in [the project story](docs/PROJECT_STORY.md), with the day-by-day record
in [PROGRESS.md](docs/PROGRESS.md) and the August analysis in
[the final report](docs/FINAL_REPORT.md).

## Two results worth knowing before reading the code

**Schema selection is settled, and a trained selector did not beat a twenty-line linker.**
A lexical linker that keeps any table sharing a word with the question reached 99% per-table
recall by keeping 25 of 42 tables. Training a model to do better worked on its own terms:
it keeps the gold tables for 95% of questions while handing over 3 to 8 tables instead of
25. The SQL generator scored the same either way. Cutting a database from 42 tables to 25
is worth about 3 points of execution accuracy; cutting it further is worth nothing. The
oracle's remaining 6 points come from handing over exactly the gold tables and no others,
which leaks the join structure and no real selector can reproduce.

**Execution feedback fixes crashes, not misunderstandings.** Letting the model run a query,
read the result or the error, and revise cut execution failures from 142 to 55 out of 788.
Accuracy did not move, because three quarters of the failures are queries that execute
perfectly and return the wrong rows: `COUNT(*)` where the question needed
`COUNT(DISTINCT ...)`, a missing `LIMIT 1`. An executor has no reason to complain about
those. The loop does help where the model has to work: on questions it takes three or four
turns to answer it beats single-shot generation.

## Why two execution metrics

Every evaluation reports two scores over the same predictions:

- `official_ex` reproduces BIRD's set-based Execution Accuracy and is the main reward and
  reportable benchmark metric.
- `strict_ex` additionally preserves duplicate rows and checks column count, so it remains
  a monitoring metric for cases that pass the official scorer without matching the full
  result semantics.

The two metrics are deliberately not collapsed into one. Training is aligned with the final
BIRD scorer, while the stricter result exposes possible metric exploitation. It found 933
answers that omit a required de-duplication and are credited anyway.

## The agent loop

`src/text2sql_rlvr/agent.py` is an execute-observe-revise loop in about a hundred lines
over pieces the project already has. One action per reply, a hard turn budget, and a
plain-text protocol any chat model can follow without native tool calling:

```text
DESCRIBE <table>       the table's definition and a few example rows
a sql code block       the query is executed; its result or its error comes back
FINAL + a sql block    this query is the answer; the loop stops
```

Observations are appended as ordinary conversation turns, so a trajectory is a chat
transcript: it can be scored like any other prediction, or used as training data with the
loss restricted to the model's own turns. No agent framework is involved.

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

GPU dependencies are isolated in `requirements-train.txt`, pinned as a set that is known to work
together; the Blackwell workarounds it needed are documented there and in
[docs/grpo-runbook.md](docs/grpo-runbook.md). Neither TRL nor LLaMA-Factory is used: the SFT and
DPO trainers are single files over transformers and peft, so that environment stays undisturbed.

## Repository layout

```text
configs/                      split manifests, SFT / GRPO / DPO / selector dataset manifests
scripts/                      data, generation, evaluation, agent, training, export, diagnostics
src/text2sql_rlvr/sql/        SQL extraction, read-only validation, lexical scanning
src/text2sql_rlvr/data/       BIRD loading, schema introspection, prompts, schema linking,
                              SFT and preference data construction, selector data
src/text2sql_rlvr/rewards/    SQLite sandbox, result normalisation and comparison, rewards
src/text2sql_rlvr/eval/       Execution Accuracy, per-example outcomes, failure analysis
src/text2sql_rlvr/agent.py    execute-observe-revise loop
src/text2sql_rlvr/ledger.py   append-only experiment ledger
tests/                        357 unit and end-to-end tests on temporary SQLite databases
docs/                         project story, progress record, report, runbooks, analyses
results/runs.jsonl            tracked, append-only metric ledger
```

## Scope of the claim

This repository demonstrates a reproducible execution-grounded Text-to-SQL system and one
controlled improvement, schema linking. It does **not** claim that GRPO produced a meaningful
gain, that the trained schema selector beat the lexical one, that the agent loop improved
accuracy, or that any number here is a BIRD dev score. Where an experiment came out negative
it is written up as a negative result, with the diagnosis that follows from it.
