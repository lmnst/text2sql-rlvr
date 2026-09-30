# Text2SQL-RLVR

**English** | [中文](README.zh-CN.md)

Can a small open model learn to write correct SQL for large, messy databases, if the database
itself is used as the teacher? This project answers that on BIRD with Qwen3 models. Every
query the model writes is executed against a read-only copy of the database, and the result
is used in three ways: as the reward during reinforcement learning, as the filter that
decides which training data is worth keeping, and as a tool the model can call to check its
own answer before committing to it.

## Where the project stands

- **August, Qwen3-1.7B:** the full pipeline from data to SFT to GRPO was built and evaluated.
- **Since September, Qwen3-4B:** the generator has moved to a larger model with a redesigned
  approach, described in the next section. That work lives on the branch
  [`selector-stage1`](https://github.com/lmnst/text2sql-rlvr/tree/selector-stage1)
  ([PR #1](https://github.com/lmnst/text2sql-rlvr/pull/1)).

## The current approach (Qwen3-4B)

```text
question ─▶ schema selection ─▶ SQL generator (Qwen3-4B) ─▶ final SQL
                                   │            ▲
                                   ▼            │
                             read-only database: execute, observe, revise
```

**1. Give the model less context, but the right context.** BIRD databases can have dozens
of tables. A lexical linker, a small trained table selector and a one-hop expansion along
foreign keys narrow the schema before generation. If a needed table is still missing, the
model can ask for it during the agent loop.

**2. Train on what the model gets wrong.** Instead of sampling SFT data at random, the base
model first answers the whole training set, and fine-tuning focuses on the questions it
fails, with a share of already-solved questions replayed so that existing skills are not
lost. Frequent error types (choosing the wrong column, aggregating at the wrong level,
formatting the output incorrectly) get additional targeted examples. Every training example
is checked by actually executing it.

**3. Learn from its own samples.** The fine-tuned model answers each training question
several times, and execution sorts the answers into right and wrong. Its own most common
correct answer and its own most common mistake form a preference pair for DPO. The gold
query is used only as a capped fallback, because its writing style differs from the
model's and DPO could end up learning the style instead of correctness.

**4. Make self-correction a trained skill.** In the agent loop the model can inspect a
table, run a query, read the result or the error, and revise, for up to four turns.
Trajectories whose final query matches the gold result become training data, with the loss
restricted to the model's own turns. A stronger teacher is only brought in for questions
the student fails and the teacher solves, and only if that overlap is large enough to be
worth it. Otherwise the student bootstraps itself: it retries its failures with a hint about
the expected shape of the result, and the hint is removed from the training data.

**5. Add the signals an executor cannot give.** Most remaining errors are queries that run
fine and return the wrong rows. Two cheap signals target them: a flag for duplicated rows,
and asking the model to state which rows and columns a correct answer should have before
comparing that with what the query returned.

## Qwen3-4B: results so far

All rows use the fixed 788-question validation set, linked schema and greedy decoding.

| Configuration | official EX | run_id |
|---|---:|---|
| Qwen3-4B base | 34.01% | `a9eaa7f7da3e` |
| Base + agent loop (up to 4 turns, no training) | 33.50% | `27c6fe6380c4` |
| Base + oracle schema (upper bound for schema selection) | 41.62% | `02fd984e7873` |
| + SFT, 1,500 random examples | 42.39% | `2c82eb0ead22` |
| + SFT, 1,500 examples chosen from the base model's failures | **45.30%** | `c0c001faa315` |
| + SFT, 550 targeted + 950 selected examples | 44.16% | `cc9818e2eeaf` |

- All three SFT variants beat the base model clearly (p < 0.0001). SFT is the largest single
  lever so far, larger than giving the base model a perfect schema.
- Choosing examples from the model's own failures beats random selection by 2.91 points
  (p = 0.0385), with most of the gain on the largest database.

## Results so far (August, Qwen3-1.7B)

Each row can be traced to a unique line in `results/runs.jsonl`.

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
  clear additional improvement, not as a successful RL gain. The most likely cause is a
  sparse 0/1 reward: when every sampled answer to a question is right, or every one is
  wrong, the group carries no learning signal.
- The 788-example set is a fixed validation split derived from BIRD train, chosen so its
  three databases appear nowhere in training. BIRD dev has never been read, for tuning or
  for anything else.

## What we learned along the way

**Less context helps, up to a point.** Cutting a database from 42 tables to about 25 is
worth roughly 3 points of execution accuracy. The trained selector plus foreign-key
expansion compresses the input much further, from 25.5 to 7.8 tables on average, while still
keeping every needed table for 95.7% of validation questions, but this did not raise
execution accuracy beyond the lexical linker. The remaining gap to an oracle comes from handing over exactly the gold tables, which
leaks the join structure and cannot be reproduced by a real selector.

**Execution feedback fixes crashes, not misunderstandings.** Letting the model run a query
and revise cut execution failures from 142 to 55 out of 788, but accuracy did not move:
three quarters of the failures are queries that execute perfectly and return the wrong
rows, such as `COUNT(*)` where `COUNT(DISTINCT ...)` was needed. This finding is what
motivates steps 4 and 5 above.

## How the numbers are kept honest

- **Two metrics on the same predictions.** `official_ex` reproduces BIRD's scorer and is the
  reward and headline metric; `strict_ex` also keeps duplicate rows and checks the column
  count. The stricter one found 933 answers that skip a required de-duplication and are
  credited anyway.
- **A sandbox for model-written SQL.** Read-only access, one statement at a time, a hard
  timeout, and no writes of any kind.
- **An append-only ledger.** Every evaluation records the Git commit, the split, decoding
  settings and the output file, so each number can be traced back to the run that
  produced it.

## Running it

The tests need no GPU and build their own temporary databases:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Data download is described in [docs/data.md](docs/data.md). GPU dependencies are pinned
separately in `requirements-train.txt`. The SFT and DPO trainers are small single-file
scripts over transformers and peft; GRPO uses verl.

## Summary

The clearest gains so far come from schema linking and from choosing fine-tuning data by the
model's own failures. Experiments that did not help, such as plain GRPO with a sparse reward
or execution feedback alone, are documented together with the diagnosis they led to, and
that diagnosis shapes the current approach.
