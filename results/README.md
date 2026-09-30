# Results

Store small, reproducible metric summaries here. Each result must identify the dataset split, sample count, seed, model checkpoint, evaluation command, hardware, and source log.

`runs.jsonl` is the tracked experiment ledger: every run that produces a metric appends exactly one line, written by script rather than by hand. See the "实验记录" section in `AGENTS.md` for the required fields. Every number that appears in the README or on a resume must map to a unique line in this file.

Large artifacts (checkpoints, raw rollouts, generation dumps) stay out of Git.

## Final reportable runs

| run_id | Purpose | Split | n |
|---|---|---|---:|
| `43bb66bbe1e8` | Qwen3-1.7B baseline (dirty, see below) | Mini-Dev | 500 |
| `f1ef9b247255` | LoRA SFT | Mini-Dev | 500 |
| `74d4d83c087c` | strong SFT, full schema | fixed train-val | 788 |
| `3b9e91f891d8` | strong SFT, linked schema | fixed train-val | 788 |
| `53d1586c7d33` | official-reward GRPO, linked schema | fixed train-val | 788 |

`43bb66bbe1e8` carries `git_dirty=true` and is therefore **not** reportable, despite having
been quoted as such since August; the 2026-09-16 ledger review caught it. Re-scoring the
surviving `results/preds/base_minidev.jsonl` on a clean tree would produce a clean row.

`74d4d83c087c` and `3b9e91f891d8` are the controlled schema comparison.
`3b9e91f891d8` and `53d1586c7d33` are the controlled SFT-to-GRPO comparison.
Do not compare Mini-Dev rows directly with fixed train-val rows.

Older ledger rows remain append-only evidence for earlier baselines and superseded experiments.
In particular, `a8f7fa3bdecc` used the old strict-reward path and is not the final GRPO claim.
BIRD dev has no ledger row because it was not read for this project iteration.

## Offline baseline failure audit (2026-09-13)

See `docs/analysis/BASELINE_FAILURE_ANALYSIS.md` for the fixed-val baseline audit,
Codex-reviewed cases and proposed failure-driven data construction experiments.
Complete audit run: `da325563f9a6`, source inference/evaluation run: `f9dc29d50c69`.
The audit is marked dirty and is internal diagnostic evidence, not a new reportable
model score. It preserves historical outcomes and separately replays the extraction
fix. Immutable evidence is under `results/analysis/base_val_v1/runs/da325563f9a6/`;
the parent directory contains convenient latest-view files. Earlier partial audit
runs from script development remain in the append-only ledger and are superseded
by this complete audit.

## Archived stage comparison (2026-09-13)

See `docs/analysis/ARCHIVED_EXPERIMENT_FAILURE_ANALYSIS.md` for the offline comparison
of Base, early SFT and old strict-reward GRPO. Complete run: `f530027b310b`;
immutable output: `results/analysis/archived_val_stages/runs/f530027b310b/`.
The source outputs reproduce their ledger scores. This is a dirty internal val
diagnostic, not current-best SFT train-side diagnosis or a new inference run.
Final strong SFT / official-reward GRPO summaries remain in the existing ledger;
their detailed cloud outputs and weights have not yet been located on the new instance.

## Targeted SFT data construction (2026-09-14)

Dataset audit run `88ce3420e942` constructed and exported 60 examples in 30 pairs,
with 20 examples per target family, from 26 train source questions and 9 databases.
See `docs/data/TARGETED_SFT_V1.md` for provenance, semantic review and limitations.
Artifacts are under `data/targeted_sft/v1/runs/88ce3420e942/`.
These are internal dataset verification statistics, not model accuracy or training gains.
No model inference or training ran; the ledger marks the construction run dirty.

## Targeted expansion plus replay (2026-09-14)

Final dataset audit `ca06ead04cbb` exports 400 targeted examples and 150 unchanged
original replay examples, mixed once with seed 0. Targeted data uses 50 reviewed
templates with four paired instances each and retains all 60 v1 examples.
See `docs/data/TARGETED_SFT_V2.md`; artifacts are under
`data/targeted_sft/v2/runs/ca06ead04cbb/`.
This is a dirty internal dataset run, with no model inference or training.
Intermediate run `3bf75dc9a93c` is superseded: its one-based mailing-ID assumption
was corrected to an actual first-N customer set check before final export.

## Schema selector, and why the line was closed (2026-09-15 to 09-16)

Ledger stage `selector` holds the table-selection metrics; stage `ablation` holds the
downstream execution accuracy those selections produced. The selector matches the lexical
linker's every-gold-table-kept rate while keeping 3 to 8 tables instead of 25, but the SQL
generator scores the same either way, so the line is closed. See `docs/PROGRESS.md`
(entries dated 2026-09-14 to 09-16) for the decomposition into retention rate times
accuracy on retained questions. Split manifest: `configs/splits/selector_split.json`.
Dataset manifests: `configs/selector/dataset_v1.json`, `_v2.json`, `_v3a.json`, `_v3b.json`.
Row `7144857b460b` is void: its generation run lost the tunnel after 48 of 788 questions,
so the evaluation scored 788 missing predictions as 0.00%.

## Qwen3-4B ladder, work in progress (2026-09-16)

The generator was changed to Qwen3-4B; the 1.7B strong-SFT and GRPO weights are lost.
Every 4B row so far is the untrained base model on the fixed val 788, and every one is
dirty: `a9eaa7f7da3e` linked schema 34.01%, `02fd984e7873` oracle 41.62%, `27c6fe6380c4`
the agent loop 33.50%. The loop cuts execution failures from 142 to 55 without moving
accuracy, because three quarters of the failures execute fine and return the wrong rows.

The 4B base was also run over all 8191 training questions to produce a per-question
outcome map (`results/outcomes/train_linked_4b.jsonl`), evaluated with `--no-ledger`
because a score on the training split is not a result and must never be quoted as one.
That map drives SFT question selection, DPO pair mining and later teacher targeting.
Dataset manifests: `configs/sft/dataset_4b_random.json`, `_4b_failure.json`,
`_4b_targeted550.json`; `configs/dpo/dataset_4b_base.json`.

## Three-arm SFT comparison on Qwen3-4B (2026-09-16)

All three arms are 1500 examples, 2 epochs, LoRA rank 32, linked schema, identical
hyperparameters; the only variable is which training questions were chosen. Scored on the
fixed val 788 at temperature 0. All dirty.

| Arm | official EX | run_id |
|---|---:|---|
| random 1500 | 42.39% | `2c82eb0ead22` |
| selected where the base model failed | 45.30% | `c0c001faa315` |
| 550 constructed plus 950 selected | 44.16% | `cc9818e2eeaf` |

Every arm beats the untrained base (`a9eaa7f7da3e`, 34.01%) with p below 0.0001 on a
paired McNemar test. Failure-selection beats random by 2.91 points at p=0.0385, which does
**not** survive Bonferroni correction over the six comparisons that were run (threshold
0.0083), and its advantage is concentrated in the 65-table works_cycles database. Neither
difference involving the constructed set is significant. See `docs/PROGRESS.md` for the
full breakdown.
