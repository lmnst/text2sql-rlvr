# Results

Store small, reproducible metric summaries here. Each result must identify the dataset split, sample count, seed, model checkpoint, evaluation command, hardware, and source log.

`runs.jsonl` is the tracked experiment ledger: every run that produces a metric appends exactly one line, written by script rather than by hand. See the "实验记录" section in `AGENTS.md` for the required fields. Every number that appears in the README or on a resume must map to a unique line in this file.

Large artifacts (checkpoints, raw rollouts, generation dumps) stay out of Git.

## Final reportable runs

| run_id | Purpose | Split | n |
|---|---|---|---:|
| `43bb66bbe1e8` | Qwen3-1.7B baseline | Mini-Dev | 500 |
| `f1ef9b247255` | LoRA SFT | Mini-Dev | 500 |
| `74d4d83c087c` | strong SFT, full schema | fixed train-val | 788 |
| `3b9e91f891d8` | strong SFT, linked schema | fixed train-val | 788 |
| `53d1586c7d33` | official-reward GRPO, linked schema | fixed train-val | 788 |

`74d4d83c087c` and `3b9e91f891d8` are the controlled schema comparison.
`3b9e91f891d8` and `53d1586c7d33` are the controlled SFT-to-GRPO comparison.
Do not compare Mini-Dev rows directly with fixed train-val rows.

Older ledger rows remain append-only evidence for earlier baselines and superseded experiments.
In particular, `a8f7fa3bdecc` used the old strict-reward path and is not the final GRPO claim.
BIRD dev has no ledger row because it was not read for this project iteration.
