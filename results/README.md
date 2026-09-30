# Results

`runs.jsonl` is the experiment ledger. Every run that produces a metric appends exactly one
line, written by script rather than by hand, recording the Git commit, split, sample count,
decoding settings, checkpoint, config hash, metrics and output path. Every number quoted in
the project README maps to a unique line in this file.

Large artifacts (checkpoints, raw rollouts, generation dumps) stay out of Git.

## Qwen3-1.7B (August)

| run_id | Purpose | Split | n |
|---|---|---|---:|
| `43bb66bbe1e8` | Qwen3-1.7B baseline | Mini-Dev | 500 |
| `f1ef9b247255` | LoRA SFT | Mini-Dev | 500 |
| `74d4d83c087c` | strong SFT, full schema | fixed train-val | 788 |
| `3b9e91f891d8` | strong SFT, linked schema | fixed train-val | 788 |
| `53d1586c7d33` | official-reward GRPO, linked schema | fixed train-val | 788 |

`74d4d83c087c` and `3b9e91f891d8` are the controlled schema comparison; `3b9e91f891d8` and
`53d1586c7d33` are the controlled SFT-to-GRPO comparison. Mini-Dev rows and fixed train-val
rows are not compared directly. BIRD dev has no ledger row because it is held out.

## Qwen3-4B (September)

All rows are on the fixed train-val 788 with linked schema and greedy decoding, except the
oracle row, which hands the model exactly the gold tables.

| run_id | Purpose | official EX |
|---|---|---:|
| `a9eaa7f7da3e` | Qwen3-4B base | 34.01% |
| `27c6fe6380c4` | base + agent loop, up to 4 turns | 33.50% |
| `02fd984e7873` | base + oracle schema | 41.62% |
| `2c82eb0ead22` | SFT, 1,500 random examples | 42.39% |
| `c0c001faa315` | SFT, 1,500 examples chosen from base-model failures | 45.30% |
| `cc9818e2eeaf` | SFT, 550 targeted + 950 selected examples | 44.16% |

The three SFT arms share every hyperparameter (1,500 examples, 2 epochs, LoRA rank 32); the
only variable is which training questions were chosen. Every arm beats the base model with
p < 0.0001 on a paired McNemar test. Failure-driven selection beats random selection by
2.91 points (p = 0.0385), with most of the gain on the 65-table `works_cycles` database.

The agent loop cuts execution failures from 142 to 55 without moving accuracy, because most
remaining failures execute fine and return the wrong rows.

## Other stages in the ledger

- `selector`: table-selection metrics for the trained schema selector (split manifest
  `configs/splits/selector_split.json`, dataset manifests under `configs/selector/`).
- `ablation`: downstream execution accuracy for different schema inputs.
- Dataset construction runs for the failure audit and targeted SFT data, whose manifests
  are under `configs/analysis/` and `configs/data_construction/`.
