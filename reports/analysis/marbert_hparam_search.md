# MARBERT — Hyperparameter Search (issue 6.3)

**Owner:** Member 4 (Ahmed Maboud) · **Status:** default setting measured; optional grid not run.

## Why only three settings

Fine-tuning a ~163M-parameter model costs real time. Rather than a large grid,
we vary the two knobs that usually matter most for a small NER dataset:

| Run | Learning rate | Epochs | Batch | Max len | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lr1e5_e2_bs16_len128` | 1e-5 | 2 | 16 | 128 | conservative, lower overfitting risk |
| `lr2e5_e3_bs16_len128` | 2e-5 | 3 | 16 | 128 | current default in `configs/marbert.yaml` |
| `lr3e5_e2_bs32_len64` | 3e-5 | 2 | 32 | 64 | faster learning, shorter context window |

Everything else stays fixed so the comparison is fair: `eval_batch_size=32`,
`seed=42`, `warmup_ratio=0.06`, `weight_decay=0.01`, and `UBC-NLP/MARBERT` as
the checkpoint. Learning rate, epochs, batch size and max length are all varied
across the three runs, matching the issue 6.3 task list.

## How to run it

```bash
python src/approach_4_marbert/hparam_search.py --dry-run   # inspect commands
python src/approach_4_marbert/hparam_search.py             # actual search
```

The helper trains, predicts and evaluates each run through the canonical
entry points. After each run it archives the metrics and predictions into
`reports/analysis/marbert_hparam_runs/<run_name>/` before the next run
overwrites the canonical files. It finally writes
`reports/analysis/marbert_hparam_search.json` with the best F1 and the command
to retrain that setting.

> Note: `hparam_search.py` is an optional addition to the canonical directory
> tree. Per `ROADMAP.md` §2, ask M1 to acknowledge it in the roadmap (or fold
> the commands into the Makefile) before merging.

## Results table (fill from the archived metrics)

| Run | LR | Epochs | Batch | Max len | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `lr1e5_e2_bs16_len128` | 1e-5 | 2 | 16 | 128 | _not run_ | _not run_ | _not run_ |
| `lr2e5_e3_bs16_len128` | 2e-5 | 3 | 16 | 128 | 0.8889 | 0.8960 | 0.8924 |
| `lr3e5_e2_bs32_len64` | 3e-5 | 2 | 32 | 64 | _not run_ | _not run_ | _not run_ |

**Best so far:** `lr2e5_e3_bs16_len128` with test F1 **0.8924** and validation
F1 **0.8969** (`results/metrics/marbert_validation.json`) — the default in
`configs/marbert.yaml`. The other two settings were deliberately not run: each
full CPU training pass takes about three hours, and the default already gives a
strong baseline. The helper now varies all four knobs and can be run later on a
GPU to complete the grid.

## Reproducibility rule

Whichever run wins, copy its values into `configs/marbert.yaml` (the YAML is the
source of truth), retrain once from scratch with that config, and report the
`results/metrics/marbert.json` produced by that final run — not the best number
from the search log.
