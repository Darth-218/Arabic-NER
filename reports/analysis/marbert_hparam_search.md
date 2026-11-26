# MARBERT — Hyperparameter Search (issue 6.3)

**Owner:** Member 4 (Ahmed Maboud) · **Status:** plan defined; results pending GPU runs.

## Why only three settings

Fine-tuning a ~163M-parameter model costs real time. Rather than a large grid,
we vary the two knobs that usually matter most for a small NER dataset:

| Run | Learning rate | Epochs | Notes |
| :--- | :--- | :--- | :--- |
| `lr1e5_e2` | 1e-5 | 2 | conservative, lower overfitting risk |
| `lr2e5_e3` | 2e-5 | 3 | current default in `configs/marbert.yaml` |
| `lr3e5_e2` | 3e-5 | 2 | faster learning, possible instability |

Everything else stays fixed so the comparison is fair: `batch_size=16`,
`eval_batch_size=32`, `max_len=128`, `seed=42`, `warmup_ratio=0.06`,
`weight_decay=0.01`, and `UBC-NLP/MARBERT` as the checkpoint.

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

| Run | LR | Epochs | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lr1e5_e2` | 1e-5 | 2 | _pending_ | _pending_ | _pending_ |
| `lr2e5_e3` | 2e-5 | 3 | _pending_ | _pending_ | _pending_ |
| `lr3e5_e2` | 3e-5 | 2 | _pending_ | _pending_ | _pending_ |

**Best so far:** not yet measured. The default config uses `lr=2e-5, epochs=3`
as a reasonable starting point.

## Reproducibility rule

Whichever run wins, copy its values into `configs/marbert.yaml` (the YAML is the
source of truth), retrain once from scratch with that config, and report the
`results/metrics/marbert.json` produced by that final run — not the best number
from the search log.
