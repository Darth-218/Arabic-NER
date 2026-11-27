# Approach 4 — Advanced Arabic MARBERT

Owner: Member 4 (`@ahmedmaboud`).

## Files

| File | Issue | Purpose |
| :--- | :--- | :--- |
| `train.py` | 6.1, 6.2 | Fine-tuning CLI; canonical flags from `ROADMAP.md` §2.4 |
| `predict.py` | 6.4 | Canonical JSONL predictions with one tag per word |
| `hparam_search.py` | 6.3 | Optional 3-run grid over LR, epochs, batch size and max length (`--dry-run` to preview) |
| `configs/marbert.yaml` | 6.2 | Hyperparameters and model checkpoint (single source of truth) |

Reports: `reports/phase1/marbert.md` (6.5),
`reports/analysis/marbert_error_analysis.md` (6.4),
`reports/analysis/marbert_hparam_search.md` (6.3).

## Commands

```bash
python src/approach_4_marbert/train.py --config configs/marbert.yaml --epochs 3 --batch_size 16
python src/approach_4_marbert/predict.py --model_dir results/models/marbert --split test \
  --out results/predictions/marbert_test.jsonl
python src/evaluate.py --model marbert --split test --config configs/marbert.yaml
python src/approach_4_marbert/predict.py --model_dir results/models/marbert \
  --split validation --out results/predictions/marbert_validation.jsonl
python src/evaluate.py --model marbert --split validation --config configs/marbert.yaml \
  --out results/metrics/marbert_validation.json
python src/approach_4_marbert/hparam_search.py --dry-run
```

`predict.py` writes the **original** WikiANN tokens in the JSONL; Arabic
normalization is applied only to the model input. A truncation audit at
`max_len=128` found 0 over-limit test/validation sequences (longest encodings
87/101), so no tail words were converted to `O`. Keep this same `max_len` and
`truncation=True` policy for mBERT and AraBERT when their configs land.

## Compute / reproducibility

Per TL guidance (Nov 2026): **run locally, using the scripts in this repo — not
notebooks.** Cloud execution is allowed, but only with these same scripts kept in
the repository, so `main` remains the single source of truth.

The commands above work as-is from the repo root after
`pip install -r requirements-cpu.txt` (or `requirements-gpu.txt` on a GPU
machine — `requirements.txt` alone no longer installs PyTorch). CPU is
supported (the smoke test was verified on CPU), but a full 3-epoch run over 20k
sequences takes several hours; a GPU simply makes the same commands faster — no
code changes are needed.

If you do run in Colab, clone this repo there, install `requirements-gpu.txt`, and
execute the exact same commands. Commit any script change back to this branch;
do not create a separate training notebook.

### Smoke test (before any full run)

All three entry points accept a matching sample limit so the plumbing can be
verified in minutes without training on the full dataset:

```bash
python src/approach_4_marbert/train.py --config configs/marbert.yaml \
  --epochs 1 --max_train_samples 200 --max_eval_samples 100 \
  --output_dir results/models/marbert_smoke
python src/approach_4_marbert/predict.py --model_dir results/models/marbert_smoke \
  --split test --max_samples 100 --out results/predictions/marbert_smoke_test.jsonl
python src/evaluate.py --model marbert --split test --config configs/marbert.yaml \
  --max_samples 100 --predictions results/predictions/marbert_smoke_test.jsonl \
  --out results/metrics/marbert_smoke.json
```

Use the **same `--max_samples` value** in `predict.py` and `evaluate.py`, and
write smoke artifacts to separate paths so they never overwrite a real run.
The smoke F1 is meaningless — it only proves the pipeline works end to end.

On Windows, if the model download fails with a Hugging Face *Xet/CAS* error,
set `$env:HF_HUB_DISABLE_XET="1"` before running (plain HTTP download works).
Smoke artifacts match the ignore rules in `.gitignore` and are never committed.

Uses the shared loader, data utilities, evaluation and results writer — it does
**not** reimplement tokenization, alignment or metrics.

## Note on the temporary shared harness

`data/load_data.py`, `src/data_utils.py`, `src/evaluate.py`, `src/results.py`
and `src/compare.py` are currently minimal stand-ins created so this approach
can run before Member 1 merges Epic 1/2. They are explicitly marked as M1-owned
placeholders and must be reviewed/replaced by M1. This approach imports only
their documented public functions, so no M4 code changes are needed then.

`hparam_search.py` is an addition to the canonical tree in §2.1 and should be
acknowledged in `ROADMAP.md` by M1 before merging (§2 "do not deviate").
