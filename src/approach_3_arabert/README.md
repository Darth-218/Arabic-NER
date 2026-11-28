# Approach 3 — Arabic AraBERT

Owner: Member 3 (`@OmarHesham2356`).

- `train.py` — fine-tuning CLI (#26, issue 5.1; finalized in 5.2)
- `predict.py` — canonical predictions (#28, issue 5.4)

Uses the shared loader/alignment/metrics/results modules; see `ROADMAP.md` §2.4.

## Training

```bash
python src/approach_3_arabert/train.py --config configs/arabert.yaml

# Smoke test: 200 train / 100 validation sequences, one epoch
python src/approach_3_arabert/train.py --config configs/arabert.yaml \
  --epochs 1 --max_train_samples 200 --max_eval_samples 100
```

Canonical flags (`ROADMAP.md` §2.4): `--config`, `--output_dir`, `--seed`,
`--epochs`, `--batch_size`, `--lr`. Everything else comes from the YAML;
an explicitly passed flag always wins over the file.

Outputs (git-ignored):

- `results/models/arabert/` — best checkpoint + tokenizer
- `results/models/arabert/resolved_config.json` — merged settings of that run

## `configs/arabert.yaml`

| Key | Meaning |
| :--- | :--- |
| `model`, `approach`, `owner` | Metadata; validated to be `arabert` / `arabic_transformer` / `member_3` |
| `model_name` | Hugging Face checkpoint (`aubmindlab/bert-base-arabertv02`) |
| `cache_dir` | Local Hugging Face cache (`data/raw`) |
| `epochs`, `batch_size`, `eval_batch_size` | Passes over the data and per-device batch sizes |
| `gradient_accumulation_steps` | Effective batch = `batch_size × steps` (needed on 4 GB VRAM) |
| `lr`, `weight_decay`, `warmup_ratio` | Optimization; warmup is a fraction of total steps |
| `max_len` | Sub-word tokens per sequence; longer sequences are truncated |
| `seed` | Reproducibility |
| `bf16` / `fp16` | Mixed precision (`bf16` on Ampere+); at most one may be true |
| `gradient_checkpointing` | Trades speed for GPU memory |
| `label_all_subwords` | `false` = only a word's first sub-word is supervised |
| `normalization` | Arabic cleaning toggles (diacritics, tatweel, alef, ya) |
| `output_dir` | Checkpoint destination |
| `max_train_samples` / `max_eval_samples` | Smoke-test limits; `null` for real runs |

`validate_config()` rejects unknown keys and impossible values before any
download starts, so typos fail in seconds instead of after an hour.

## Tests

`tests/test_arabert.py` runs fully offline (stub splits, no torch):

```bash
python -m pytest tests/test_arabert.py
```
