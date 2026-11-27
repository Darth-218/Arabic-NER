# Data

Dataset card and loading entry point for the Arabic WikiANN data.

- **Source:** `unimelb-nlp/wikiann`, configuration `ar` (Hugging Face Datasets)
- **License:** not declared upstream (dataset card says `unknown`); text is derived from Wikipedia
- **Splits:** 20,000 train / 10,000 validation / 10,000 test sequences
- **Columns:** `tokens` (`List[str]`), `ner_tags` (`List[int]`, values from `LABEL2ID`)
- **Labels (BIO):** `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`

## Entry points

| File | Purpose | Issue |
| :--- | :--- | :--- |
| `load_data.py` | Canonical loader `load_arabic_wikiann()` | #5 (1.1) |
| `eda.ipynb` | Exploratory data analysis | #6 (1.2) |
| `quality_report.py` | Duplicates / leakage / label coverage | #10 (1.6) |

## Rules

- Load data **only** through `data/load_data.py`; do not call `load_dataset` elsewhere.
- The loader fails fast if split sizes drift (`strict=True`); use `--no-strict` only for upstream-change diagnostics.
- Extra upstream columns (`langs`, `spans`) are dropped; only `tokens`/`ner_tags` are exposed.
- Raw data is cached under `data/raw/` and is git-ignored.
- Tests use tiny local fixtures and never hit the network.
