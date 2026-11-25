# Arabic NER — Project Roadmap

This document is the **single source of truth** for how work is organized and how the
four approaches fit together. Every member must follow the **Canonical Contract** in
[Section 2](#2-canonical-contract-do-not-deviate) so that independently developed
components integrate without rework.

- **Project:** Arabic Named Entity Recognition — A Comparative Study (AIS-411)
- **Team:** 4 members, one distinct approach each
- **Models:** CRF (classical), mBERT (multilingual), AraBERT (Arabic), MARBERT (advanced Arabic)
- **Owner codes:** `[M1]`, `[M2]`, `[M3]`, `[M4]` — resolve via the Team Roster below (`M1` is Project Manager)

### Team Roster

| Code | Member | GitHub | Role | Approach |
| :--- | :--- | :--- | :--- | :--- |
| M1 | Member 1 | `@Darth-218` | **Project Manager** | Data/EDA + CRF |
| M2 | Jannah Ashraf | `@jannahashraf` | Member | mBERT |
| M3 | Omar Hesham | `@OmarHesham2356` | Member | AraBERT |
| M4 | Ahmed Maboud | `@ahmedmaboud` | Member | MARBERT |

---

## 1. Phase Model

| Phase | Milestone | Goal |
| :--- | :--- | :--- |
| Phase 0 | `M0 – Setup` | Repo scaffold, conventions, shared utilities. Nothing else starts until the Canonical Contract is in place. |
| Phase 1 | `M1 – Phase 1` | Data pipeline, evaluation harness, first working version of all four approaches, Phase 1 report. |
| Phase 2 | `M2 – Phase 2` | Hyperparameter optimization, cross-model comparison, final report & presentation. |

Dependency chain: **`0.x → 1.x → 2.x → {3.x, 4.x, 5.x, 6.x} → 7.x → 8.x`**.
Approaches 3–6 may run in parallel once the harness (Epic 2) is green.

---

## 2. Canonical Contract (do not deviate)

> Anything in this section is a hard interface. If a change is needed, open an issue,
> update this document in the same PR, and announce it to the team before merging.

### 2.1 Canonical repository structure

```
arabic-ner/
├── data/
│   ├── load_data.py            # 1.1 WikiANN loader (single entry point for data)
│   ├── eda.ipynb               # 1.2 EDA notebook
│   └── README.md               # dataset card (source, splits, license)
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py           # LABELS, LABEL2ID, ID2LABEL, MODEL_NAMES, paths
│   │   ├── seed.py             # set_seed(seed)
│   │   └── logging_utils.py    # get_logger(name)
│   ├── data_utils.py           # 1.3–1.5 tokenization, alignment, dataloaders
│   ├── evaluate.py             # 2.1–2.2 seqeval metrics + per-class report
│   ├── results.py              # 2.3 unified results writer + table generator
│   ├── compare.py              # 7.1 aggregate + comparison table
│   ├── approach_1_crf/
│   │   ├── __init__.py
│   │   ├── features.py         # 3.1 feature extraction
│   │   ├── train.py            # 3.2 training CLI
│   │   └── predict.py          # produces canonical predictions
│   ├── approach_2_mbert/
│   │   ├── train.py
│   │   └── predict.py
│   ├── approach_3_arabert/
│   │   ├── train.py
│   │   └── predict.py
│   └── approach_4_marbert/
│       ├── train.py
│       └── predict.py
├── configs/
│   ├── crf.yaml                # per-approach hyperparameters
│   ├── mbert.yaml
│   ├── arabert.yaml
│   └── marbert.yaml
├── results/
│   ├── metrics/{model}.json    # canonical metrics (schema below)
│   ├── predictions/{model}_{split}.jsonl
│   └── plots/
├── reports/
│   ├── eda/                    # 1.2 figures & summary
│   ├── analysis/               # 7.x outputs
│   ├── phase1/                 # 8.1
│   └── phase2/                 # 8.2
├── tests/                      # unit tests for data_utils, evaluate, results
├── requirements.txt
├── Makefile
├── ROADMAP.md                  # this file
└── README.md
```

**Rules**
- Only `src/common/config.py` defines labels and model names. No magic strings elsewhere.
- Every approach exposes exactly the same two entry points: `train.py` and `predict.py`.
- Approach folders import shared logic from `src/data_utils.py`, `src/evaluate.py`,
  `src/results.py`; they never re-implement clicking/token alignment or metrics.
- All generated artifacts live under `results/`; raw datasets and checkpoints are **not** committed.

### 2.2 Canonical label schema

Defined **once** in `src/common/config.py` and imported everywhere:

```python
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
IGNORE_INDEX = -100  # subword tokens that must be ignored by the loss/metrics
```

### 2.3 Canonical dataset object

`data/load_data.py` exposes:

```python
def load_arabic_wikiann(cache_dir: str | None = None) -> DatasetDict:
    """Returns splits 'train' (20k), 'validation' (10k), 'test' (10k).
    Columns: 'tokens': List[str], 'ner_tags': List[int] (values from LABEL2ID).
    """
```

No member may load the dataset any other way.

### 2.4 Canonical model I/O

Every `train.py` accepts at least:

```
--config configs/<model>.yaml
--output_dir results/models/<model>   # checkpoints (git-ignored)
--seed 42
--epochs INT
--batch_size INT
--lr FLOAT
```

Every `predict.py` accepts `--model_dir`, `--split {train,validation,test}`,
`--out results/predictions/<model>_<split>.jsonl` and writes **one JSON object per line**:

```json
{"id": 0, "tokens": ["..."], "tags": ["B-PER", "O"]}
```

Prediction length MUST equal the gold token length (post-alignment), so predictions
are directly comparable to `gold["ner_tags"]`.

### 2.5 Canonical metrics schema

`src/results.py` writes `results/metrics/<model>.json` exactly in this shape:

```json
{
  "model": "mbert",
  "approach": "multilingual_transformer",
  "owner": "member_2",
  "split": "test",
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "per_entity": {
    "PER": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    "ORG": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    "LOC": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
  },
  "config": {},
  "timestamp": "ISO-8601"
}
```

Canonical model identifiers: **`crf`, `mbert`, `arabert`, `marbert`**.

### 2.6 Canonical commands

```bash
make install     # pip install -r requirements.txt
make data        # python data/load_data.py
make train M=crf    # python src/approach_1_crf/train.py --config configs/crf.yaml
make predict M=crf
make evaluate M=crf
make compare     # regenerate results + README table
```

Member-specific equivalent (must produce identical artifacts):

```bash
python src/approach_2_mbert/train.py --config configs/mbert.yaml --epochs 3 --batch_size 16
```

### 2.7 Collaboration workflow

- **Branches:** `feat/<member>/<issue-id>-<slug>` e.g. `feat/m1/1.4-subword-alignment`.
  Fixes: `fix/<member>/<issue-id>-<slug>`; docs: `docs/<member>/<slug>`.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`),
  referencing the issue id, e.g. `feat(data): subword alignment (#14)`.
- **PRs:** title `[M2][#4.2] Training CLI & config`; body links the issue, lists
  acceptance criteria checked, and pastes the key metric (or "N/A" for docs).
- **Review:** at least one other member approves; the area owner (Epic lead) is the
  required reviewer for shared files (`src/common/`, `data_utils.py`, `evaluate.py`,
  `results.py`).
- **Definition of Done** (every issue): code merged to `main`, tests pass (`make test`),
  artifacts written to the canonical paths, docstring/config updated, and the
  acceptance criteria in the issue are verified.

### 2.8 Ownership map

| Area / file | Lead | Reviewers |
| :--- | :--- | :--- |
| `src/common/`, `ROADMAP.md`, infra | M1 | all |
| `data/`, `src/data_utils.py` | M1 | M4 |
| `src/evaluate.py`, `src/results.py` | M1 | all approach owners |
| `src/approach_1_crf/` | M1 | M4 |
| `src/approach_2_mbert/` | M2 | M1 |
| `src/approach_3_arabert/` | M3 | M1 |
| `src/approach_4_marbert/` | M4 | M1 |
| `src/compare.py`, `reports/` | M1 | all |

---

## 3. Epic & Issue Index

Issue ids below are stable and are used as the title prefix on GitHub
(e.g. `[1.4] Subword label alignment`). Execute in dependency order.

### Epic 0 — Infrastructure (`M0`) — Owner: [M1]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 0.1 | Repo scaffold & canonical directory structure | M1 | – | Tree matches §2.1; `pip install -r requirements.txt` works; `.gitignore` excludes `data/raw`, `results/models` |
| 0.2 | Labels, milestones, project board & issue/PR templates | M1 | 0.1 | Labels per §2.7; 3 milestones; templates present |
| 0.3 | Shared utilities: config, seeding, logging | M1 | 0.1 | `src/common/config.py` is the only label source; `set_seed` reproducible |
| 0.4 | Model artifact & checkpoint policy | M1 | 0.1 | Policy documented; checkpoints not in git; loading path defined |

### Epic 1 — Data Pipeline (`M1`) — Owner: [M1]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | WikiANN Arabic loader & caching | M1 | 0.3 | Returns splits 20k/10k/10k with `tokens`/`ner_tags` per §2.3 |
| 1.2 | EDA notebook & report | M1 | 1.1 | Label/entity/sequence/vocab stats + figures in `reports/eda/` |
| 1.3 | Arabic normalization & tokenization | M1 | 1.1 | Diacritics/tatweel/alef-ya normalization configurable; documented |
| 1.4 | Subword label alignment (`data_utils.py`) | M1 | 1.3 | Word→subword alignment with `-100`; unit tests pass |
| 1.5 | DataLoaders & collator | M1 | 1.4 | Dynamic padding + `max_len`; reusable by all approaches |
| 1.6 | Data quality & leakage report | M1 | 1.1 | No train/test overlap; findings in `reports/eda/` |

### Epic 2 — Evaluation Harness (`M1`) — Owner: [M1]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | `seqeval` metrics module | M1 | 0.3 | P/R/F1 micro+macro, per-entity; validated on toy data |
| 2.2 | Confusion matrix & per-class report | M1 | 2.1 | Exports matrix + classification report |
| 2.3 | Unified results writer & markdown table generator | M1 | 2.1 | Writes §2.5 schema; regenerates README results table |
| 2.4 | All-`O` baseline sanity check | M1 | 2.1 | Baseline metrics recorded in `results/metrics/baseline.json` |

### Epic 3 — Approach 1: Classical CRF (`M1`) — Owner: [M1]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | CRF feature engineering | M1 | 1.3 | Word, char n-grams, orthographic, gazetteer features documented |
| 3.2 | CRF training & tuning | M1 | 2.3, 3.1 | `configs/crf.yaml`; artifacts in canonical paths |
| 3.3 | BiLSTM-CRF comparison (stretch) | M1 | 3.2 | Trained + metrics; or documented as out of scope |
| 3.4 | CRF evaluation & artifacts | M1 | 3.2 | Predictions + `results/metrics/crf.json` valid |
| 3.5 | CRF report section | M1 | 3.4 | Methods/results/errors written for Phase 1 report |

### Epic 4 — Approach 2: mBERT (`M2`) — Owner: [M2]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 4.1 | mBERT baseline fine-tune | M2 | 1.5, 2.3 | Token classification head; runs end-to-end |
| 4.2 | Training CLI & config | M2 | 4.1 | Conforms to §2.4; `configs/mbert.yaml` |
| 4.3 | Hyperparameter search | M2 | 4.2 | LR/epochs/batch/max_len searched; best config recorded |
| 4.4 | Evaluation & error analysis | M2 | 4.3 | `results/metrics/mbert.json`; error examples documented |
| 4.5 | mBERT report section | M2 | 4.4 | Methods/results/errors written for Phase 1 report |

### Epic 5 — Approach 3: AraBERT (`M3`) — Owner: [M3]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 5.1 | AraBERT baseline fine-tune | M3 | 1.5, 2.3 | Runs end-to-end with canonical interfaces |
| 5.2 | Training CLI & config | M3 | 5.1 | Conforms to §2.4; `configs/arabert.yaml` |
| 5.3 | Hyperparameter search | M3 | 5.2 | Best config recorded |
| 5.4 | Evaluation & error analysis | M3 | 5.3 | `results/metrics/arabert.json`; error examples documented |
| 5.5 | AraBERT report section | M3 | 5.4 | Methods/results/errors written for Phase 1 report |

### Epic 6 — Approach 4: MARBERT (`M4`) — Owner: [M4]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 6.1 | MARBERT baseline fine-tune | M4 | 1.5, 2.3 | Runs end-to-end with canonical interfaces |
| 6.2 | Training CLI & config | M4 | 6.1 | Conforms to §2.4; `configs/marbert.yaml` |
| 6.3 | Hyperparameter search | M4 | 6.2 | Best config recorded |
| 6.4 | Evaluation & error analysis | M4 | 6.3 | `results/metrics/marbert.json`; error examples documented |
| 6.5 | MARBERT report section | M4 | 6.4 | Methods/results/errors written for Phase 1 report |

### Epic 7 — Comparative Analysis (`M2`) — Owner: [M1]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 7.1 | Aggregate results & comparison table | M1 | 3.4, 4.4, 5.4, 6.4 | README table populated from `results/metrics/*.json` |
| 7.2 | Cross-model error analysis | M1 | 7.1 | Shared/differing error patterns in `reports/analysis/` |
| 7.3 | Significance / statistical comparison | M1 | 7.1 | Method + result documented |
| 7.4 | Visualization dashboard | M1 | 7.1 | Per-entity F1, F1 vs size/latency in `results/plots/` |

### Epic 8 — Reporting (`M2`) — Owner: [M1]

| ID | Title | Owner | Depends | Acceptance criteria |
| :--- | :--- | :--- | :--- | :--- |
| 8.1 | Phase 1 report | M1 | 3.5, 4.5, 5.5, 6.5 | Report in `reports/phase1/` |
| 8.2 | Phase 2 report | M1 | 7.1–7.4 | Report in `reports/phase2/` |
| 8.3 | Presentation slides | M1 | 8.2 | Final deck in `reports/phase2/` |
| 8.4 | README results & reproducibility update | M1 | 7.1, 8.2 | Results table + repro steps current |

---

## 4. Progress Tracking

- Use the GitHub **Project board** with columns: `Backlog`, `In Progress`, `In Review`, `Done`.
- An issue moves to `In Progress` only when its dependencies are `Done`.
- The PM (`M1`) reviews the board at the start of each work session and unblocks
  `blocked` issues.

## 5. Definition of Done (global)

1. Code merged to `main` via reviewed PR.
2. `make test` passes; new logic has unit tests.
3. Artifacts written to the canonical paths in §2.
4. `ROADMAP.md` conventions unchanged (or intentionally updated in the same PR).
5. Issue acceptance criteria verified and checked off.
