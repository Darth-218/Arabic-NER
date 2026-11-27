# Arabic Named Entity Recognition: A Comparative Study

## 📝 Project Overview
This repository contains the code, data processing scripts, and results for the Arabic Named Entity Recognition (NER) project, developed for AIS-411, Natural Language Processing. 

Following the course guidelines for Phase 1 and Phase 2, this project is structured as a **Comparative Study**. Our goal is to identify Persons (`PER`), Organizations (`ORG`), and Locations (`LOC`) in Arabic text using the WikiANN dataset. To fulfill the requirement that each team member implements a distinct approach, we compare classical machine learning methods against multilingual and Arabic-specific Transformer architectures.

> **Planning & conventions:** see [`ROADMAP.md`](ROADMAP.md) for the authoritative
> directory structure, shared interfaces, evaluation schema, and the issue breakdown
> each member is responsible for. All work must follow that Canonical Contract.

## 📊 Dataset
*   **Source:** `unimelb-nlp/wikiann` (Hugging Face Datasets) - Arabic (`ar`)
*   **Splits:** 20,000 train, 10,000 validation, 10,000 test sequences.
*   **Labels (BIO Tagging):** 
    *   `B-PER`, `I-PER` (Person)
    *   `B-ORG`, `I-ORG` (Organization)
    *   `B-LOC`, `I-LOC` (Location)
    *   `O` (Outside)

## 👥 Team Members & Assigned Approaches
To satisfy the Phase 1 requirement that *each team member implements one approach*, our comparative study is divided as follows:

*   **Member 1 — Project Manager:** Data Preprocessing, EDA & **Classical Approach** (Conditional Random Fields / BiLSTM-CRF) — [@Darth-218](https://github.com/Darth-218)
*   **Member 2:** **Multilingual Transformer** (mBERT) — Jannah Ashraf ([@jannahashraf](https://github.com/jannahashraf))
*   **Member 3:** **Arabic-Specific Transformer** (AraBERT) — Omar Hesham ([@OmarHesham2356](https://github.com/OmarHesham2356))
*   **Member 4:** **Advanced Arabic Transformer** (MARBERT) — Ahmed Maboud ([@ahmedmaboud](https://github.com/ahmedmaboud))

## 📂 Repository Structure
The full canonical layout is defined in [`ROADMAP.md` §2.1](ROADMAP.md#21-canonical-repository-structure).
Summary:
```text
├── data/                    # Dataset loader + EDA (Member 1)
├── src/
│   ├── common/              # Labels, seeding, logging (shared source of truth)
│   ├── data_utils.py        # Tokenization + sub-word label alignment
│   ├── evaluate.py          # seqeval metrics + per-class report
│   ├── results.py           # Unified metrics writer + README table
│   ├── compare.py           # Cross-model comparison
│   ├── approach_1_crf/      # Classical ML (Member 1)
│   ├── approach_2_mbert/    # Multilingual transformer (Member 2)
│   ├── approach_3_arabert/  # Arabic transformer (Member 3)
│   └── approach_4_marbert/  # Advanced Arabic transformer (Member 4)
├── configs/                 # One YAML per approach
├── results/                 # metrics/, predictions/, plots/
├── reports/                 # eda/, analysis/, phase1/, phase2/
├── tests/                   # Unit tests for shared modules
├── requirements.txt
├── ROADMAP.md               # Canonical contract + issue breakdown
└── README.md
```

## 🚀 Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Darth-218/Arabic-NER.git
   cd Arabic-NER
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

   # data-only first (datasets/pandas/pyarrow)
   pip install -r requirements-data.txt

   # full environment (modeling): choose one
   pip install -r requirements-cpu.txt   # CPU torch
   pip install -r requirements-gpu.txt   # CUDA torch
   ```

3. Download the dataset (cached under `data/raw/`, git-ignored):
   ```python
   from datasets import load_dataset
   data = load_dataset("unimelb-nlp/wikiann", "ar", cache_dir="data/raw")
   ```

## 💻 Usage

Each member is responsible for training and evaluating their specific model. Please use the corresponding commands below:

*   **Member 1 (CRF):** 
    ```bash
    python src/approach_1_crf/train.py
    ```
*   **Member 2 (mBERT):** 
    ```bash
    python src/approach_2_mbert/train.py --epochs 3 --batch_size 16
    ```
*   **Member 3 (AraBERT):** 
    ```bash
    python src/approach_3_arabert/train.py --epochs 3 --batch_size 16
    ```
*   **Member 4 (MARBERT):** 
    ```bash
    python src/approach_4_marbert/train.py --epochs 3 --batch_size 16
    ```

## 📈 Comparative Results (Phase 2)
*(This section will be populated during Phase 2 after hyperparameter optimization and evaluation using `seqeval`)*

| Approach | Model | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Classical | CRF | 0.8509 | 0.7771 | 0.8123 |
| Multilingual | mBERT | - | - | - |
| Arabic Transformer | AraBERT | - | - | - |
| Advanced Arabic | MARBERT | 0.8889 | 0.8960 | 0.8924 |
