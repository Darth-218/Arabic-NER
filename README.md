# Arabic Named Entity Recognition: A Comparative Study

## 📝 Project Overview
This repository contains the code, data processing scripts, and results for the Arabic Named Entity Recognition (NER) project, developed for AIS-411, Natural Language Processing. 

Following the course guidelines for Phase 1 and Phase 2, this project is structured as a **Comparative Study**. Our goal is to identify Persons (`PER`), Organizations (`ORG`), and Locations (`LOC`) in Arabic text using the WikiANN dataset. To fulfill the requirement that each team member implements a distinct approach, we compare classical machine learning methods against multilingual and Arabic-specific Transformer architectures.

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

*   **[Member 1 Name]:** Data Preprocessing, EDA & **Classical Approach** (Conditional Random Fields / BiLSTM-CRF)
*   **[Member 2 Name]:** **Multilingual Transformer** (mBERT or XLM-RoBERTa)
*   **[Member 3 Name]:** **Arabic-Specific Transformer** (AraBERT or CAMeLBERT)
*   **[Member 4 Name]:** **Advanced Arabic Transformer** (MARBERT) & Project Manager

## 📂 Repository Structure
```text
├── data/                  # Dataset loading and EDA scripts (Member 1)
├── src/
│   ├── data_utils.py      # Sub-word token alignment logic
│   ├── approach_1_crf/    # Classical ML implementation (Member 1)
│   ├── approach_2_mbert/  # Multilingual transformer (Member 2)
│   ├── approach_3_arabert/# Arabic transformer (Member 3)
│   └── approach_4_marbert/# Advanced Arabic transformer (Member 4)
├── results/               # Unified evaluation metrics and F1-scores
├── reports/               # Phase 1 & Phase 2 reports & presentation
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/arabic-ner-comparative-study.git
   cd arabic-ner-comparative-study
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```python
   from datasets import load_dataset
   data = load_dataset("unimelb-nlp/wikiann", "ar")
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
| Classical | CRF | - | - | - |
| Multilingual | mBERT | - | - | - |
| Arabic Transformer | AraBERT | - | - | - |
| Advanced Arabic | MARBERT | - | - | - |
