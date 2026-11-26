# Phase 1 Report — MARBERT (Member 4, issue 6.5)

**Owner:** Ahmed Maboud (`@ahmedmaboud`) · **Approach:** advanced Arabic transformer.

## 1. Method in plain terms

Named-entity recognition is a labelling task: given a sequence of Arabic words,
assign each word a BIO tag — `B-PER`/`I-PER` for people, `B-ORG`/`I-ORG` for
organizations, `B-LOC`/`I-LOC` for places, and `O` for everything else. `B`
means "beginning" and `I` means "inside" of the same entity.

MARBERT is a **pretrained** language model: before seeing any NER data it read
roughly 1 billion Arabic tweets and learned statistical patterns of Arabic
words, spelling and context. We replace its output layer with a small "tag
head" and fine-tune the whole network on WikiANN's 20,000 training sentences.
Because it has already learned Arabic, it needs far less labelled data than a
model trained from scratch.

## 2. Why this approach, and how it differs from the team

| Member | Model | What makes it different |
| :--- | :--- | :--- |
| M1 | CRF / BiLSTM-CRF | Classical ML: no pretrained language knowledge, relies on hand-designed features. |
| M2 | mBERT | One model for 100+ languages; broad but not Arabic-specialized. |
| M3 | AraBERT | Arabic-specific, trained mainly on news/MSA text. |
| **M4 (this work)** | **MARBERT** | Arabic-specific **and** dialect-aware (tweets), ~163M parameters, BERT-base architecture. |

This position lets the team ask a clear question: *does Arabic specialization
and dialect exposure help on a Wikipedia-style NER benchmark?* Only M1's
classical model and the shared evaluation harness make that comparison fair.

## 3. Data and preprocessing

- Data comes **only** from `data/load_data.py::load_arabic_wikiann()`
  (20k train / 10k validation / 10k test), never from a private copy.
- Labels are defined **only** in `src/common/config.py`; `train.py` and
  `predict.py` import them.
- Arabic normalization (diacritics, tatweel, alef and ya variants) is switched
  on in `configs/marbert.yaml` so spelling variants collapse to one form.
- A word can be split into several subwords by MARBERT's tokenizer. Only the
  **first** subword keeps the word's BIO label; the rest get `-100` and are
  ignored by the loss. At prediction time the first subword's guess is mapped
  back to one tag per original word, which is what the evaluator expects.

## 4. Training setup

- Checkpoint: `UBC-NLP/MARBERT`, sequence length 128 (matching pre-training).
- Optimizer setup via Hugging Face `Trainer`: learning rate 2e-5, 3 epochs,
  batch size 16, warmup 6%, weight decay 0.01, seed 42.
- Best checkpoint is chosen by validation F1; final numbers are computed on the
  test split with `seqeval` through the shared `src/evaluate.py`.

## 5. Results

_Pending the first full run. Fill from `results/metrics/marbert.json` after
`train.py → predict.py → evaluate.py`; do not type numbers by hand._

| Split | Precision | Recall | F1 | PER F1 | ORG F1 | LOC F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| test | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

## 6. Error analysis

See `reports/analysis/marbert_error_analysis.md`. The main hypothesis to test
is the domain gap: MARBERT learned from tweets, while WikiANN is Wikipedia
text, so formal multi-word proper names may be harder than colloquial Arabic.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python src/approach_4_marbert/train.py --config configs/marbert.yaml
python src/approach_4_marbert/predict.py --model_dir results/models/marbert \
  --split test --out results/predictions/marbert_test.jsonl
python src/evaluate.py --model marbert --split test --config configs/marbert.yaml
make test
make compare   # adds MARBERT's row to the root README table
```
