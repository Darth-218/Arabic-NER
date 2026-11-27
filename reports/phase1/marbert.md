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
- Normalization changes the **model input only**; prediction JSONLs echo the
  original WikiANN surface forms (ROADMAP §2.4).

## 4. Training setup

- Checkpoint: `UBC-NLP/MARBERT`, sequence length 128 (matching pre-training).
  A truncation audit found **0/10,000 test** and **0/10,000 validation**
  sequences over 128 subword positions (longest encodings 87 and 101), so word
  coverage is 100%. All transformer approaches should keep this same `max_len`
  and `truncation=True` policy for a fair comparison.
- Optimizer setup via Hugging Face `Trainer`: learning rate 2e-5, 3 epochs,
  batch size 16, warmup 6%, weight decay 0.01, seed 42.
- Best checkpoint is chosen by validation F1; final numbers are computed on the
  test split with `seqeval` through the shared `src/evaluate.py`.

## 5. Results

Numbers below are read from `results/metrics/marbert.json`, produced by
`train.py → predict.py → evaluate.py` on the full 20k/10k/10k WikiANN splits.
The saved model is the final epoch checkpoint (`checkpoint-3750`): a temporary
evaluator bug initially made in-training F1 read as 0, so all checkpoints were
re-scored after the fix and the best one was promoted to the canonical path.
See `reports/analysis/marbert_error_analysis.md` for the full breakdown.

Validation numbers come from `results/metrics/marbert_validation.json`; test
numbers from `results/metrics/marbert.json`.

| Split | Precision | Recall | F1 | PER F1 | ORG F1 | LOC F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| validation | 0.8939 | 0.8999 | 0.8969 | 0.9195 | 0.8579 | 0.9114 |
| test | 0.8889 | 0.8960 | 0.8924 | 0.9100 | 0.8573 | 0.9087 |

## 6. Error analysis

See `reports/analysis/marbert_error_analysis.md`. The main hypothesis to test
is the domain gap: MARBERT learned from tweets, while WikiANN is Wikipedia
text, so formal multi-word proper names may be harder than colloquial Arabic.

## 7. Reproducibility

```bash
pip install -r requirements-cpu.txt   # or requirements-gpu.txt
python src/approach_4_marbert/train.py --config configs/marbert.yaml
python src/approach_4_marbert/predict.py --model_dir results/models/marbert \
  --split test --out results/predictions/marbert_test.jsonl
python src/evaluate.py --model marbert --split test --config configs/marbert.yaml
make test
make compare   # adds MARBERT's row to the root README table
```
