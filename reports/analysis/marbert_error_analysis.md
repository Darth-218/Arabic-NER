# MARBERT — Error Analysis (issue 6.4)

**Owner:** Member 4 (Ahmed Maboud) · **Status:** template ready, fill after the first real run.

## How to produce the evidence

```bash
python src/approach_4_marbert/train.py --config configs/marbert.yaml
python src/approach_4_marbert/predict.py --model_dir results/models/marbert --split test \
  --out results/predictions/marbert_test.jsonl
python src/evaluate.py --model marbert --split test --config configs/marbert.yaml
```

`evaluate.py` also writes `results/plots/marbert_confusion.json` and
`results/plots/marbert_classification_report.txt`. Paste the important rows
below; do **not** hand-edit `results/metrics/marbert.json` — it is generated.

## Headline numbers (fill from `results/metrics/marbert.json`)

| Metric | Value |
| :--- | :--- |
| Overall precision | _pending_ |
| Overall recall | _pending_ |
| Overall F1 | _pending_ |
| PER F1 | _pending_ |
| ORG F1 | _pending_ |
| LOC F1 | _pending_ |

## Confusion patterns to check first

MARBERT was pretrained on about **1 billion Arabic tweets**, so it is strong on
informal and dialectal Arabic. WikiANN, however, is built from Wikipedia — more
formal, more proper nouns, more long names. Expect this domain shift to show up
as:

1. **ORG vs LOC** —_city names used as organization names_ (e.g. a club or a
   ministry named after a place) and _organizations named after a city_.
2. **B- vs I- errors** — splitting one long Arabic entity into two, or merging
   two adjacent entities of the same type. Remember: `B-` starts an entity,
   `I-` continues it.
3. **Person names without context** — transliterated foreign names are often
   ambiguous without a title or verb around them.
4. **Normalization side effects** — `configs/marbert.yaml` strips diacritics and
   unifies alef/ya spellings. Check whether any errors are caused by a word
   being normalized into a different surface form.
5. **Truncation** — `max_len: 128` matches MARBERT's pre-training length, but a
   long sentence can push later entities off the end; these become `O`.

## Example table (fill with 5–10 real disagreements)

| # | Tokens | Gold tags | MARBERT tags | What went wrong |
| :--- | :--- | :--- | :--- | :--- |
| 1 | _pending_ | _pending_ | _pending_ | _pending_ |

## What to compare with teammates

- Copy the same examples into the shared `reports/analysis/` comparison (issue
  7.2) so M1 can see which errors are **unique to MARBERT** versus **shared by
  all four approaches**.
- M2's mBERT has broader language coverage but less Arabic specialization; M3's
  AraBERT is trained on news/MSA text; M1's CRF has no contextual understanding
  at all. Any pattern that appears in all four is probably a dataset/label
  problem, not a model problem.
