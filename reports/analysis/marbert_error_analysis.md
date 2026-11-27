# MARBERT — Error Analysis (issue 6.4)

**Owner:** Member 4 (Ahmed Maboud) · **Status:** filled from the full test run (F1 0.8924).

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
| Overall precision | 0.8889 |
| Overall recall | 0.8960 |
| Overall F1 | 0.8924 |
| PER F1 | 0.9100 |
| ORG F1 | 0.8573 |
| LOC F1 | 0.9087 |

PER is the easiest class and ORG the hardest. The ORG gap is driven mostly by
long or transliterated organization names and by ORG/LOC confusion, not by a
failure to detect short, common organizations.

## Truncation coverage and comparison fairness

Computed with the same MARBERT fast tokenizer, the normalization settings in
`configs/marbert.yaml`, and `truncation=True, max_length=128`. "Covered words"
means `max(word_ids) + 1`; any tail beyond that would be emitted as `O` by
`align_predictions_to_words`. Raw counts are also saved in
`reports/analysis/marbert_truncation.json`.

| Split | Sequences | Encodings > 128 | Sequences losing words | Word coverage | Longest full encoding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| test | 10,000 | 0 (0.0%) | 0 (0.0%) | 64,347 / 64,347 (100.0%) | 87 |
| validation | 10,000 | 0 (0.0%) | 0 (0.0%) | 64,291 / 64,291 (100.0%) | 101 |

No sequence reaches the limit, so no tail is labelled `O` because of truncation,
and moving to `max_len=256` or a sliding window would not change these results.
The agreed fairness policy is: **all transformer approaches use the same
`max_len=128` with `truncation=True`**. At review time only `configs/marbert.yaml`
exists on this branch; M2/mBERT and M3/AraBERT must copy the same setting when
their configs land.

## Prediction surface forms

`predict.py` writes the **original** WikiANN tokens, not the normalized strings
fed to the model (ROADMAP §2.4). Verification: all 10,000 test prediction rows
have `tokens` identical to the gold `tokens` (0 mismatches). Normalization
therefore affects model input only.

## Verification evidence

- Library versions: `torch==2.15.0`, `transformers==5.24.1`,
  `datasets==5.2.0`, `seqeval==1.2.2`.
- Prediction row count: 10,000 test and 10,000 validation JSONL records.
- seqeval test F1: **0.8924** (precision 0.8889, recall 0.8960); validation F1
  0.8969. Both are in the committed metrics JSON files.
- `evaluate.py` checks the prediction row count against the gold split and the
  tag length of every row; both full runs completed without a length-mismatch
  error, so there are **0 prediction/gold length mismatches**.

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

### What the run actually showed

The ten disagreements below (IDs 18–113 from `marbert_test.jsonl`) confirm
three of those hypotheses:

- **ORG → PER** for transliterated company names (`مومويرو كلوفر زد`,
  `جي بي مورجان تشايس`) — MARBERT treats the name as a person.
- **ORG ↔ LOC** for place names used as organizations (`غوتنبرغ`,
  `لإسرائيل`) and for a broken multi-token ORG (`غدة تحت الفك السفلي`), where
  the tail is relabelled LOC.
- **B- vs I-** errors inside multi-word names (`استوديو حملة`, `سبتة مليلة`,
  `أور نو ديل`) and one completely missed ORG (`بالنمو الأسي`).

## Example table (10 real disagreements)

| ID | Tokens | Gold tags | MARBERT tags | What went wrong |
| :--- | :--- | :--- | :--- | :--- |
| 18 | مومويرو كلوفر زد | B-ORG I-ORG I-ORG | B-PER I-PER I-PER | Transliterated company read as a person. |
| 25 | شركة يونيفرسال استوديو حملة التوزيع | O B-ORG I-ORG O O | O B-ORG O O O | `I-` dropped; multi-word ORG truncated. |
| 34 | تفكك الاتحاد السوفيتي هذه الأهداف | B-ORG I-ORG I-ORG O O | B-ORG I-ORG I-LOC O O | Last token of the ORG read as a location. |
| 50 | في الإرسال بالنمو الأسي . | O O B-ORG I-ORG O | O O O O O | Whole ORG missed. |
| 59 | مدينتي سبتة مليلة المتواجدتين في | O B-LOC B-LOC O O | O B-LOC I-LOC O O | Two adjacent cities merged into one LOC. |
| 68 | ' '' غوتنبرغ '' ' | O O B-LOC O O | O O B-ORG O O | City name classified as an organization. |
| 72 | بمواقفه الداعمة لإسرائيل . | O O B-LOC O | O O B-ORG O | Country name labelled ORG. |
| 76 | تقع غدة تحت الفك السفلي . | O B-ORG I-ORG I-ORG I-ORG O | O B-ORG O B-LOC I-LOC O | One ORG split; tail relabelled LOC. |
| 95 | جي بي مورجان تشايس | B-ORG I-ORG I-ORG I-ORG | B-PER I-PER I-PER I-PER | Company name read as a person. |
| 113 | أور نو ديل ( النسخة | I-ORG I-ORG B-ORG I-ORG I-ORG | I-ORG I-ORG I-ORG I-ORG I-ORG | `B-ORG` boundary missed inside a long name. |

## What to compare with teammates

- Copy the same examples into the shared `reports/analysis/` comparison (issue
  7.2) so M1 can see which errors are **unique to MARBERT** versus **shared by
  all four approaches**.
- M2's mBERT has broader language coverage but less Arabic specialization; M3's
  AraBERT is trained on news/MSA text; M1's CRF has no contextual understanding
  at all. Any pattern that appears in all four is probably a dataset/label
  problem, not a model problem.
