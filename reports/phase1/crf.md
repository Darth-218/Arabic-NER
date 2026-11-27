# Classical CRF — Phase 1 methods, results, and errors

Owner: Member 1 (`@Darth-218`). Artifacts: `results/metrics/crf.json`,
`results/metrics/crf_validation.json`, `results/predictions/crf_{test,validation}.jsonl`,
`results/plots/crf_*`, `results/models/crf/` (git-ignored checkpoints).

## Method

- **Model:** linear-chain conditional random field (`sklearn-crfsuite`, L-BFGS,
  `all_possible_transitions=true`), with one label per original WikiANN word.
- **Input:** canonical loader `load_arabic_wikiann()` (20k/10k/10k). Arabic
  normalization (diacritics, tatweel, alef/ya variants) is applied to **model
  input only**; prediction JSONL echoes the original surface forms.
- **Features** (`src/approach_1_crf/features.py`): token, lowercase token,
  1–3 character prefixes/suffixes, length bucket, coarse word shape, Arabic /
  digit / punctuation flags, first/last position, previous and next word and
  shape, and a token-level gazetteer built from **train B- spans only**
  (7,482 entries). BOS/EOS sentinels are used at sentence boundaries.
- **Baseline floor:** all-O predictions score **P/R/F1 = 0.0000**
  (`results/metrics/baseline.json`); the CRF clears this floor by +0.8123 F1.
- **BiLSTM-CRF (issue #17): out of scope.** The linear-chain CRF is the
  designated classical Phase 1 baseline; the three transformer approaches
  provide the neural comparison. Implementing a separate BiLSTM-CRF would add
  a fifth model without improving the four-way comparison.

## Hyperparameter selection

Validation F1 from the four-fit grid (`configs/crf.yaml`, seed 42,
`max_iterations=100`):

| c1 | c2 | Validation F1 |
| ---: | ---: | ---: |
| 0.1 | 0.05 | **0.8146** |
| 0.1 | 0.1 | 0.8145 |
| 0.5 | 0.05 | 0.8081 |
| 0.5 | 0.1 | 0.8046 |

Selected: **c1 = 0.1, c2 = 0.05**. Lower L1 regularization wins narrowly;
the differences are small (≤0.01 F1), so the choice is not fragile. The best
model is saved as `results/models/crf/model.joblib` alongside the saved config
and gazetteer.

## Results (seqeval, entity-level micro average)

| Split | Precision | Recall | F1 | PER F1 | ORG F1 | LOC F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8544 | 0.7784 | 0.8146 | 0.8250 | 0.8031 | 0.8152 |
| **Test** | **0.8509** | **0.7771** | **0.8123** | **0.8117** | **0.8088** | **0.8164** |
| Baseline (all-O, test) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Canonical reported number: **CRF test F1 = 0.8123**.

## Confusion notes

Token-level diagnostics (`results/plots/crf_classification_report.txt`) show
accuracy 0.91 but a precision/recall gap: precision ≈ 0.85 vs recall ≈ 0.78.
The model is conservative — `O` recall is 0.97, while `B-PER`/`B-ORG`/`B-LOC`
recall is 0.80–0.82. Continuation tokens (`I-*`) are easier (F1 0.88–0.91)
because the span has already been started. ORG is the weakest entity type
(F1 0.8088), often confused with LOC in sports/tournament names.

## Error examples (test split)

1. **Missed PER spans** — row 39: `... بترو بوروشنكو ... يوليا تيموشينكو .`
   Gold marks both people as `B-PER I-PER`; the CRF labels both as `I-ORG`
   because they follow `حزب كل الأوكرانيين`. Row 41 (`إلهام علييف الرئيس الحالي`)
   is entirely `O`.
2. **ORG → LOC confusion** — row 38: `- بطل كوبا أمريكا 1995`; gold `B-ORG I-ORG I-ORG`,
   prediction `B-LOC I-LOC I-LOC`. Row 113 (`تحويل ديل أور نو ديل ...`) shows
   the same pattern inside a redirect-style sentence.
3. **False ORG at span boundary** — row 25: `شركة يونيفرسال استوديو ...`;
   gold starts `B-ORG` at `يونيفرسال`, prediction starts one token earlier at
   `شركة` and extends the span by one token.

## Limitations

`reports/eda/data_quality.md` documents 1,454 exact test↔train duplicate
sequences in the official WikiANN splits. All four approaches are scored on the
same official test split for comparability; a de-duplicated robustness check is
planned under Epic 7. These F1 values should therefore be read as *relative*
comparisons, not as a fully unseen-test estimate.

## Reproduction

```bash
make train MODEL=crf
python src/approach_1_crf/predict.py --split test
python src/evaluate.py --model crf --config configs/crf.yaml
python src/compare.py
```
