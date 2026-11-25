# Source

Shared code lives at the top level; each member owns exactly one approach folder.

- `common/` — labels, seeding, logging (single source of truth)
- `data_utils.py` — normalization, subword alignment, dataloaders (Member 1)
- `evaluate.py` — seqeval metrics and diagnostics (Member 1)
- `results.py` — canonical metrics writer and comparison table (Member 1)
- `compare.py` — cross-model aggregation (Member 1, PM)
- `approach_1_crf/` — Member 1
- `approach_2_mbert/` — Member 2
- `approach_3_arabert/` — Member 3
- `approach_4_marbert/` — Member 4

Every approach must expose `train.py` and `predict.py` and follow the CLI,
prediction format and metrics schema in `ROADMAP.md` §2.4–§2.5.
