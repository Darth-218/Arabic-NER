# Results

Generated artifacts (never hand-edited):

- `metrics/<model>.json` — canonical metrics, schema in `ROADMAP.md` §2.5
- `predictions/<model>_<split>.jsonl` — one JSON object per line: `{id, tokens, tags}`
- `plots/` — figures for reports
- `models/` — checkpoints (git-ignored)

Model identifiers: `crf`, `mbert`, `arabert`, `marbert` (plus `baseline`).

## Checkpoint policy (issue 0.4)

- **Location:** `results/models/<model>/<run-tag>/`, where `<run-tag>` is a short
  config hash or UTC timestamp that uniquely identifies a training run.
- **Default output:** each `train.py` writes to `results/models/<model>` unless
  `--output_dir` is given; the run-tag directory lives underneath it.
- **Loading:** `predict.py --model_dir results/models/<model>/<run-tag>` loads the
  tokenizer and model with `from_pretrained` from that directory.
- **Never committed:** `results/models/` is git-ignored. Share large checkpoints
  via the external store agreed in the roadmap, not via git.
- **Reproducibility:** the run's config and seed travel with the metrics JSON
  (`config` field), so any checkpoint can be traced back to its settings.
