# Results

Generated artifacts (never hand-edited):

- `metrics/<model>.json` — canonical metrics, schema in `ROADMAP.md` §2.5
- `predictions/<model>_<split>.jsonl` — one JSON object per line: `{id, tokens, tags}`
- `plots/` — figures for reports
- `models/` — checkpoints (git-ignored)

Model identifiers: `crf`, `mbert`, `arabert`, `marbert` (plus `baseline`).
