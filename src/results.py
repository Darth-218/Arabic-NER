"""TEMPORARY PLACEHOLDER — OWNED BY MEMBER 1 (PM).

Do not treat this as finished M1 work. It exists only so Member 4's MARBERT
pipeline can write canonical metrics before Epic 2 is merged. Member 1 must
review/replace this file; the JSON shape below is the hard contract from
ROADMAP.md §2.5 and must not change.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):  # allow `python src/results.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import APPROACHES, METRICS_DIR, PROJECT_ROOT, ensure_dirs

ENTITY_NAMES = ("PER", "ORG", "LOC")
TABLE_LABELS = {
    "crf": ("Classical", "CRF"),
    "mbert": ("Multilingual", "mBERT"),
    "arabert": ("Arabic Transformer", "AraBERT"),
    "marbert": ("Advanced Arabic", "MARBERT"),
}


def _empty_entity() -> dict[str, float | int]:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def write_metrics(
    *,
    model: str,
    split: str,
    precision: float,
    recall: float,
    f1: float,
    per_entity: Mapping[str, Mapping[str, float | int]] | None = None,
    config: Mapping[str, Any] | None = None,
    approach: str | None = None,
    owner: str | None = None,
    out: str | Path | None = None,
) -> Path:
    """Write one canonical `results/metrics/<model>.json` document."""
    ensure_dirs()
    entities = {
        name: {
            **_empty_entity(),
            **{
                key: (
                    int(value)
                    if key == "support"
                    else float(value)
                )
                for key, value in (per_entity or {}).get(name, {}).items()
                if key in _empty_entity()
            },
        }
        for name in ENTITY_NAMES
    }
    payload = {
        "model": model,
        "approach": approach or APPROACHES.get(model, model),
        "owner": owner or "unknown",
        "split": split,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "per_entity": entities,
        "config": dict(config or {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(out) if out is not None else METRICS_DIR / f"{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def results_table(metrics_dir: Path | None = None) -> str:
    """Render the README comparison table from whatever metrics exist."""
    directory = metrics_dir or METRICS_DIR
    lines = [
        "| Approach | Model | Precision | Recall | F1-Score |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for model, (approach_label, display_name) in TABLE_LABELS.items():
        path = directory / f"{model}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            values = tuple(f"{float(data[key]):.4f}" for key in ("precision", "recall", "f1"))
        else:
            values = ("-", "-", "-")
        lines.append(
            f"| {approach_label} | {display_name} | "
            f"{values[0]} | {values[1]} | {values[2]} |"
        )
    return "\n".join(lines)


def update_readme_table(readme: str | Path | None = None) -> Path:
    """Regenerate the four data rows of the README's results table in place."""
    path = Path(readme) if readme is not None else PROJECT_ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\| Approach \| Model \| Precision \| Recall \| F1-Score \|\n"
        r"\| :--- \| :--- \| :--- \| :--- \| :--- \|\n"
        r"(?:\|.*\n)+"
    )
    updated, count = pattern.subn(results_table() + "\n", text)
    if count != 1:
        raise ValueError(f"Could not find exactly one results table in {path}")
    path.write_text(updated, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(results_table())
