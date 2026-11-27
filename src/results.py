"""Canonical results writer and README table generator (ROADMAP.md §2.5).

`results/metrics/<model>.json` is always the **test** result; non-test splits
default to `<model>_<split>.json` so a validation run can never silently
overwrite the canonical test metrics (PR #43 follow-up).
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

from src.common.config import APPROACHES, METRICS_DIR, MODEL_NAMES, PROJECT_ROOT, ensure_dirs

ENTITY_NAMES = ("PER", "ORG", "LOC")
SPLITS = ("train", "validation", "test")
ALLOWED_MODELS = frozenset([*MODEL_NAMES, "baseline"])
TABLE_LABELS = {
    "crf": ("Classical", "CRF"),
    "mbert": ("Multilingual", "mBERT"),
    "arabert": ("Arabic Transformer", "AraBERT"),
    "marbert": ("Advanced Arabic", "MARBERT"),
}
REQUIRED_KEYS = frozenset(
    {
        "model",
        "approach",
        "owner",
        "split",
        "precision",
        "recall",
        "f1",
        "per_entity",
        "config",
        "timestamp",
    }
)


def _empty_entity() -> dict[str, float | int]:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def validate_metrics(payload: Mapping[str, Any], source: str = "metrics") -> dict[str, Any]:
    """Validate the hard metrics schema shared by every approach."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must be a JSON object")
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise ValueError(f"{source} is missing keys: {sorted(missing)}")
    if payload["model"] not in ALLOWED_MODELS:
        raise ValueError(f"{source}: unknown model {payload['model']!r}")
    if payload["split"] not in SPLITS:
        raise ValueError(f"{source}: split must be one of {SPLITS}")
    for key in ("precision", "recall", "f1"):
        value = payload[key]
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{source}: {key} must be a float in [0, 1]")
    per_entity = payload["per_entity"]
    if not isinstance(per_entity, Mapping):
        raise ValueError(f"{source}: per_entity must be a mapping")
    for name in ENTITY_NAMES:
        scores = per_entity.get(name)
        if not isinstance(scores, Mapping):
            raise ValueError(f"{source}: per_entity is missing {name!r}")
        for key in ("precision", "recall", "f1"):
            if key not in scores:
                raise ValueError(f"{source}: per_entity[{name!r}] is missing {key!r}")
        if "support" not in scores:
            raise ValueError(f"{source}: per_entity[{name!r}] is missing 'support'")
    if not isinstance(payload["config"], Mapping):
        raise ValueError(f"{source}: config must be a mapping")
    return dict(payload)


def read_metrics(path: str | Path) -> dict[str, Any]:
    """Read and validate one canonical metrics document."""
    source = Path(path)
    return validate_metrics(
        json.loads(source.read_text(encoding="utf-8")), source=str(source)
    )


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
    """Write one canonical metrics document.

    Defaults to `results/metrics/<model>.json` for the test split and
    `results/metrics/<model>_<split>.json` for every other split.
    """
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
    validate_metrics(payload)
    if out is not None:
        path = Path(out)
    else:
        filename = f"{model}.json" if split == "test" else f"{model}_{split}.json"
        path = METRICS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def results_table(metrics_dir: Path | None = None) -> str:
    """Render the README comparison table from whatever (valid) metrics exist."""
    directory = metrics_dir or METRICS_DIR
    lines = [
        "| Approach | Model | Precision | Recall | F1-Score |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for model, (approach_label, display_name) in TABLE_LABELS.items():
        path = directory / f"{model}.json"
        if path.exists():
            data = read_metrics(path)
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
