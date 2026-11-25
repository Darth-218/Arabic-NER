"""Central project configuration.

This module is the single source of truth for the canonical label schema,
model identifiers and artifact paths. Do not redefine labels or model names
elsewhere; import them from here.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CONFIGS_DIR = PROJECT_ROOT / "configs"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
IGNORE_INDEX = -100

MODEL_NAMES = ["crf", "mbert", "arabert", "marbert"]

APPROACHES = {
    "crf": "classical",
    "mbert": "multilingual_transformer",
    "arabert": "arabic_transformer",
    "marbert": "advanced_arabic_transformer",
}

DEFAULT_SEED = 42


def ensure_dirs() -> None:
    """Create the canonical artifact directories if they do not exist."""
    for path in (
        RAW_DATA_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        PLOTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
