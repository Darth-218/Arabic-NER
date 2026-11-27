"""TEMPORARY PLACEHOLDER — OWNED BY MEMBER 1 (PM).

Do not treat this as finished M1 work. It exists only so Member 4's MARBERT
pipeline can run locally before Epic 1 is merged. Member 1 must review/replace
this file; the public function signature below is the canonical contract from
ROADMAP.md §2.3 and must not change.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python data/load_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import DATA_DIR, LABEL2ID, ensure_dirs

DATASET_NAME = "unimelb-nlp/wikiann"
DATASET_CONFIG = "ar"
EXPECTED_SPLITS = {"train": 20_000, "validation": 10_000, "test": 10_000}


def load_arabic_wikiann(cache_dir: str | None = None) -> Any:
    """Return the canonical WikiANN-ar DatasetDict.

    Splits: 'train' (20k), 'validation' (10k), 'test' (10k).
    Columns: 'tokens': List[str], 'ner_tags': List[int] using LABEL2ID.
    """
    from datasets import load_dataset  # lazy: keeps import side-effect free

    ensure_dirs()
    raw = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        cache_dir=cache_dir or str(DATA_DIR / "raw"),
    )

    # WikiANN uses the same BIO names as our canonical schema, but map by NAME
    # rather than by position so a different label order can never silently
    # corrupt Member 1's / Member 4's training targets.
    source_names = list(raw["train"].features["ner_tags"].feature.names)
    mapping: dict[int, int] = {}
    for source_id, name in enumerate(source_names):
        if name not in LABEL2ID:
            raise ValueError(
                f"Unknown WikiANN label {name!r}; canonical labels are {list(LABEL2ID)}"
            )
        mapping[source_id] = LABEL2ID[name]

    def to_canonical(batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "tokens": batch["tokens"],
            "ner_tags": [
                [mapping[int(tag)] for tag in sequence]
                for sequence in batch["ner_tags"]
            ],
        }

    extra_columns = [
        col for col in raw["train"].column_names if col not in ("tokens", "ner_tags")
    ]
    dataset = raw.map(to_canonical, batched=True, remove_columns=extra_columns)

    for split, expected in EXPECTED_SPLITS.items():
        if split not in dataset:
            raise ValueError(f"WikiANN ar is missing the {split!r} split")
        if len(dataset[split]) != expected:
            warnings.warn(
                f"{split} has {len(dataset[split])} sequences, expected {expected}",
                stacklevel=2,
            )
    return dataset


if __name__ == "__main__":
    splits = load_arabic_wikiann()
    for name, part in splits.items():
        print(f"{name}: {len(part)} sequences, columns={part.column_names}")
