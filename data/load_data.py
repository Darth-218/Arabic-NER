"""Canonical WikiANN-ar loader (ROADMAP.md §2.3).

All approaches must load the dataset through `load_arabic_wikiann`; no member
may call `datasets.load_dataset` directly. The returned splits always expose
only `tokens` (List[str]) and `ner_tags` (List[int], values from `LABEL2ID`).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in (None, ""):  # allow `python data/load_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import DATA_DIR, LABEL2ID, ensure_dirs

DATASET_NAME = "unimelb-nlp/wikiann"
DATASET_CONFIG = "ar"
EXPECTED_SPLITS = {"train": 20_000, "validation": 10_000, "test": 10_000}
CANONICAL_COLUMNS = ("tokens", "ner_tags")


def label_mapping(source_names: Iterable[str]) -> dict[int, int]:
    """Map WikiANN label ids onto canonical ids by NAME, never by position."""
    mapping: dict[int, int] = {}
    for source_id, name in enumerate(source_names):
        if name not in LABEL2ID:
            raise ValueError(
                f"Unknown WikiANN label {name!r}; canonical labels are {list(LABEL2ID)}"
            )
        mapping[source_id] = LABEL2ID[name]
    return mapping


def canonicalize_split(split: Any, mapping: Mapping[int, int]) -> Any:
    """Keep canonical columns and remap `ner_tags`; drop `langs`/`spans`."""

    def to_canonical(batch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "tokens": batch["tokens"],
            "ner_tags": [
                [mapping[int(tag)] for tag in sequence]
                for sequence in batch["ner_tags"]
            ],
        }

    extra_columns = [col for col in split.column_names if col not in CANONICAL_COLUMNS]
    return split.map(to_canonical, batched=True, remove_columns=extra_columns)


def _validate_split(split: Any, name: str, expected: int, *, strict: bool) -> None:
    actual = len(split)
    if actual != expected:
        message = f"{name} has {actual} sequences, expected {expected}"
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)

    tokens_column = split["tokens"]
    tags_column = split["ner_tags"]
    for index, (tokens, tags) in enumerate(zip(tokens_column, tags_column)):
        if len(tokens) != len(tags):
            raise ValueError(
                f"{name}[{index}]: {len(tokens)} tokens but {len(tags)} tags"
            )
        if not tokens:
            raise ValueError(f"{name}[{index}] is an empty sequence")


def load_arabic_wikiann(
    cache_dir: str | Path | None = None, *, strict: bool = True
) -> Any:
    """Return the canonical WikiANN-ar `DatasetDict`.

    Splits: 'train' (20k), 'validation' (10k), 'test' (10k).
    Columns: 'tokens': List[str], 'ner_tags': List[int] using LABEL2ID.

    `strict=True` (default) fails fast when the upstream split sizes drift;
    smoke tests that intentionally subsample do so *after* loading, so they are
    unaffected. Pass `strict=False` to downgrade size mismatches to warnings.
    """
    from datasets import DatasetDict, load_dataset  # lazy: import side-effect free

    ensure_dirs()
    raw = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        cache_dir=str(Path(cache_dir) if cache_dir is not None else DATA_DIR / "raw"),
    )
    mapping = label_mapping(raw["train"].features["ner_tags"].feature.names)

    dataset = DatasetDict(
        {name: canonicalize_split(raw[name], mapping) for name in raw}
    )

    for name, expected in EXPECTED_SPLITS.items():
        if name not in dataset:
            raise ValueError(f"WikiANN ar is missing the {name!r} split")
        _validate_split(dataset[name], name, expected, strict=strict)
    return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load/cache the WikiANN ar dataset")
    parser.add_argument("--cache_dir", default=None, help="default: data/raw")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="warn instead of failing when an expected split size changes",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    splits = load_arabic_wikiann(
        cache_dir=arguments.cache_dir, strict=not arguments.no_strict
    )
    for name, part in splits.items():
        print(f"{name}: {len(part)} sequences, columns={part.column_names}")
