"""Tests for the canonical WikiANN loader (issue 1.1, ROADMAP §2.3)."""

from __future__ import annotations

import pytest

from data.load_data import (
    EXPECTED_SPLITS,
    _validate_split,
    canonicalize_split,
    label_mapping,
)
from tests._stubs import FakeSplit


def test_expected_split_sizes_are_canonical() -> None:
    assert EXPECTED_SPLITS == {"train": 20_000, "validation": 10_000, "test": 10_000}


def test_label_mapping_is_by_name() -> None:
    mapping = label_mapping(["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])
    assert mapping == {index: index for index in range(7)}


def test_label_mapping_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown WikiANN label"):
        label_mapping(["O", "B-MISC"])


def test_canonicalize_split_remaps_and_drops_extra_columns() -> None:
    split = FakeSplit(
        tokens=[["القاهرة"]],
        ner_tags=[[5]],
        extra={"langs": ["ar"], "spans": ["LOC"]},
    )
    canonical = canonicalize_split(split, {5: 2})
    assert split.last_remove_columns == ["langs", "spans"]
    assert canonical.column_names == ["tokens", "ner_tags"]
    assert canonical["ner_tags"] == [[2]]


def test_split_size_mismatch_fails_in_strict_mode() -> None:
    split = FakeSplit(tokens=[["ا"]], ner_tags=[[0]])
    with pytest.raises(ValueError, match="expected 20000"):
        _validate_split(split, "train", 20_000, strict=True)


def test_split_size_mismatch_warns_when_not_strict() -> None:
    split = FakeSplit(tokens=[["ا"]], ner_tags=[[0]])
    with pytest.warns(UserWarning, match="expected 20000"):
        _validate_split(split, "train", 20_000, strict=False)


def test_token_tag_length_mismatch_is_rejected() -> None:
    split = FakeSplit(tokens=[["ا", "ب"]], ner_tags=[[0]])
    with pytest.raises(ValueError, match="tokens but"):
        _validate_split(split, "train", 1, strict=True)
