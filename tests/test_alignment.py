"""Tests for Arabic normalization and subword alignment (issues 1.3–1.4)."""

from __future__ import annotations

from src.common.config import IGNORE_INDEX, LABEL2ID
from src.data_utils import (
    align_labels,
    align_labels_with_tokens,
    align_predictions_to_words,
    decode_tags,
    encode_tags,
    normalize_arabic,
)


def test_contract_alias_matches_long_name() -> None:
    assert align_labels is align_labels_with_tokens


def test_normalize_arabic_variants() -> None:
    assert normalize_arabic("اَلْعَرَبِيَّة") == "العربية"
    assert normalize_arabic("أحمدى") == "احمدي"
    assert normalize_arabic("أحمدى", normalize_alef=False, normalize_ya=False) == "أحمدى"
    assert normalize_arabic("مـحـمـد") == "محمد"


def test_align_labels_only_first_subword() -> None:
    labels = [LABEL2ID["B-PER"], LABEL2ID["O"], LABEL2ID["B-LOC"]]
    word_ids: list[int | None] = [0, 0, 1, None, 2, 2]
    assert align_labels_with_tokens(labels, word_ids) == [
        labels[0],
        IGNORE_INDEX,
        labels[1],
        IGNORE_INDEX,
        labels[2],
        IGNORE_INDEX,
    ]


def test_align_labels_all_subwords() -> None:
    labels = [1, 2, 3]
    word_ids: list[int | None] = [0, 0, 1, None, 2]
    assert align_labels_with_tokens(labels, word_ids, label_all_subwords=True) == [
        1,
        1,
        2,
        IGNORE_INDEX,
        3,
    ]


def test_align_labels_rejects_out_of_range_word_ids() -> None:
    assert align_labels_with_tokens([1], [5]) == [IGNORE_INDEX]


def test_align_predictions_to_words_uses_first_subword() -> None:
    assert align_predictions_to_words([1, 5, 2, 7, 3], [0, 0, 1, None, 2], num_words=3) == [
        1,
        2,
        3,
    ]


def test_truncated_word_falls_back_to_O() -> None:
    result = align_predictions_to_words([2], [0], num_words=3)
    assert result == [2, LABEL2ID["O"], LABEL2ID["O"]]


def test_label_roundtrip() -> None:
    tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    assert decode_tags(encode_tags(tags)) == tags
