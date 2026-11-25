"""Canonical contract tests for the shared configuration."""

from src.common.config import (
    APPROACHES,
    ID2LABEL,
    IGNORE_INDEX,
    LABEL2ID,
    LABELS,
    MODEL_NAMES,
)


def test_labels_order_is_canonical():
    assert LABELS == ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def test_label_id_bijection():
    assert len(LABEL2ID) == len(LABELS)
    for idx, label in enumerate(LABELS):
        assert LABEL2ID[label] == idx
        assert ID2LABEL[idx] == label


def test_ignore_index():
    assert IGNORE_INDEX == -100
    assert IGNORE_INDEX not in ID2LABEL


def test_model_names_and_approaches():
    assert MODEL_NAMES == ["crf", "mbert", "arabert", "marbert"]
    assert set(MODEL_NAMES) == set(APPROACHES)
