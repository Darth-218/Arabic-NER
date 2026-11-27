"""Tests for seqeval metrics and prediction validation (issues 2.1–2.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluate import compute_metrics, load_predictions, validate_tags


def test_compute_metrics_perfect_micro_and_per_entity() -> None:
    metrics = compute_metrics([["B-PER", "I-PER", "O"]], [["B-PER", "I-PER", "O"]])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["per_entity"]["PER"]["f1"] == 1.0


def test_compute_metrics_hand_computed_partial_match() -> None:
    # One PER span: predicted PER (correct) and a spurious LOC.
    metrics = compute_metrics([["B-PER", "B-LOC"]], [["B-PER", "O"]])
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["per_entity"]["LOC"]["support"] == 0


def test_compute_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="predicted tags vs"):
        compute_metrics([["B-PER"]], [["B-PER", "O"]])


def test_validate_tags_rejects_empty_and_unknown() -> None:
    with pytest.raises(ValueError, match="is empty"):
        validate_tags([[]], "predicted tags")
    with pytest.raises(ValueError, match="non-canonical"):
        validate_tags([["B-MISC"]], "predicted tags")


def test_load_predictions_validates_and_allows_out_of_order_ids(tmp_path: Path) -> None:
    path = tmp_path / "good.jsonl"
    rows = [
        {"id": 1, "tokens": ["ب"], "tags": ["O"]},
        {"id": 0, "tokens": ["ا"], "tags": ["O"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert [row["id"] for row in load_predictions(path)] == [1, 0]


def test_load_predictions_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": 0, "tokens": ["ا"], "tags": ["O"]}) for _ in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate ids"):
        load_predictions(path)


def test_load_predictions_rejects_non_contiguous_ids(tmp_path: Path) -> None:
    path = tmp_path / "gap.jsonl"
    path.write_text(
        json.dumps({"id": 1, "tokens": ["ا"], "tags": ["O"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous"):
        load_predictions(path)


def test_load_predictions_rejects_token_tag_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": 0, "tokens": ["ا", "ب"], "tags": ["O"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokens but"):
        load_predictions(path)
