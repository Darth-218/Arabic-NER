"""Offline tests for the CRF/baseline pipeline helpers."""

from __future__ import annotations

from typing import Any

from src.approach_1_crf.predict import build_records, predict_tags
from src.approach_1_crf.train import build_gazetteer, candidates, resolve_config
from src.baseline import all_o_records
from src.common.config import LABEL2ID


def test_resolve_config_applies_cli_overrides() -> None:
    config = resolve_config(
        {"config": "configs/crf.yaml", "epochs": 25, "batch_size": 32, "c1": None, "c2": None, "tune": None},
        {"c2": 0.5},
    )
    assert config["max_iterations"] == 25
    assert config["c2"] == 0.5
    assert config["tune"] is True


def test_candidates_from_grid_and_single_setting() -> None:
    assert candidates({"c1": 0.1, "c2": 0.2, "tune": False}) == [(0.1, 0.2)]
    grid = candidates(
        {"c1": 0.1, "c2": 0.2, "tune": True, "tuning": {"c1": [0.1, 0.5], "c2": [0.05]}}
    )
    assert grid == [(0.1, 0.05), (0.5, 0.05)]


def test_gazetteer_uses_train_b_tags_only() -> None:
    dataset = {
        "train": {
            "tokens": [["أحمد", "في", "القاهرة"]],
            "ner_tags": [
                [LABEL2ID["B-PER"], LABEL2ID["O"], LABEL2ID["B-LOC"]]
            ],
        }
    }
    gazetteer = build_gazetteer(dataset, lambda tokens: tokens)
    assert gazetteer == {"أحمد": "PER", "القاهرة": "LOC"}


class AlwaysO:
    def predict(self, features):
        return [["O"] * len(sequence) for sequence in features]


def test_predict_tags_preserves_word_count() -> None:
    tags = predict_tags(
        AlwaysO(),
        [["أحمد", "في", "القاهرة"]],
        normalization={},
        gazetteer=None,
    )
    assert tags == [["O", "O", "O"]]


def test_build_records_uses_canonical_schema() -> None:
    records = build_records([["أحمد"]], [["B-PER"]])
    assert records == [{"id": 0, "tokens": ["أحمد"], "tags": ["B-PER"]}]


def test_all_o_records_from_fake_dataset() -> None:
    dataset: dict[str, Any] = {"test": {"tokens": [["ا", "ب"], ["ج"]]}}
    records = all_o_records(dataset, "test")
    assert [record["tags"] for record in records] == [["O", "O"], ["O"]]
    assert [record["id"] for record in records] == [0, 1]
