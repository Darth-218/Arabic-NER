"""Tests for tokenization helpers and dataloader contract (issues 1.3–1.5)."""

from __future__ import annotations

from src.common.config import IGNORE_INDEX, LABEL2ID
from src.data_utils import (
    build_dataloaders,
    make_prediction_batches,
    make_token_classification_dataloader,
    tokenize_batch,
)
from tests._stubs import StubTokenizer


def test_contract_alias_matches_long_name() -> None:
    assert build_dataloaders is make_token_classification_dataloader


def test_tokenize_batch_aligns_labels_and_keeps_word_ids() -> None:
    batch = {
        "tokens": [["أحمد", "في", "القاهرة"]],
        "ner_tags": [[LABEL2ID["B-PER"], LABEL2ID["O"], LABEL2ID["B-LOC"]]],
    }
    encoded = tokenize_batch(batch, StubTokenizer(), max_len=128)
    assert encoded["word_ids"] == [[0, 0, 1, 2, 2]]
    assert encoded["labels"] == [
        [LABEL2ID["B-PER"], IGNORE_INDEX, LABEL2ID["O"], LABEL2ID["B-LOC"], IGNORE_INDEX]
    ]


def test_tokenize_batch_without_labels() -> None:
    encoded = tokenize_batch(
        {"tokens": [["شركة"]], "ner_tags": [[LABEL2ID["B-ORG"]]]},
        StubTokenizer(),
        with_labels=False,
    )
    assert "labels" not in encoded


def test_prediction_batches_pad_and_keep_word_ids() -> None:
    batches = list(
        make_prediction_batches(
            [["أحمد", "في", "القاهرة"], ["شركة"]],
            StubTokenizer(),
            batch_size=2,
            max_len=128,
        )
    )
    assert len(batches) == 1
    batch = batches[0]
    assert batch["ids"] == [0, 1]
    assert batch["tokens"][0] == ["احمد", "في", "القاهرة"]
    assert batch["word_ids"][0] == [0, 0, 1, 2, 2]
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert batch["attention_mask"][1][-1] == 0
