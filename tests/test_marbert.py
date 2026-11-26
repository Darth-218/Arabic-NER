"""Offline unit tests for Member 4's MARBERT pipeline.

No network, dataset download, model download, torch or transformers required:
the pure helpers are tested with a stub tokenizer (see tests/README.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.approach_4_marbert.predict import build_prediction_records, write_jsonl
from src.approach_4_marbert.train import DEFAULTS, build_parser, resolve_config
from src.common.config import IGNORE_INDEX, LABEL2ID, PROJECT_ROOT
from src.data_utils import (
    align_labels_with_tokens,
    align_predictions_to_words,
    decode_tags,
    encode_tags,
    make_prediction_batches,
    normalize_arabic,
)
from src.results import write_metrics


class StubEncoding(dict):
    """Minimal stand-in for transformers' BatchEncoding."""

    def word_ids(self, index: int) -> list[int | None]:
        return list(self["word_ids"][index])


class StubTokenizer:
    """Splits words longer than 3 characters into two subwords."""

    pad_token_id = 0
    is_fast = True

    def __call__(self, texts: list[list[str]], **_: Any) -> StubEncoding:
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        word_ids: list[list[int | None]] = []
        for words in texts:
            ids: list[int] = []
            mask: list[int] = []
            wids: list[int | None] = []
            for word_index, word in enumerate(words):
                pieces = 2 if len(word) > 3 else 1
                for _ in range(pieces):
                    ids.append(10 + word_index)
                    mask.append(1)
                    wids.append(word_index)
            input_ids.append(ids)
            attention_mask.append(mask)
            word_ids.append(wids)
        return StubEncoding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids,
        )


def test_align_labels_only_first_subword() -> None:
    labels = [LABEL2ID["B-PER"], LABEL2ID["O"], LABEL2ID["B-LOC"]]
    word_ids: list[int | None] = [0, 0, 1, None, 2, 2]
    assert align_labels_with_tokens(labels, word_ids) == [
        labels[0], IGNORE_INDEX, labels[1], IGNORE_INDEX, labels[2], IGNORE_INDEX,
    ]


def test_align_labels_all_subwords() -> None:
    labels = [1, 2, 3]
    word_ids: list[int | None] = [0, 0, 1, None, 2]
    assert align_labels_with_tokens(labels, word_ids, label_all_subwords=True) == [
        1, 1, 2, IGNORE_INDEX, 3,
    ]


def test_align_predictions_to_words() -> None:
    predictions = [1, 5, 2, 7, 3]
    word_ids: list[int | None] = [0, 0, 1, None, 2]
    assert align_predictions_to_words(predictions, word_ids, num_words=3) == [1, 2, 3]


def test_truncated_word_falls_back_to_O() -> None:
    predictions = [2]
    word_ids: list[int | None] = [0]
    result = align_predictions_to_words(predictions, word_ids, num_words=3)
    assert result == [2, LABEL2ID["O"], LABEL2ID["O"]]


def test_normalize_arabic_variants() -> None:
    assert normalize_arabic("اَلْعَرَبِيَّة") == "العربية"
    assert normalize_arabic("أحمدى") == "احمدي"
    assert normalize_arabic("أحمدى", normalize_alef=False, normalize_ya=False) == "أحمدى"


def test_label_roundtrip() -> None:
    tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    assert decode_tags(encode_tags(tags)) == tags


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
    assert batch["attention_mask"][1][-1] == 0  # padding position


def test_build_prediction_records() -> None:
    records = build_prediction_records(
        ids=[0],
        tokens_per_example=[["احمد", "في", "القاهرة"]],
        predicted_ids_per_example=[[1, 0, 0, 5]],
        word_ids_per_example=[[0, 0, 1, 2, 2]],
    )
    assert records == [
        {
            "id": 0,
            "tokens": ["احمد", "في", "القاهرة"],
            "tags": ["B-PER", "O", "B-LOC"],
        }
    ]


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    path = write_jsonl([{"id": 0, "tokens": ["x"], "tags": ["O"]}], tmp_path / "p.jsonl")
    assert json.loads(path.read_text(encoding="utf-8"))["tags"] == ["O"]


def test_resolve_config_cli_overrides() -> None:
    resolved = resolve_config(
        {"epochs": 5, "lr": 1e-4, "output_dir": None},
        {"epochs": 3, "lr": 2e-5, "batch_size": 8},
    )
    assert resolved["epochs"] == 5
    assert resolved["lr"] == 1e-4
    assert resolved["batch_size"] == 8
    assert resolved["output_dir"] == DEFAULTS["output_dir"]


def test_train_parser_has_canonical_flags() -> None:
    options = {action.dest for action in build_parser()._actions}
    assert {"config", "output_dir", "seed", "epochs", "batch_size", "lr"} <= options


def test_write_metrics_schema(tmp_path: Path) -> None:
    path = write_metrics(
        model="marbert",
        split="test",
        precision=0.5,
        recall=0.4,
        f1=0.44,
        per_entity={"PER": {"precision": 0.6, "f1": 0.5, "support": 10}},
        config={"model": "marbert"},
        approach="advanced_arabic_transformer",
        owner="member_4",
        out=tmp_path / "marbert.json",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {
        "model", "approach", "owner", "split", "precision", "recall",
        "f1", "per_entity", "config", "timestamp",
    }
    assert set(data["per_entity"]) == {"PER", "ORG", "LOC"}
    assert data["per_entity"]["PER"]["support"] == 10
    assert data["owner"] == "member_4"


def test_config_declares_member_4_and_marbert() -> None:
    text = (PROJECT_ROOT / "configs" / "marbert.yaml").read_text(encoding="utf-8")
    assert "owner: member_4" in text
    assert "model_name: UBC-NLP/MARBERT" in text
