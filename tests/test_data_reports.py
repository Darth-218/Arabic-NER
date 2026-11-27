"""Tests for EDA and data-quality helpers (issues 1.2 and 1.6)."""

from __future__ import annotations

from data.eda import entity_span_counts, label_counts, length_stats, percentile
from data.quality_report import bio_errors, group_duplicates, summarize_split
from src.common.config import LABEL2ID


def test_percentile_and_length_stats() -> None:
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0.50) == 3.0
    stats = length_stats(values)
    assert stats["median"] == 3.0
    assert stats["max"] == 5
    assert stats["mean"] == 3.0


def test_label_and_entity_counts() -> None:
    tags = [
        [LABEL2ID["B-PER"], LABEL2ID["I-PER"], LABEL2ID["O"]],
        [LABEL2ID["B-LOC"], LABEL2ID["B-ORG"]],
    ]
    labels = label_counts(tags)
    assert labels["O"] == 1
    assert labels["I-PER"] == 1
    entities = entity_span_counts(tags)
    assert entities == {"PER": 1, "ORG": 1, "LOC": 1}


def test_group_duplicates_within_and_cross_split() -> None:
    within, cross = group_duplicates(
        {
            "train": [["a", "b"], ["a", "b"], ["c"]],
            "test": [["a", "b"], ["d"]],
        }
    )
    assert within["train"]["duplicate_groups"] == 1
    assert within["train"]["rows_in_duplicate_groups"] == 2
    assert within["test"]["duplicate_groups"] == 0
    assert cross["train<->test"]["groups"] == 1
    assert cross["train<->test"]["pairs"] == 2


def test_bio_errors_detects_orphan_i_tags() -> None:
    tags = [
        [LABEL2ID["B-PER"], LABEL2ID["I-PER"]],
        [LABEL2ID["I-ORG"]],
        [LABEL2ID["O"], LABEL2ID["I-LOC"]],
    ]
    errors = bio_errors(tags)
    assert len(errors) == 2
    assert errors[0]["tag"] == "I-ORG"
    assert errors[1]["previous"] == "O"


def test_summarize_split_counts_all_o_and_presence() -> None:
    dataset = {
        "train": {
            "tokens": [["a"], ["b"], ["c"]],
            "ner_tags": [
                [LABEL2ID["O"]],
                [LABEL2ID["B-PER"]],
                [LABEL2ID["O"]],
            ],
        }
    }
    summary = summarize_split(dataset, "train")
    assert summary["sequences"] == 3
    assert summary["all_O_sequences"] == 2
    assert summary["token_tag_mismatches"] == 0
    assert summary["entity_presence"]["PER"] == 1
    assert summary["entity_presence"]["ORG"] == 0
