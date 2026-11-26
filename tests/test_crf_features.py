"""Offline tests for CRF feature engineering (issue 3.1)."""

import pytest

from src.approach_1_crf.features import (
    has_digit,
    is_arabic,
    is_punct,
    sentence_features,
    word_features,
    word_shape,
)

TOKENS = ["زار", "محمد", "القاهرة", "2024"]


def test_one_feature_dict_per_token():
    features = sentence_features(TOKENS)
    assert len(features) == len(TOKENS)
    assert all(isinstance(item, dict) for item in features)


def test_boundary_and_context_features():
    features = sentence_features(TOKENS)
    assert features[0]["is_first"] is True
    assert features[0]["is_last"] is False
    assert features[0]["prev_word"] == "<BOS>"
    assert features[0]["next_word"] == "محمد"
    assert features[-1]["is_last"] is True
    assert features[-1]["next_word"] == "<EOS>"


def test_prefix_suffix_and_shape():
    features = sentence_features(TOKENS)
    assert features[1]["prefix1"] == "م"
    assert features[1]["prefix3"] == "محم"
    assert features[2]["suffix1"] == "ة"
    assert features[2]["shape"] == "a"
    assert features[2]["len_bucket"] == "7+"


def test_orthographic_flags():
    assert is_arabic("القاهرة") is True
    assert is_arabic("Cairo") is False
    assert has_digit("2024") is True
    assert has_digit("محمد") is False
    assert is_punct("،") is True
    assert is_punct("محمد") is False


def test_shape_collapses_digits_and_letters():
    assert word_shape("2024") == "d"
    assert word_shape("ABC") == "x"
    assert word_shape("!!!") == "p"


def test_gazetteer_feature():
    gazetteer = {"القاهرة": "LOC", "محمد": "PER"}
    features = sentence_features(TOKENS, gazetteer=gazetteer)
    assert features[1]["gaz_type"] == "PER"
    assert features[2]["gaz_type"] == "LOC"
    assert features[0]["gaz_type"] == "O"


def test_out_of_range_index_raises():
    with pytest.raises(IndexError):
        word_features(TOKENS, len(TOKENS))


def test_deterministic():
    assert sentence_features(TOKENS) == sentence_features(TOKENS)
