"""Canonical tokenization / alignment helpers (ROADMAP.md §2.2, issues 1.3–1.5).

`align_labels` and `build_dataloaders` are the public aliases used by the
approach scripts; the longer names remain available for readability. Heavy ML
imports are deliberately lazy so the pure helpers can be unit-tested offline
with a stub tokenizer (see tests/README.md).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):  # allow `python src/data_utils.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ID2LABEL, IGNORE_INDEX, LABEL2ID

# Arabic normalization tables (issue 1.3). All switches default to True and can
# be overridden by the `normalization:` block in configs/marbert.yaml.
_DIACRITIC_CODES = dict.fromkeys(
    [ord(char) for char in "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670"],
    None,
)
_TATWEEL = "\u0640"
_ALEF_TABLE = str.maketrans("أإآ", "ااا")
_YA_TABLE = str.maketrans("ى", "ي")

NORMALIZATION_DEFAULTS: dict[str, bool] = {
    "strip_diacritics": True,
    "remove_tatweel": True,
    "normalize_alef": True,
    "normalize_ya": True,
}


def normalization_from_config(config: Mapping[str, Any] | None) -> dict[str, bool]:
    """Turn a YAML `normalization:` block into keyword arguments."""
    options = dict(NORMALIZATION_DEFAULTS)
    if config:
        for key in options:
            if key in config:
                options[key] = bool(config[key])
    return options


def normalize_arabic(
    text: str,
    *,
    strip_diacritics: bool = True,
    remove_tatweel: bool = True,
    normalize_alef: bool = True,
    normalize_ya: bool = True,
) -> str:
    """Normalize one Arabic token so spelling variants map to one form."""
    result = str(text)
    if strip_diacritics:
        result = result.translate(_DIACRITIC_CODES)
    if remove_tatweel:
        result = result.replace(_TATWEEL, "")
    if normalize_alef:
        result = result.translate(_ALEF_TABLE)
    if normalize_ya:
        result = result.translate(_YA_TABLE)
    return result


def align_labels_with_tokens(
    word_labels: Sequence[int],
    word_ids: Sequence[int | None],
    *,
    ignore_index: int = IGNORE_INDEX,
    label_all_subwords: bool = False,
) -> list[int]:
    """Map word-level BIO ids onto subword positions.

    Special tokens (None), out-of-range words and — by default — continuation
    subwords receive `ignore_index` (-100) so they do not affect loss/metrics.
    """
    aligned: list[int] = []
    seen: set[int] = set()
    for word_id in word_ids:
        if word_id is None or int(word_id) < 0 or int(word_id) >= len(word_labels):
            aligned.append(ignore_index)
            continue
        index = int(word_id)
        if label_all_subwords or index not in seen:
            aligned.append(int(word_labels[index]))
            seen.add(index)
        else:
            aligned.append(ignore_index)
    return aligned


def align_predictions_to_words(
    predicted_ids: Sequence[int],
    word_ids: Sequence[int | None],
    num_words: int | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Collapse subword predictions back to exactly one id per original word.

    The first subword of a word owns the prediction (the inverse of
    `align_labels_with_tokens` with `label_all_subwords=False`). Missing words
    — e.g. because of truncation — fall back to `O`.
    """
    if num_words is None:
        valid = [int(w) for w in word_ids if w is not None and int(w) >= 0]
        num_words = max(valid) + 1 if valid else 0

    by_word: list[int | None] = [None] * num_words
    for predicted_id, word_id in zip(predicted_ids, word_ids):
        if word_id is None:
            continue
        index = int(word_id)
        if index < 0 or index >= num_words:
            continue
        if by_word[index] is None and int(predicted_id) != ignore_index:
            by_word[index] = int(predicted_id)
    return [LABEL2ID["O"] if value is None else value for value in by_word]


def decode_tags(ids: Sequence[int]) -> list[str]:
    """Turn canonical label ids into BIO strings."""
    return [ID2LABEL.get(int(label_id), "O") for label_id in ids]


def encode_tags(tags: Sequence[str]) -> list[int]:
    """Turn BIO strings into canonical label ids."""
    return [LABEL2ID[tag] for tag in tags]


def tokenize_batch(
    batch: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_len: int = 128,
    normalization: Mapping[str, bool] | None = None,
    label_all_subwords: bool = False,
    with_labels: bool = True,
) -> dict[str, Any]:
    """Tokenize a HF-style batch of `tokens`/`ner_tags` and align the labels."""
    options = normalization or normalization_from_config(None)
    texts = [
        [normalize_arabic(word, **options) for word in words]
        for words in batch["tokens"]
    ]
    encoded = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    word_ids = [list(encoded.word_ids(i)) for i in range(len(texts))]
    output: dict[str, Any] = {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded["attention_mask"]),
        "word_ids": word_ids,
    }
    if with_labels:
        output["labels"] = [
            align_labels_with_tokens(
                list(labels),
                ids,
                label_all_subwords=label_all_subwords,
            )
            for labels, ids in zip(batch["ner_tags"], word_ids)
        ]
    return output


def make_token_classification_dataloader(
    dataset: Any,
    tokenizer: Any,
    *,
    batch_size: int = 16,
    max_len: int = 128,
    shuffle: bool = False,
    seed: int = 42,
    normalization: Mapping[str, bool] | None = None,
    label_all_subwords: bool = False,
    num_workers: int = 0,
) -> Any:
    """Build a padded, reusable DataLoader for token classification."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification

    def encode(batch: Mapping[str, Any]) -> dict[str, Any]:
        encoded = tokenize_batch(
            batch,
            tokenizer,
            max_len=max_len,
            normalization=normalization,
            label_all_subwords=label_all_subwords,
            with_labels=True,
        )
        # `word_ids` is Python metadata, not a model input.
        return {
            key: encoded[key]
            for key in ("input_ids", "attention_mask", "labels")
        }

    encoded_dataset = dataset.map(
        encode,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing and aligning labels",
    )
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        encoded_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        generator=generator,
    )


def collate_prediction_batch(
    items: Sequence[Mapping[str, Any]], pad_token_id: int | None
) -> dict[str, Any]:
    """Pad a list of tokenized examples without converting to tensors."""
    pad_id = 0 if pad_token_id is None else int(pad_token_id)
    longest = max(len(item["input_ids"]) for item in items)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for item in items:
        length = len(item["input_ids"])
        input_ids.append(list(item["input_ids"]) + [pad_id] * (longest - length))
        attention_mask.append(
            list(item["attention_mask"]) + [0] * (longest - length)
        )
    return {
        "ids": [item["id"] for item in items],
        "tokens": [item["tokens"] for item in items],
        "word_ids": [item["word_ids"] for item in items],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def make_prediction_batches(
    token_sequences: Iterable[Sequence[str]],
    tokenizer: Any,
    *,
    batch_size: int = 16,
    max_len: int = 128,
    normalization: Mapping[str, bool] | None = None,
) -> Any:
    """Yield padded batches that keep `word_ids` for word-level reconstruction."""
    options = normalization or normalization_from_config(None)
    buffer: list[dict[str, Any]] = []
    for index, words in enumerate(token_sequences):
        normalized = [normalize_arabic(word, **options) for word in words]
        encoded = tokenizer(
            [normalized],
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        buffer.append(
            {
                "id": index,
                "tokens": normalized,
                "input_ids": list(encoded["input_ids"][0]),
                "attention_mask": list(encoded["attention_mask"][0]),
                "word_ids": list(encoded.word_ids(0)),
            }
        )
        if len(buffer) == batch_size:
            yield collate_prediction_batch(buffer, tokenizer.pad_token_id)
            buffer = []
    if buffer:
        yield collate_prediction_batch(buffer, tokenizer.pad_token_id)


# Public contract aliases (review request on PR #43).
align_labels = align_labels_with_tokens
build_dataloaders = make_token_classification_dataloader
