"""Shared offline stubs for harness tests (no torch/transformers/network)."""

from __future__ import annotations

from typing import Any


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


class FakeSplit:
    """Minimal stand-in for a HF Dataset split used by the loader helpers."""

    def __init__(self, tokens, ner_tags, extra=None):
        self._data = {"tokens": tokens, "ner_tags": ner_tags, **(extra or {})}
        self.column_names = list(self._data)
        self.last_remove_columns: list[str] | None = None

    def __len__(self) -> int:
        return len(self._data["tokens"])

    def __getitem__(self, key):
        return self._data[key]

    def map(self, function, batched=True, remove_columns=None):
        assert batched is True
        self.last_remove_columns = list(remove_columns or [])
        output = function({key: self._data[key] for key in ("tokens", "ner_tags")})
        return FakeSplit(output["tokens"], output["ner_tags"])
