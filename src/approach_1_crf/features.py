"""CRF feature engineering for the classical approach (issue 3.1).

Pure-Python, dependency-free feature extraction so the features can be built
and unit-tested offline. Tokens are expected to be already normalized upstream
(see `src/data_utils.py`, issue 1.3); this module never loads data or models.

`sklearn-crfsuite` expects one feature dictionary per token, which is exactly
what `sentence_features()` returns.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Mapping, Sequence

_ARABIC_BLOCK_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

_BOS = "<BOS>"
_EOS = "<EOS>"


def _in_arabic_block(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _ARABIC_BLOCK_RANGES)


def is_arabic(token: str) -> bool:
    """True when every alphabetic character is in an Arabic Unicode block."""
    letters = [char for char in token if char.isalpha()]
    return bool(letters) and all(_in_arabic_block(char) for char in letters)


def has_digit(token: str) -> bool:
    """True when the token contains at least one digit."""
    return any(char.isdigit() for char in token)


def is_punct(token: str) -> bool:
    """True when the token consists only of punctuation characters."""
    return bool(token) and all(
        unicodedata.category(char).startswith("P") for char in token
    )


def word_shape(token: str, max_len: int = 8) -> str:
    """Collapse a token into a coarse shape, e.g. ``القاهرة`` -> ``aaaaaaa``."""
    classes: list[str] = []
    for char in token:
        if char.isdigit():
            kind = "d"
        elif _in_arabic_block(char) and char.isalpha():
            kind = "a"
        elif char.isalpha():
            kind = "x"
        elif unicodedata.category(char).startswith("P"):
            kind = "p"
        else:
            kind = "o"
        if not classes or classes[-1] != kind:
            classes.append(kind)
    return "".join(classes)[:max_len] or "empty"


def _length_bucket(length: int) -> str:
    if length <= 2:
        return "1-2"
    if length <= 4:
        return "3-4"
    if length <= 6:
        return "5-6"
    return "7+"


def word_features(
    tokens: Sequence[str],
    index: int,
    gazetteer: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the feature dictionary for the token at ``index``."""
    if not 0 <= index < len(tokens):
        raise IndexError(f"index {index} out of range for {len(tokens)} tokens")

    token = str(tokens[index])
    previous = str(tokens[index - 1]) if index > 0 else _BOS
    following = str(tokens[index + 1]) if index + 1 < len(tokens) else _EOS

    features: dict[str, Any] = {
        "bias": 1.0,
        "word": token,
        "word_lower": token.lower(),
        "prefix1": token[:1],
        "len_bucket": _length_bucket(len(token)),
        "is_arabic": is_arabic(token),
        "has_digit": has_digit(token),
        "is_punct": is_punct(token),
        "shape": word_shape(token),
        "is_first": index == 0,
        "is_last": index == len(tokens) - 1,
        "prev_word": previous,
        "next_word": following,
        "prev_shape": word_shape(previous),
        "next_shape": word_shape(following),
        "gaz_type": (gazetteer or {}).get(token, "O"),
    }

    for size in (2, 3):
        if len(token) >= size:
            features[f"prefix{size}"] = token[:size]
    for size in (1, 2, 3):
        if len(token) >= size:
            features[f"suffix{size}"] = token[-size:]

    return features


def sentence_features(
    tokens: Sequence[str],
    gazetteer: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return one feature dictionary per token, suitable for sklearn-crfsuite."""
    return [word_features(tokens, index, gazetteer) for index in range(len(tokens))]
