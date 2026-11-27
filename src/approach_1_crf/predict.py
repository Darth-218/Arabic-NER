"""Canonical predictions from the trained CRF (issues 3.4 / ROADMAP §2.4).

    python src/approach_1_crf/predict.py --model_dir results/models/crf \
        --split test --out results/predictions/crf_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.load_data import load_arabic_wikiann
from src.common.config import MODELS_DIR, PREDICTIONS_DIR
from src.common.logging_utils import get_logger
from src.data_utils import normalization_from_config, normalize_arabic

LOGGER = get_logger("crf.predict")


def load_model_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "crf_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; train the model first with train.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_gazetteer(model_dir: Path) -> dict[str, str] | None:
    path = model_dir / "gazetteer.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predict_tags(
    model: Any,
    tokens: Sequence[Sequence[str]],
    *,
    normalization: Mapping[str, bool] | None = None,
    gazetteer: Mapping[str, str] | None = None,
) -> list[list[str]]:
    from src.approach_1_crf.features import sentence_features

    options = normalization or normalization_from_config(None)
    predictions: list[list[str]] = []
    for sentence in tokens:
        normalized = [normalize_arabic(token, **options) for token in sentence]
        features = sentence_features(normalized, gazetteer)
        tags = list(model.predict([features])[0])
        if len(tags) != len(sentence):
            raise ValueError(
                f"CRF returned {len(tags)} tags for {len(sentence)} tokens"
            )
        predictions.append(tags)
    return predictions


def build_records(tokens: Sequence[Sequence[str]], tags: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    if len(tokens) != len(tags):
        raise ValueError("tokens and tag lists have different lengths")
    return [
        {"id": index, "tokens": list(sentence), "tags": list(labels)}
        for index, (sentence, labels) in enumerate(zip(tokens, tags))
    ]


def write_jsonl(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Arabic NER tags with the CRF")
    parser.add_argument("--model_dir", default=str(MODELS_DIR / "crf"))
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Smoke-test only: predict the first N sequences",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    arguments = build_parser().parse_args(argv)
    import joblib

    model_dir = Path(arguments.model_dir)
    config = load_model_config(model_dir)
    gazetteer = load_gazetteer(model_dir) if config.get("use_gazetteer") else None

    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} is missing; train the model first")
    model = joblib.load(model_path)

    dataset = load_arabic_wikiann()
    tokens = [list(sequence) for sequence in dataset[arguments.split]["tokens"]]
    if arguments.max_samples is not None:
        if not 0 < arguments.max_samples <= len(tokens):
            raise ValueError(
                f"--max_samples must be between 1 and {len(tokens)}"
            )
        tokens = tokens[: arguments.max_samples]
        LOGGER.warning("SMOKE TEST: predicting only %d sequences", len(tokens))

    tags = predict_tags(
        model,
        tokens,
        normalization=config.get("normalization"),
        gazetteer=gazetteer,
    )
    output = Path(arguments.out) if arguments.out else PREDICTIONS_DIR / f"crf_{arguments.split}.jsonl"
    path = write_jsonl(build_records(tokens, tags), output)
    LOGGER.info("Wrote %d predictions to %s", len(tags), path)
    return path


if __name__ == "__main__":
    main()
