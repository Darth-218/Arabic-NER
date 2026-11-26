"""Member 4 — MARBERT prediction CLI (Epic 6, issues 6.4/6.5 support).

Usage:
    python src/approach_4_marbert/predict.py \
        --model_dir results/models/marbert \
        --split test \
        --out results/predictions/marbert_test.jsonl

Writes one JSON object per line: {"id": 0, "tokens": [...], "tags": [...]}
with exactly one BIO tag per original word (ROADMAP.md §2.4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # allow `python src/approach_4_marbert/predict.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.load_data import load_arabic_wikiann
from src.common.config import PREDICTIONS_DIR, PROJECT_ROOT, ensure_dirs
from src.common.logging_utils import get_logger
from src.data_utils import (
    align_predictions_to_words,
    decode_tags,
    make_prediction_batches,
    normalization_from_config,
)

LOGGER = get_logger("marbert.predict")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "marbert.yaml"


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load YAML settings (normalization/max_len) when a config is supplied."""
    if path is None:
        return {}
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def build_prediction_records(
    ids: Sequence[Any],
    tokens_per_example: Sequence[Sequence[str]],
    predicted_ids_per_example: Sequence[Sequence[int]],
    word_ids_per_example: Sequence[Sequence[int | None]],
) -> list[dict[str, Any]]:
    """Collapse subword predictions into canonical one-tag-per-word records."""
    lengths = {
        len(ids),
        len(tokens_per_example),
        len(predicted_ids_per_example),
        len(word_ids_per_example),
    }
    if len(lengths) != 1:
        raise ValueError("Prediction batch lists must have the same length")

    records: list[dict[str, Any]] = []
    for example_id, tokens, predicted_ids, word_ids in zip(
        ids, tokens_per_example, predicted_ids_per_example, word_ids_per_example
    ):
        word_label_ids = align_predictions_to_words(
            predicted_ids, word_ids, num_words=len(tokens)
        )
        records.append(
            {
                "id": int(example_id),
                "tokens": list(tokens),
                "tags": decode_tags(word_label_ids),
            }
        )
    return records


def write_jsonl(records: Sequence[Mapping[str, Any]], out: str | Path) -> Path:
    """Write canonical JSONL, creating parent directories when needed."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Arabic NER tags with MARBERT")
    parser.add_argument("--model_dir", required=True, help="trained checkpoint dir")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto (default)")
    return parser


def predict(arguments: argparse.Namespace) -> Path:
    """Run inference over a split and return the written JSONL path."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    ensure_dirs()
    model_dir = Path(arguments.model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. Run train.py first."
        )

    config = load_config(arguments.config)
    normalization = normalization_from_config(config.get("normalization"))
    max_len = arguments.max_len or int(config.get("max_len", 128))
    batch_size = arguments.batch_size or int(
        config.get("eval_batch_size", config.get("batch_size", 16))
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()
    if arguments.device and arguments.device != "auto":
        device = torch.device(arguments.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_arabic_wikiann(cache_dir=config.get("cache_dir"))
    if arguments.split not in dataset:
        raise ValueError(f"Dataset has no {arguments.split!r} split")

    output = Path(arguments.out) or PREDICTIONS_DIR / f"marbert_{arguments.split}.jsonl"
    all_records: list[dict[str, Any]] = []
    for batch in make_prediction_batches(
        dataset[arguments.split]["tokens"],
        tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        normalization=normalization,
    ):
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            batch["attention_mask"], dtype=torch.long, device=device
        )
        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        all_records.extend(
            build_prediction_records(
                batch["ids"],
                batch["tokens"],
                predicted_ids,
                batch["word_ids"],
            )
        )

    path = write_jsonl(all_records, output)
    LOGGER.info("Wrote %d predictions to %s", len(all_records), path)
    return path


if __name__ == "__main__":
    predict(build_parser().parse_args())
