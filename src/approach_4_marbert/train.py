"""Member 4 — MARBERT training CLI (Epic 6, issues 6.1 and 6.2).

Usage:
    python src/approach_4_marbert/train.py --config configs/marbert.yaml

Heavy ML imports are intentionally inside functions so this module can be
imported by offline unit tests without torch/transformers installed (the same
pattern used by src/common/seed.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # allow `python src/approach_4_marbert/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.load_data import load_arabic_wikiann
from src.common.config import (
    DEFAULT_SEED,
    ID2LABEL,
    LABEL2ID,
    PROJECT_ROOT,
    ensure_dirs,
)
from src.common.logging_utils import get_logger
from src.common.seed import set_seed
from src.data_utils import (
    make_token_classification_dataloader,
    normalization_from_config,
)
from src.evaluate import compute_metrics as seqeval_metrics

LOGGER = get_logger("marbert.train")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "marbert.yaml"

# Used only when a key is absent from the YAML. Config is the source of truth.
DEFAULTS: dict[str, Any] = {
    "model_name": "UBC-NLP/MARBERT",
    "epochs": 3,
    "batch_size": 16,
    "eval_batch_size": 32,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_len": 128,
    "seed": DEFAULT_SEED,
    "fp16": False,
    "label_all_subwords": False,
    "output_dir": "results/models/marbert",
    "max_train_samples": None,
    "max_eval_samples": None,
}

CLI_OVERRIDABLE = (
    "output_dir",
    "seed",
    "epochs",
    "batch_size",
    "eval_batch_size",
    "lr",
    "max_len",
    "model_name",
    "fp16",
    "max_train_samples",
    "max_eval_samples",
)


def build_parser() -> argparse.ArgumentParser:
    """Canonical CLI from ROADMAP.md §2.4, plus a few convenience flags."""
    parser = argparse.ArgumentParser(description="Fine-tune MARBERT for Arabic NER")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Smoke-test only: use the first N training sequences",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Smoke-test only: use the first N validation sequences",
    )
    return parser


def limit_split(split: Any, limit: int | None, name: str) -> Any:
    """Take the first `limit` rows of a HF split; None means no limiting."""
    if limit is None:
        return split
    if limit <= 0:
        raise ValueError(f"--max_{name}_samples must be positive, got {limit}")
    if limit > len(split):
        raise ValueError(f"--max_{name}_samples={limit} exceeds split size {len(split)}")
    return split.select(range(limit))


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML config lazily (yaml is a listed dependency)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def resolve_config(
    cli_values: Mapping[str, Any], file_values: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge DEFAULTS <- YAML <- explicitly passed CLI flags."""
    resolved = dict(DEFAULTS)
    resolved.update(file_values)
    for key in CLI_OVERRIDABLE:
        value = cli_values.get(key)
        if value is not None:
            resolved[key] = value
    return resolved


def build_model_and_tokenizer(model_name: str, num_labels: int) -> tuple[Any, Any]:
    """Load MARBERT with a tag head whose labels are the canonical LABELS."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(
            "A fast tokenizer is required: word_ids() is needed for subword alignment"
        )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={int(key): value for key, value in ID2LABEL.items()},
        label2id=dict(LABEL2ID),
    )
    return model, tokenizer


def compute_metrics_fn(eval_prediction: Any) -> dict[str, float]:
    """Trainer-facing model-selection metric (final metrics come from evaluate.py).

    Trainer does not hand us word_ids, so this scores the surviving non-ignored
    subword positions. It is used only to pick the best checkpoint; the reported
    numbers are produced word-by-word by predict.py + evaluate.py.
    """
    import numpy as np

    predictions = eval_prediction.predictions
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    predicted_ids = np.argmax(logits, axis=-1)
    label_ids = eval_prediction.label_ids

    predicted_tags: list[list[str]] = []
    gold_tags: list[list[str]] = []
    for predicted_sequence, label_sequence in zip(predicted_ids, label_ids):
        predicted_sequence_tags: list[str] = []
        gold_sequence_tags: list[str] = []
        for predicted_id, label_id in zip(predicted_sequence, label_sequence):
            label_id = int(label_id)
            if label_id == -100:  # IGNORE_INDEX: padding or continuation subword
                continue
            predicted_sequence_tags.append(ID2LABEL.get(int(predicted_id), "O"))
            gold_sequence_tags.append(ID2LABEL.get(label_id, "O"))
        if predicted_sequence_tags:
            predicted_tags.append(predicted_sequence_tags)
            gold_tags.append(gold_sequence_tags)

    if not predicted_tags:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    scores = seqeval_metrics(predicted_tags, gold_tags)
    flat = {
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
    }
    for entity, values in scores["per_entity"].items():
        flat[f"{entity}_f1"] = values["f1"]
    return flat


def make_trainer(
    model: Any,
    config: Mapping[str, Any],
    train_loader: Any,
    eval_loader: Any,
) -> Any:
    """Create a Hugging Face Trainer with reproducible, comparable settings."""
    from transformers import Trainer, TrainingArguments

    arguments = {
        "output_dir": str(config["output_dir"]),
        "num_train_epochs": int(config["epochs"]),
        "per_device_train_batch_size": int(config["batch_size"]),
        "per_device_eval_batch_size": int(config["eval_batch_size"]),
        "learning_rate": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": int(config["seed"]),
        "fp16": bool(config["fp16"]),
    }
    try:  # transformers >= 4.46 renamed evaluation_strategy -> eval_strategy
        training_args = TrainingArguments(eval_strategy="epoch", **arguments)
    except TypeError as error:
        if "eval_strategy" not in str(error):
            raise
        training_args = TrainingArguments(evaluation_strategy="epoch", **arguments)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=eval_loader.dataset,
        data_collator=train_loader.collate_fn,
        compute_metrics=compute_metrics_fn,
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    file_config = load_config(arguments.config)
    config = resolve_config(vars(arguments), file_config)
    set_seed(int(config["seed"]))
    ensure_dirs()

    LOGGER.info("Loading WikiANN ar through the canonical loader")
    dataset = load_arabic_wikiann(cache_dir=config.get("cache_dir"))
    train_split = limit_split(dataset["train"], config.get("max_train_samples"), "train")
    validation_split = limit_split(
        dataset["validation"], config.get("max_eval_samples"), "eval"
    )
    if config.get("max_train_samples") or config.get("max_eval_samples"):
        LOGGER.warning(
            "SMOKE TEST: train=%d validation=%d (not a reportable run)",
            len(train_split),
            len(validation_split),
        )
    LOGGER.info(
        "Fine-tuning %s: epochs=%s batch=%s lr=%s max_len=%s",
        config["model_name"],
        config["epochs"],
        config["batch_size"],
        config["lr"],
        config["max_len"],
    )

    model, tokenizer = build_model_and_tokenizer(
        str(config["model_name"]), num_labels=len(LABEL2ID)
    )
    normalization = normalization_from_config(config.get("normalization"))
    train_loader = make_token_classification_dataloader(
        train_split,
        tokenizer,
        batch_size=int(config["batch_size"]),
        max_len=int(config["max_len"]),
        shuffle=True,
        seed=int(config["seed"]),
        normalization=normalization,
        label_all_subwords=bool(config["label_all_subwords"]),
    )
    eval_loader = make_token_classification_dataloader(
        validation_split,
        tokenizer,
        batch_size=int(config["eval_batch_size"]),
        max_len=int(config["max_len"]),
        shuffle=False,
        normalization=normalization,
        label_all_subwords=bool(config["label_all_subwords"]),
    )

    trainer = make_trainer(model, config, train_loader, eval_loader)
    trainer.train()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved model + tokenizer to %s", output_dir)
    LOGGER.info(
        "Next: python src/approach_4_marbert/predict.py --model_dir %s "
        "--split test --out results/predictions/marbert_test.jsonl",
        output_dir,
    )


if __name__ == "__main__":
    main()
