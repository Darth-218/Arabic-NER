"""Canonical evaluation utilities (ROADMAP.md §2.5, issues 2.1–2.2).

Predictions are scored with seqeval using **explicit micro averaging** for the
overall P/R/F1 and per-entity scores for PER/ORG/LOC. The functions here are
also used by the training scripts, so they must stay import-light.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # allow `python src/evaluate.py --model marbert`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data import load_arabic_wikiann
from src.common.config import LABELS, METRICS_DIR, MODEL_NAMES, PLOTS_DIR, PREDICTIONS_DIR
from src.data_utils import decode_tags
from src.results import write_metrics

SPLITS = ("train", "validation", "test")
ENTITY_NAMES = ("PER", "ORG", "LOC")
LABEL_SET = frozenset(LABELS)


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    """Read an approach YAML, or return {} when none is supplied."""
    if config_path is None:
        return {}
    import yaml  # lazy: offline tests need no dependencies

    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return data


def validate_tags(sequences: Sequence[Sequence[str]], name: str) -> None:
    """Reject empty sequences and tags outside the canonical label schema."""
    for index, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError(f"{name}[{index}] is empty")
        unknown = {tag for tag in sequence if tag not in LABEL_SET}
        if unknown:
            raise ValueError(
                f"{name}[{index}] contains non-canonical tags: {sorted(unknown)}"
            )


def compute_metrics(
    pred_tags: Sequence[Sequence[str]],
    gold_tags: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Compute micro P/R/F1 overall and per-entity scores (issues 2.1–2.2)."""
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    if len(pred_tags) != len(gold_tags):
        raise ValueError("prediction and gold sequence counts differ")
    for index, (predicted, gold) in enumerate(zip(pred_tags, gold_tags)):
        if len(predicted) != len(gold):
            raise ValueError(
                f"sequence {index}: {len(predicted)} predicted tags vs "
                f"{len(gold)} gold tags"
            )
    validate_tags(pred_tags, "predicted tags")
    validate_tags(gold_tags, "gold tags")

    # Canonical overall metrics are explicitly micro-averaged over entity spans.
    micro_precision = float(
        precision_score(gold_tags, pred_tags, average="micro", zero_division=0)
    )
    micro_recall = float(
        recall_score(gold_tags, pred_tags, average="micro", zero_division=0)
    )
    micro_f1 = float(
        f1_score(gold_tags, pred_tags, average="micro", zero_division=0)
    )
    # Macro values are exposed for diagnostics; the canonical metrics JSON keeps
    # only micro overall + per-entity (ROADMAP §2.5).
    macro_precision = float(
        precision_score(gold_tags, pred_tags, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(gold_tags, pred_tags, average="macro", zero_division=0)
    )
    macro_f1 = float(
        f1_score(gold_tags, pred_tags, average="macro", zero_division=0)
    )

    report = classification_report(
        gold_tags, pred_tags, output_dict=True, digits=4, zero_division=0
    )

    def f1_value(scores: Mapping[str, Any]) -> float:
        # seqeval names the F1 key "f1-score"; accept "f1" as a fallback so a
        # schema change cannot silently turn every F1 into 0 (PR #43 finding).
        return float(scores.get("f1-score", scores.get("f1", 0.0)))

    per_entity: dict[str, dict[str, float | int]] = {}
    for name in ENTITY_NAMES:
        scores = report.get(name, {})
        per_entity[name] = {
            "precision": float(scores.get("precision", 0.0)),
            "recall": float(scores.get("recall", 0.0)),
            "f1": f1_value(scores),
            "support": int(scores.get("support", 0)),
        }
    return {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_entity": per_entity,
    }


def write_diagnostics(
    model: str,
    pred_tags: Sequence[Sequence[str]],
    gold_tags: Sequence[Sequence[str]],
    output_dir: Path | None = None,
    *,
    split: str = "test",
) -> None:
    """Export a confusion matrix and classification report under results/plots/."""
    try:
        from sklearn.metrics import classification_report as sk_report
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("sklearn not installed; skipping confusion-matrix export")
        return

    directory = output_dir or PLOTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if split == "test" else f"_{split}"
    gold_flat = [tag for sequence in gold_tags for tag in sequence]
    pred_flat = [tag for sequence in pred_tags for tag in sequence]
    labels = sorted(set(gold_flat) | set(pred_flat))
    matrix = confusion_matrix(gold_flat, pred_flat, labels=labels).tolist()
    (directory / f"{model}{suffix}_confusion.json").write_text(
        json.dumps({"labels": labels, "matrix": matrix}, indent=2), encoding="utf-8"
    )
    (directory / f"{model}{suffix}_classification_report.txt").write_text(
        sk_report(gold_flat, pred_flat, labels=labels, zero_division=0),
        encoding="utf-8",
    )


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate one canonical `{id,tokens,tags}` record per line.

    Rows may be stored out of order; ids must be unique and contiguous from 0
    so they can be matched back to gold sequences deterministically.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not {"id", "tokens", "tags"} <= set(record):
                raise ValueError(f"{path}:{line_number} is missing id/tokens/tags")
            if not isinstance(record["id"], int):
                raise ValueError(f"{path}:{line_number} id must be an integer")
            if not isinstance(record["tokens"], list) or not isinstance(
                record["tags"], list
            ):
                raise ValueError(f"{path}:{line_number} tokens/tags must be lists")
            if len(record["tokens"]) != len(record["tags"]):
                raise ValueError(
                    f"{path}:{line_number} has {len(record['tokens'])} tokens but "
                    f"{len(record['tags'])} tags"
                )
            validate_tags([record["tags"]], f"{path}:{line_number}")
            records.append(record)

    ids = [int(record["id"]) for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate ids")
    if set(ids) != set(range(len(records))):
        raise ValueError(
            f"{path} ids must be contiguous from 0 to {len(records) - 1}"
        )
    return records


def evaluate_model(
    model: str,
    *,
    split: str = "test",
    config_path: str | Path | None = None,
    out: str | Path | None = None,
    diagnostics: bool = True,
    max_samples: int | None = None,
    predictions: str | Path | None = None,
    check_tokens: bool = True,
) -> dict[str, Any]:
    """Score predictions against the gold data.

    By default reads `results/predictions/<model>_<split>.jsonl`. Smoke runs
    (`max_samples`) must not overwrite canonical metrics: when `out` is omitted
    they write `<model>_smoke.json` instead of `<model>.json`.
    """
    if split not in SPLITS:
        raise ValueError(f"--split must be one of {SPLITS}")
    prediction_path = (
        Path(predictions)
        if predictions is not None
        else PREDICTIONS_DIR / f"{model}_{split}.jsonl"
    )
    records = load_predictions(prediction_path)

    dataset = load_arabic_wikiann()
    if split not in dataset:
        raise ValueError(f"Dataset has no {split!r} split")
    gold_tokens = [list(tokens) for tokens in dataset[split]["tokens"]]
    gold_sequences = [decode_tags(tags) for tags in dataset[split]["ner_tags"]]
    if max_samples is not None:
        if max_samples <= 0 or max_samples > len(gold_sequences):
            raise ValueError(
                f"--max_samples must be between 1 and {len(gold_sequences)}, "
                f"got {max_samples}"
            )
        gold_tokens = gold_tokens[:max_samples]
        gold_sequences = gold_sequences[:max_samples]
        if out is None:
            out = METRICS_DIR / f"{model}_smoke.json"
            warnings.warn(
                f"smoke evaluation writing metrics to {out}; pass --out to override",
                stacklevel=2,
            )
    if len(records) != len(gold_sequences):
        raise ValueError(
            f"{prediction_path} has {len(records)} rows but {split} has "
            f"{len(gold_sequences)} sequences"
        )

    by_id = {int(record["id"]): record for record in records}
    pred_tags: list[list[str]] = []
    aligned_gold: list[list[str]] = []
    for index, gold in enumerate(gold_sequences):
        record = by_id[index]
        tags = list(record["tags"])
        if len(tags) != len(gold):
            raise ValueError(
                f"row {index}: {len(tags)} predicted tags vs {len(gold)} gold tokens"
            )
        if check_tokens and list(record["tokens"]) != gold_tokens[index]:
            raise ValueError(
                f"row {index}: prediction tokens differ from gold surface forms; "
                "write original WikiANN tokens to the JSONL (ROADMAP §2.4)"
            )
        pred_tags.append(tags)
        aligned_gold.append(gold)

    metrics = compute_metrics(pred_tags, aligned_gold)
    if diagnostics:
        write_diagnostics(model, pred_tags, aligned_gold, split=split)
    config = load_config(config_path)
    write_metrics(
        model=model,
        split=split,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        per_entity=metrics["per_entity"],
        config=config,
        approach=config.get("approach"),
        owner=config.get("owner"),
        out=out,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate canonical NER predictions")
    parser.add_argument("--model", required=True, choices=[*MODEL_NAMES, "baseline"])
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--config", default=None, help="optional configs/<model>.yaml")
    parser.add_argument("--out", default=None, help="override metrics output path")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Smoke-test only: score only the first N gold sequences",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="override the predictions file (use a separate path for smoke runs)",
    )
    parser.add_argument(
        "--no-token-check",
        action="store_true",
        help="skip the original-surface-form check on prediction tokens",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    scores = evaluate_model(
        arguments.model,
        split=arguments.split,
        config_path=arguments.config,
        out=arguments.out,
        max_samples=arguments.max_samples,
        predictions=arguments.predictions,
        check_tokens=not arguments.no_token_check,
    )
    print(json.dumps(scores, indent=2))
