"""TEMPORARY PLACEHOLDER — OWNED BY MEMBER 1 (PM).

Do not treat this as finished M1 work. It exists only so Member 4's MARBERT
pipeline can be scored before Epic 2 is merged. Member 1 must review/replace
this file; the metrics schema is the hard contract from ROADMAP.md §2.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # allow `python src/evaluate.py --model marbert`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data import load_arabic_wikiann
from src.common.config import MODEL_NAMES, PLOTS_DIR, PREDICTIONS_DIR
from src.data_utils import decode_tags
from src.results import write_metrics

SPLITS = ("train", "validation", "test")


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    """Read an approach YAML, or return {} when none is supplied."""
    if config_path is None:
        return {}
    import yaml  # lazy: offline tests need no dependencies

    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return data


def compute_metrics(
    pred_tags: Sequence[Sequence[str]],
    gold_tags: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Compute seqeval P/R/F1 overall and per entity (issues 2.1–2.2)."""
    from seqeval.metrics import classification_report

    if len(pred_tags) != len(gold_tags):
        raise ValueError("prediction and gold sequence counts differ")
    report = classification_report(gold_tags, pred_tags, output_dict=True, digits=4)
    summary = report.get("micro avg") or report.get("macro avg") or {}
    per_entity: dict[str, dict[str, float | int]] = {}
    # seqeval names the F1 key "f1-score" (not "f1"); accept both so a schema
    # change cannot silently turn every F1 into 0.
    def f1_score(scores: Mapping[str, Any]) -> float:
        return float(scores.get("f1-score", scores.get("f1", 0.0)))

    for name in ("PER", "ORG", "LOC"):
        scores = report.get(name, {})
        per_entity[name] = {
            "precision": float(scores.get("precision", 0.0)),
            "recall": float(scores.get("recall", 0.0)),
            "f1": f1_score(scores),
            "support": int(scores.get("support", 0)),
        }
    return {
        "precision": float(summary.get("precision", 0.0)),
        "recall": float(summary.get("recall", 0.0)),
        "f1": f1_score(summary),
        "per_entity": per_entity,
    }


def write_diagnostics(
    model: str,
    pred_tags: Sequence[Sequence[str]],
    gold_tags: Sequence[Sequence[str]],
    output_dir: Path | None = None,
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
    gold_flat = [tag for sequence in gold_tags for tag in sequence]
    pred_flat = [tag for sequence in pred_tags for tag in sequence]
    labels = sorted(set(gold_flat) | set(pred_flat))
    matrix = confusion_matrix(gold_flat, pred_flat, labels=labels).tolist()
    (directory / f"{model}_confusion.json").write_text(
        json.dumps({"labels": labels, "matrix": matrix}, indent=2), encoding="utf-8"
    )
    (directory / f"{model}_classification_report.txt").write_text(
        sk_report(gold_flat, pred_flat, labels=labels, zero_division=0),
        encoding="utf-8",
    )


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSON object per line from a canonical predictions file."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not {"id", "tokens", "tags"} <= set(record):
                raise ValueError(f"{path}:{line_number} is missing id/tokens/tags")
            records.append(record)
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
) -> dict[str, Any]:
    """Score predictions against the gold data.

    By default reads `results/predictions/<model>_<split>.jsonl`; `--predictions`
    exists so smoke runs cannot overwrite canonical artifacts.
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
    gold_sequences = [decode_tags(tags) for tags in dataset[split]["ner_tags"]]
    if max_samples is not None:
        if max_samples <= 0 or max_samples > len(gold_sequences):
            raise ValueError(
                f"--max_samples must be between 1 and {len(gold_sequences)}, "
                f"got {max_samples}"
            )
        gold_sequences = gold_sequences[:max_samples]
    if len(records) != len(gold_sequences):
        raise ValueError(
            f"{prediction_path} has {len(records)} rows but {split} has "
            f"{len(gold_sequences)} sequences"
        )

    pred_tags: list[list[str]] = []
    aligned_gold: list[list[str]] = []
    for index, (record, gold) in enumerate(zip(records, gold_sequences)):
        tags = list(record["tags"])
        if len(tags) != len(gold):
            raise ValueError(
                f"row {index}: {len(tags)} predicted tags vs {len(gold)} gold tokens"
            )
        pred_tags.append(tags)
        aligned_gold.append(gold)

    metrics = compute_metrics(pred_tags, aligned_gold)
    if diagnostics:
        write_diagnostics(model, pred_tags, aligned_gold)
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
    )
    print(json.dumps(scores, indent=2))
