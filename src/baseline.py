"""All-O baseline: the trivial floor for the comparative study (issue 2.4).

Writes canonical predictions and metrics with the shared harness:

    python src/baseline.py
    python src/baseline.py --split validation --out /tmp/baseline_val.jsonl \
        --metrics-out results/metrics/baseline_validation.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data import load_arabic_wikiann
from src.common.config import METRICS_DIR, PREDICTIONS_DIR
from src.evaluate import evaluate_model


def all_o_records(dataset: Any, split: str) -> list[dict[str, Any]]:
    """Return canonical one-tag-per-word records with every tag set to O."""
    tokens = dataset[split]["tokens"]
    return [
        {"id": index, "tokens": list(sentence), "tags": ["O"] * len(sentence)}
        for index, sentence in enumerate(tokens)
    ]


def write_jsonl(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="All-O baseline predictions/metrics")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--out", default=None, help="default: results/predictions/baseline_<split>.jsonl")
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="default: results/metrics/baseline_<split>.json (test: baseline.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    dataset = load_arabic_wikiann()
    predictions_path = (
        Path(arguments.out)
        if arguments.out
        else PREDICTIONS_DIR / f"baseline_{arguments.split}.jsonl"
    )
    write_jsonl(all_o_records(dataset, arguments.split), predictions_path)

    metrics_path = arguments.metrics_out
    if metrics_path is None:
        name = "baseline.json" if arguments.split == "test" else f"baseline_{arguments.split}.json"
        metrics_path = METRICS_DIR / name
    metrics = evaluate_model(
        "baseline",
        split=arguments.split,
        predictions=predictions_path,
        out=metrics_path,
    )
    print(
        f"Baseline {arguments.split}: P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
        f"(expected 0.0 for all-O)"
    )
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
