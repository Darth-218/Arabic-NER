"""Train and tune the classical CRF baseline (issues 3.2–3.3).

Canonical CLI (ROADMAP §2.4):
    python src/approach_1_crf/train.py --config configs/crf.yaml

The CRF reuses `src/approach_1_crf/features.py` and the canonical loader.
Features are computed on normalized tokens; predictions remain word-level and
echo the original surface forms.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.load_data import load_arabic_wikiann
from src.approach_1_crf.features import sentence_features
from src.common.config import ID2LABEL, PROJECT_ROOT, ensure_dirs
from src.common.logging_utils import get_logger
from src.common.seed import set_seed
from src.data_utils import decode_tags, normalization_from_config, normalize_arabic
from src.evaluate import compute_metrics

LOGGER = get_logger("crf.train")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "crf.yaml"

DEFAULTS: dict[str, Any] = {
    "normalization": {
        "strip_diacritics": True,
        "remove_tatweel": True,
        "normalize_alef": True,
        "normalize_ya": True,
    },
    "use_gazetteer": True,
    "algorithm": "lbfgs",
    "c1": 0.1,
    "c2": 0.1,
    "max_iterations": 100,
    "all_possible_transitions": True,
    "tune": True,
    "tuning": {"c1": [0.1, 0.5], "c2": [0.05, 0.1]},
    "seed": 42,
    "output_dir": "results/models/crf",
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def resolve_config(cli: Mapping[str, Any], file_config: Mapping[str, Any]) -> dict[str, Any]:
    config = {**DEFAULTS, **file_config}
    for key, value in cli.items():
        if value is not None:
            if key == "epochs":
                config["max_iterations"] = int(value)
            elif key in {"config", "max_iterations"}:
                if key == "max_iterations":
                    config[key] = int(value)
            else:
                config[key] = value
    if cli.get("c1") is not None or cli.get("c2") is not None:
        config["tune"] = False
    return config


def token_normalizer(config: Mapping[str, Any]):
    options = normalization_from_config(config.get("normalization"))

    def normalize(tokens: Sequence[str]) -> list[str]:
        return [normalize_arabic(token, **options) for token in tokens]

    return normalize


def limit_split(split: Any, limit: int | None, name: str) -> Any:
    if limit is None:
        return split
    if limit <= 0 or limit > len(split):
        raise ValueError(f"--max_{name}_samples must be between 1 and {len(split)}")
    LOGGER.warning("SMOKE TEST: using only %d %s sequences", limit, name)
    return split.select(range(limit))


def build_gazetteer(dataset: Any, normalize) -> dict[str, str]:
    counts: dict[str, Counter[str]] = {}
    for tokens, tags in zip(dataset["train"]["tokens"], dataset["train"]["ner_tags"]):
        for token, tag in zip(tokens, tags):
            label = ID2LABEL[int(tag)]
            if not label.startswith("B-"):
                continue
            entity = label[2:]
            counts.setdefault(normalize([token])[0], Counter())[entity] += 1
    return {
        token: counter.most_common(1)[0][0]
        for token, counter in counts.items()
    }


def build_features(
    dataset: Any,
    split: str,
    normalize,
    gazetteer: Mapping[str, str] | None,
) -> tuple[list[list[dict[str, Any]]], list[list[str]]]:
    features: list[list[dict[str, Any]]] = []
    labels: list[list[str]] = []
    for tokens, tags in zip(dataset[split]["tokens"], dataset[split]["ner_tags"]):
        normalized = normalize(tokens)
        features.append(sentence_features(normalized, gazetteer))
        labels.append(decode_tags(tags))
    return features, labels


def make_crf(config: Mapping[str, Any]):
    from sklearn_crfsuite import CRF

    return CRF(
        algorithm=str(config.get("algorithm", "lbfgs")),
        c1=float(config["c1"]),
        c2=float(config["c2"]),
        max_iterations=int(config["max_iterations"]),
        all_possible_transitions=bool(config.get("all_possible_transitions", True)),
    )


def candidates(config: Mapping[str, Any]) -> list[tuple[float, float]]:
    c1 = float(config["c1"])
    c2 = float(config["c2"])
    if not config.get("tune"):
        return [(c1, c2)]
    tuning = config.get("tuning") or {}
    c1_values = [float(value) for value in tuning.get("c1", [c1])]
    c2_values = [float(value) for value in tuning.get("c2", [c2])]
    return list(dict.fromkeys(product(c1_values, c2_values)))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a classical CRF for Arabic NER")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="maps to max_iterations")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="accepted; unused for CRF")
    parser.add_argument("--lr", type=float, default=None, help="accepted; unused for CRF")
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--c2", type=float, default=None)
    parser.add_argument("--tune", dest="tune", action="store_true", default=None)
    parser.add_argument("--no-tune", dest="tune", action="store_false")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    arguments = parser.parse_args(argv)

    cli = vars(arguments)
    config = resolve_config(cli, load_config(arguments.config))
    if (
        (cli.get("max_train_samples") is not None or cli.get("max_eval_samples") is not None)
        and cli.get("tune") is None
    ):
        config["tune"] = False
    if cli.get("batch_size") is not None or cli.get("lr") is not None:
        LOGGER.warning("--batch_size/--lr are accepted for CLI parity but unused by CRF")
    set_seed(int(config["seed"]))
    ensure_dirs()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    normalize = token_normalizer(config)
    dataset = load_arabic_wikiann()
    splits = {
        "train": limit_split(dataset["train"], cli.get("max_train_samples"), "train"),
        "validation": limit_split(
            dataset["validation"], cli.get("max_eval_samples"), "eval"
        ),
    }
    gazetteer = build_gazetteer(splits, normalize) if config.get("use_gazetteer") else None
    if gazetteer:
        LOGGER.info("Gazetteer entries: %d", len(gazetteer))

    LOGGER.info("Building features")
    train_x, train_y = build_features(splits, "train", normalize, gazetteer)
    validation_x, validation_y = build_features(splits, "validation", normalize, gazetteer)

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for c1, c2 in candidates(config):
        LOGGER.info("Fitting c1=%s c2=%s", c1, c2)
        model = make_crf({**config, "c1": c1, "c2": c2})
        model.fit(train_x, train_y)
        predictions = model.predict(validation_x)
        metrics = compute_metrics(predictions, validation_y)
        entry = {
            "c1": c1,
            "c2": c2,
            "validation": {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "per_entity": metrics["per_entity"],
            },
        }
        results.append(entry)
        LOGGER.info("c1=%s c2=%s validation F1=%.4f", c1, c2, metrics["f1"])
        if best is None or metrics["f1"] > best["selection"]["validation"]["f1"]:
            best = {
                "selection": entry,
                "model": model,
                "c1": c1,
                "c2": c2,
                "validation_predictions": predictions,
                "validation_gold": validation_y,
            }

    assert best is not None
    import joblib

    model_path = output_dir / "model.joblib"
    joblib.dump(best["model"], model_path)

    saved_config = {
        **config,
        "c1": best["c1"],
        "c2": best["c2"],
        "tuning_results": results,
        "train_sequences": len(train_y),
        "validation_sequences": len(validation_y),
        "gazetteer_size": len(gazetteer or {}),
    }
    (output_dir / "crf_config.json").write_text(
        json.dumps(saved_config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if gazetteer:
        (output_dir / "gazetteer.json").write_text(
            json.dumps(gazetteer, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best": best["selection"],
        "candidates": results,
        "model_path": str(model_path),
        "train_sequences": len(train_y),
        "validation_sequences": len(validation_y),
        "gazetteer_size": len(gazetteer or {}),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Best c1={best['c1']} c2={best['c2']} "
        f"validation F1={best['selection']['validation']['f1']:.4f}"
    )
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
