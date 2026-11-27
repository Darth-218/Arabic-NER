"""Member 3 — AraBERT hyperparameter search (Epic 5, issue 5.3).

Usage:
    python src/approach_3_arabert/hparam_search.py --dry-run
    python src/approach_3_arabert/hparam_search.py

Selection metric: **validation F1**, never test. Test predictions do not exist
until issue 5.4, and tuning on test would leak the numbers we later report.

Each trial runs through the same code paths as `train.py` (canonical loader,
dataloaders, label schema, Trainer settings); only the knob under test changes.
Trials do not save checkpoints — a trial is a measurement, not a deliverable.
Whichever setting wins is copied into `configs/arabert.yaml` and retrained once,
and that retrained model's `results/metrics/arabert.json` is what the report uses.

Grid (5 runs, ~12 min each on the RTX 3050):

  stage 1  lr 2e-5 / 3e-5 / 5e-5, epochs 3, batch 8, max_len 128
  stage 2  best lr, epochs 4
  stage 3  best lr+epochs, max_len 96

`batch_size` is not searched: 8 is already the largest the 4 GB card holds at
`max_len 128` (with gradient checkpointing), and accumulation already gives an
effective batch of 16.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.approach_3_arabert.train import (
    DEFAULT_CONFIG_PATH,
    build_model_and_tokenizer,
    limit_split,
    load_config,
    make_trainer,
    resolve_config,
    validate_config,
)
from src.common.config import (
    LABEL2ID,
    PROJECT_ROOT,
    RESULTS_DIR,
    ensure_dirs,
)
from src.common.logging_utils import get_logger
from src.common.seed import set_seed
from src.data_utils import normalization_from_config

LOGGER = get_logger("arabert.hparam")

SUMMARY_PATH = PROJECT_ROOT / "reports" / "analysis" / "arabert_hparam_search.json"
TRIAL_MODEL_ROOT = RESULTS_DIR / "models" / "arabert_hparam"

BATCH_SIZE = 8
BASE_MAX_LEN = 128
STAGE1_LRS = (2e-5, 3e-5, 5e-5)
STAGE1_EPOCHS = 3


def trial_name(lr: float, epochs: int, max_len: int) -> str:
    """Stable, readable id, e.g. ``lr3e5_e3_bs8_len128``."""
    rate = f"{lr:.0e}".replace("e-0", "e").replace("e+0", "e")
    return f"lr{rate}_e{epochs}_bs{BATCH_SIZE}_len{max_len}"


def stage1_grid() -> list[dict[str, Any]]:
    return [
        {
            "stage": 1,
            "name": trial_name(lr, STAGE1_EPOCHS, BASE_MAX_LEN),
            "lr": lr,
            "epochs": STAGE1_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_len": BASE_MAX_LEN,
            "note": "learning-rate sweep at the default schedule",
        }
        for lr in STAGE1_LRS
    ]


def follow_up_trials(best: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage 2 (more epochs) and stage 3 (shorter max_len) around the winner."""
    longer_epochs = best["epochs"] + 1
    shorter_len = 96
    return [
        {
            "stage": 2,
            "name": trial_name(best["lr"], longer_epochs, best["max_len"]),
            "lr": best["lr"],
            "epochs": longer_epochs,
            "batch_size": BATCH_SIZE,
            "max_len": best["max_len"],
            "note": "one extra epoch at the winning learning rate",
        },
        {
            "stage": 3,
            "name": trial_name(best["lr"], longer_epochs, shorter_len),
            "lr": best["lr"],
            "epochs": longer_epochs,
            "batch_size": BATCH_SIZE,
            "max_len": shorter_len,
            "note": "shorter context window to test the truncation trade-off",
        },
    ]


def build_run_config(
    base_config: dict[str, Any],
    trial: dict[str, Any],
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Overlay one grid point on the shipped config and re-validate it."""
    config = copy.deepcopy(base_config)
    config.update(
        {
            "lr": float(trial["lr"]),
            "epochs": int(trial["epochs"]),
            "batch_size": int(trial["batch_size"]),
            "max_len": int(trial["max_len"]),
            "output_dir": str(TRIAL_MODEL_ROOT / trial["name"]),
            "max_train_samples": max_samples,
            "max_eval_samples": max_samples,
        }
    )
    validate_config(config)
    return config


def run_trial(
    config: dict[str, Any],
    tokenizer: Any,
    train_split: Any,
    validation_split: Any,
    normalizer: Any,
) -> dict[str, float]:
    """Train one trial with no checkpointing and score it on validation."""
    import torch

    from src.data_utils import make_token_classification_dataloader as make_loader

    train_loader = make_loader(
        limit_split(train_split, config.get("max_train_samples"), "train"),
        tokenizer,
        batch_size=int(config["batch_size"]),
        max_len=int(config["max_len"]),
        shuffle=True,
        seed=int(config["seed"]),
        normalization=normalizer,
        label_all_subwords=bool(config["label_all_subwords"]),
    )
    eval_loader = make_loader(
        limit_split(validation_split, config.get("max_eval_samples"), "eval"),
        tokenizer,
        batch_size=int(config["eval_batch_size"]),
        max_len=int(config["max_len"]),
        shuffle=False,
        normalization=normalizer,
        label_all_subwords=bool(config["label_all_subwords"]),
    )

    model, _ = build_model_and_tokenizer(
        str(config["model_name"]), num_labels=len(LABEL2ID)
    )
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainer = make_trainer(
        model, config, train_loader, eval_loader, save_checkpoints=False
    )
    trainer.train()
    metrics = trainer.evaluate()

    del trainer, model, train_loader, eval_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def build_summary(results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in results if "f1" in row]
    best = max(completed, key=lambda row: row["f1"]) if completed else None
    return {
        "model": "arabert",
        "approach": "arabic_transformer",
        "owner": "member_3",
        "selection_split": "validation",
        "baseline": {"lr": 3e-5, "epochs": 3, "batch_size": 8, "max_len": 128, "f1": 0.9128956778950157},
        "grid_note": (
            "batch_size fixed at 8 (4 GB VRAM limit); lr, epochs and max_len "
            "searched in three stages"
        ),
        "results": results,
        "failures": failures,
        "best": best,
        "next_step": (
            None
            if best is None
            else (
                "write lr={lr}, epochs={epochs}, batch_size={batch_size}, "
                "max_len={max_len} into configs/arabert.yaml, then retrain once with "
                "`make train MODEL=arabert`"
            ).format(**best)
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AraBERT — Hyperparameter Search (issue 5.3)",
        "",
        "**Owner:** Member 3 · **Selection split:** validation F1 (test is untouched "
        "until issue 5.4)",
        "",
        "`batch_size` is not searched: 8 is already the largest the 4 GB RTX 3050 holds "
        "at `max_len 128` with gradient checkpointing, and accumulation already yields "
        "an effective batch of 16. Every other setting (bf16, gradient checkpointing, "
        "warmup, weight decay, seed) is identical to the baseline so the comparison "
        "isolates the tuned knobs.",
        "",
        "| Stage | Run | LR | Epochs | Max len | Precision | Recall | F1 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in summary["results"]:
        if "f1" not in row:
            continue
        lines.append(
            "| {stage} | `{name}` | {lr:g} | {epochs} | {max_len} | "
            "{precision:.4f} | {recall:.4f} | **{f1:.4f}** |".format(**row)
        )
    for row in summary["failures"]:
        lines.append(
            "| {stage} | `{name}` | {lr:g} | {epochs} | {max_len} | — | — | {error} |".format(
                **row
            )
        )
    best = summary["best"]
    if best:
        lines += [
            "",
            f"**Best:** `{best['name']}` with validation F1 **{best['f1']:.4f}** "
            f"(precision {best['precision']:.4f}, recall {best['recall']:.4f}).",
            "",
            "Baseline for reference: lr 3e-5 / epochs 3 / max_len 128, validation F1 "
            f"{summary['baseline']['f1']:.4f} (issue 5.1 full run).",
            "",
            "Which run wins is measured on **validation**; the number that goes into the "
            "phase-1 report is the **test** score from a single retrain with the winning "
            "config (issue 5.4).",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="SMOKE TEST: cap train and validation rows per trial (not reportable)",
    )
    arguments = parser.parse_args(argv)

    base_config = resolve_config({}, load_config(arguments.config))
    validate_config(base_config)

    if arguments.dry_run:
        for trial in stage1_grid():
            print(f"# stage {trial['stage']}: {trial['name']} — {trial['note']}")
        print("# stage 2: best lr with one extra epoch")
        print("# stage 3: best lr+epochs with max_len 96")
        return 0

    ensure_dirs()
    TRIAL_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if arguments.max_samples:
        LOGGER.warning(
            "SMOKE TEST: %d train / %d validation rows per trial (not reportable)",
            arguments.max_samples,
            arguments.max_samples,
        )

    from data.load_data import load_arabic_wikiann

    LOGGER.info("Loading WikiANN ar through the canonical loader")
    dataset = load_arabic_wikiann(cache_dir=base_config.get("cache_dir"))
    train_split, validation_split = dataset["train"], dataset["validation"]

    set_seed(int(base_config["seed"]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_config["model_name"]), use_fast=True
    )
    normalizer = normalization_from_config(base_config.get("normalization"))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def execute(trials: list[dict[str, Any]]) -> None:
        for trial in trials:
            LOGGER.info(
                "Trial %s (stage %s): lr=%s epochs=%s batch=%s max_len=%s",
                trial["name"], trial["stage"], trial["lr"], trial["epochs"],
                trial["batch_size"], trial["max_len"],
            )
            config = build_run_config(base_config, trial, arguments.max_samples)
            started = time.time()
            try:
                metrics = run_trial(
                    config, tokenizer, train_split, validation_split, normalizer
                )
            except Exception as error:  # one OOM must not throw away the whole search
                LOGGER.warning("Trial %s failed: %s", trial["name"], error)
                failures.append({**trial, "error": f"{type(error).__name__}: {error}"})
                continue
            row = {
                **{key: trial[key] for key in ("stage", "name", "lr", "epochs", "batch_size", "max_len", "note")},
                "precision": metrics.get("eval_precision", 0.0),
                "recall": metrics.get("eval_recall", 0.0),
                "f1": metrics.get("eval_f1", 0.0),
                "loss": metrics.get("eval_loss", 0.0),
                "per_entity": {
                    label: metrics.get(f"eval_{label}_f1", 0.0)
                    for label in ("PER", "ORG", "LOC")
                },
                "seconds": round(time.time() - started, 1),
            }
            results.append(row)
            LOGGER.info("Trial %s: validation F1 %.4f", trial["name"], row["f1"])

    execute(stage1_grid())
    completed = [row for row in results if "f1" in row]
    if not completed:
        LOGGER.error("No trial completed; see failures in %s", SUMMARY_PATH)
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(
            json.dumps(build_summary(results, failures), indent=2) + "\n",
            encoding="utf-8",
        )
        return 1

    stage1_best = max(completed, key=lambda row: row["f1"])
    execute(follow_up_trials(stage1_best))

    summary = build_summary(results, failures)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    SUMMARY_PATH.with_suffix(".md").write_text(render_markdown(summary), encoding="utf-8")
    LOGGER.info("Wrote %s", SUMMARY_PATH)
    print(json.dumps(summary["best"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
