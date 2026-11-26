"""Optional helper for issue 6.3 — MARBERT hyperparameter search.

NOTE: this file is an addition to the canonical tree in ROADMAP.md §2.1.
Per §2's "do not deviate" rule, Member 1 should acknowledge it in the roadmap
(or we can fold these commands into a Makefile target) before merging.

Each run reuses the canonical paths, then archives its metrics/predictions so
the next run cannot overwrite the evidence. Use --dry-run to inspect the plan.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import PROJECT_ROOT, RESULTS_DIR

BASE_CONFIG = PROJECT_ROOT / "configs" / "marbert.yaml"
TRAIN = PROJECT_ROOT / "src" / "approach_4_marbert" / "train.py"
PREDICT = PROJECT_ROOT / "src" / "approach_4_marbert" / "predict.py"
EVALUATE = PROJECT_ROOT / "src" / "evaluate.py"
ARCHIVE_DIR = PROJECT_ROOT / "reports" / "analysis" / "marbert_hparam_runs"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "analysis" / "marbert_hparam_search.json"

# Keep the grid small: every row trains a ~163M-parameter model.
GRID: list[dict[str, Any]] = [
    {"name": "lr1e5_e2", "lr": 1e-5, "epochs": 2},
    {"name": "lr2e5_e3", "lr": 2e-5, "epochs": 3},
    {"name": "lr3e5_e2", "lr": 3e-5, "epochs": 2},
]


def commands_for(run: dict[str, Any]) -> list[list[str]]:
    """Return the three canonical commands for one grid point."""
    python = sys.executable
    model_dir = RESULTS_DIR / "models" / "marbert_hparam" / run["name"]
    return [
        [
            python, str(TRAIN), "--config", str(BASE_CONFIG),
            "--lr", str(run["lr"]), "--epochs", str(run["epochs"]),
            "--output_dir", str(model_dir),
        ],
        [
            python, str(PREDICT), "--model_dir", str(model_dir),
            "--split", "test", "--config", str(BASE_CONFIG),
            "--out", str(RESULTS_DIR / "predictions" / "marbert_test.jsonl"),
        ],
        [
            python, str(EVALUATE), "--model", "marbert", "--split", "test",
            "--config", str(BASE_CONFIG),
            "--out", str(RESULTS_DIR / "metrics" / "marbert.json"),
        ],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.dry_run:
        for run in GRID:
            print(f"# {run['name']}: lr={run['lr']}, epochs={run['epochs']}")
            for command in commands_for(run):
                print(" ".join(command))
            print()
        return 0

    results: list[dict[str, Any]] = []
    for run in GRID:
        print(f"=== {run['name']} ===", flush=True)
        for command in commands_for(run):
            subprocess.run(command, check=True)  # noqa: S603 - local, fixed commands

        run_dir = ARCHIVE_DIR / run["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_copy = run_dir / "metrics.json"
        shutil.copy(RESULTS_DIR / "metrics" / "marbert.json", metrics_copy)
        shutil.copy(
            RESULTS_DIR / "predictions" / "marbert_test.jsonl",
            run_dir / "predictions_test.jsonl",
        )
        scores = json.loads(metrics_copy.read_text(encoding="utf-8"))
        results.append(
            {
                "name": run["name"],
                "lr": run["lr"],
                "epochs": run["epochs"],
                "f1": scores["f1"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "artifacts": str(run_dir),
            }
        )

    best = max(results, key=lambda row: row["f1"])
    summary = {
        "model": "marbert",
        "grid": GRID,
        "results": results,
        "best": best,
        "next_step": (
            f"python src/approach_4_marbert/train.py --config configs/marbert.yaml "
            f"--lr {best['lr']} --epochs {best['epochs']}"
        ),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["best"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
