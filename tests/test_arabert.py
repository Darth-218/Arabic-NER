"""Offline unit tests for Member 3's AraBERT pipeline (Epic 5, issue 5.2).

No network, dataset download, model download, torch or transformers required:
only the pure CLI/config helpers are exercised (see tests/README.md).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.approach_3_arabert.train import (
    CLI_OVERRIDABLE,
    CONFIG_KEYS,
    DEFAULTS,
    build_parser,
    compute_metrics_fn,
    limit_split,
    load_config,
    resolve_config,
    validate_config,
)
from src.common.config import APPROACHES, PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "arabert.yaml"

CANONICAL_TRAIN_FLAGS = {
    "config",
    "output_dir",
    "seed",
    "epochs",
    "batch_size",
    "lr",
}


def base_config(**overrides: Any) -> dict[str, Any]:
    """A minimal config that passes validate_config()."""
    config: dict[str, Any] = {
        "model": "arabert",
        "approach": APPROACHES["arabert"],
        "owner": "member_3",
        "normalization": {"strip_diacritics": True},
    }
    config.update(DEFAULTS)
    config.update(overrides)
    return config


class StubSplit:
    """Stand-in for a Hugging Face Dataset split."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.selected: list[int] | None = None

    def __len__(self) -> int:
        return self.size

    def select(self, indices: list[int]) -> "StubSplit":
        self.selected = list(indices)
        return StubSplit(len(self.selected))


class StubEvalPrediction:
    def __init__(self, predictions: Any, label_ids: Any) -> None:
        self.predictions = predictions
        self.label_ids = label_ids


# --- CLI contract (ROADMAP.md §2.4) ----------------------------------------


def test_parser_exposes_every_canonical_flag() -> None:
    parser = build_parser()
    dests = {action.dest for action in parser._actions}
    assert CANONICAL_TRAIN_FLAGS <= dests


def test_every_canonical_flag_can_override_the_yaml() -> None:
    assert CANONICAL_TRAIN_FLAGS - {"config"} <= set(CLI_OVERRIDABLE)


def test_cli_defaults_do_not_override_yaml() -> None:
    parsed = vars(build_parser().parse_args(["--config", str(CONFIG_PATH)]))
    assert all(parsed[key] is None for key in CLI_OVERRIDABLE)


# --- config resolution -----------------------------------------------------


def test_yaml_overrides_defaults_and_cli_overrides_yaml() -> None:
    resolved = resolve_config(
        {"epochs": 1, "lr": None},
        {"epochs": 3, "lr": 5e-5},
    )
    assert resolved["epochs"] == 1  # explicitly passed
    assert resolved["lr"] == 5e-5  # unset flag keeps YAML
    assert resolved["batch_size"] == DEFAULTS["batch_size"]  # absent everywhere


def test_arabert_yaml_is_canonical_and_valid() -> None:
    pytest.importorskip("yaml")
    file_config = load_config(CONFIG_PATH)
    assert set(file_config) <= CONFIG_KEYS
    assert file_config["model"] == "arabert"
    assert file_config["approach"] == "arabic_transformer"
    assert file_config["owner"] == "member_3"
    assert file_config["model_name"].startswith("aubmindlab/")
    assert file_config["output_dir"] == "results/models/arabert"
    validate_config(resolve_config({}, file_config))


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"batch_size": -1},
        {"max_len": 1024},
        {"lr": 0},
        {"weight_decay": -0.1},
        {"warmup_ratio": 1.0},
        {"seed": -1},
        {"bf16": True, "fp16": True},
        {"max_train_samples": 0},
        {"normalization": "none"},
    ],
)
def test_validate_config_rejects_impossible_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_config(base_config(**overrides))


def test_validate_config_rejects_typos() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        validate_config(base_config(epoch=3))


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "marbert"},
        {"approach": "classical"},
        {"owner": "member_4"},
    ],
)
def test_validate_config_enforces_ownership_metadata(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_config(base_config(**overrides))


def test_validate_config_accepts_the_shipped_config() -> None:
    validate_config(base_config())


# --- helpers ---------------------------------------------------------------


def test_limit_split_selects_the_first_n_rows() -> None:
    split = StubSplit(20_000)
    limited = limit_split(split, 200, "train")
    assert len(limited) == 200
    assert split.selected == list(range(200))


def test_limit_split_without_a_limit_returns_the_same_split() -> None:
    split = StubSplit(20_000)
    assert limit_split(split, None, "train") is split


@pytest.mark.parametrize("bad_limit", [0, -5, 25])
def test_limit_split_rejects_invalid_limits(bad_limit: int) -> None:
    with pytest.raises(ValueError):
        limit_split(StubSplit(20), bad_limit, "train")


def test_compute_metrics_ignores_padding_and_continuation_subwords() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("seqeval")
    # 7 labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
    logits = np.zeros((1, 4, 7), dtype=float)
    for position, label_id in enumerate((1, 2, 3, 0)):
        logits[0, position, label_id] = 10.0
    labels = np.array([[1, 2, 3, -100]])  # last position is padding

    scores = compute_metrics_fn(StubEvalPrediction(logits, labels))

    assert scores["f1"] == pytest.approx(1.0)
    assert {"PER_f1", "ORG_f1", "LOC_f1"} <= set(scores)


# --- hyperparameter search (issue 5.3) -------------------------------------


def _trial(**overrides: Any) -> dict[str, Any]:
    trial = {
        "stage": 1,
        "name": "lr3e5_e3_bs8_len128",
        "lr": 3e-5,
        "epochs": 3,
        "batch_size": 8,
        "max_len": 128,
        "note": "test",
    }
    trial.update(overrides)
    return trial


def test_trial_names_are_stable_and_readable() -> None:
    from src.approach_3_arabert import hparam_search

    assert hparam_search.trial_name(3e-5, 3, 128) == "lr3e5_e3_bs8_len128"
    assert hparam_search.trial_name(2e-5, 4, 96) == "lr2e5_e4_bs8_len96"
    assert hparam_search.trial_name(5e-5, 3, 128) == "lr5e5_e3_bs8_len128"


def test_stage1_grid_sweeps_learning_rate() -> None:
    from src.approach_3_arabert import hparam_search

    trials = hparam_search.stage1_grid()
    assert [trial["lr"] for trial in trials] == [2e-5, 3e-5, 5e-5]
    assert len({trial["name"] for trial in trials}) == len(trials)
    assert all(trial["epochs"] == 3 and trial["max_len"] == 128 for trial in trials)


def test_follow_up_trials_add_an_epoch_then_shorten_the_context() -> None:
    from src.approach_3_arabert import hparam_search

    stages = hparam_search.follow_up_trials({"lr": 5e-5, "epochs": 3, "max_len": 128})
    assert [stage["stage"] for stage in stages] == [2, 3]
    assert stages[0]["epochs"] == 4 and stages[0]["max_len"] == 128
    assert stages[1]["epochs"] == 4 and stages[1]["max_len"] == 96
    assert {stage["lr"] for stage in stages} == {5e-5}


def test_build_run_config_overrides_only_the_tuned_knobs() -> None:
    from src.approach_3_arabert import hparam_search

    before = base_config()
    config = hparam_search.build_run_config(before, _trial(lr=2e-5, epochs=4, max_len=96))

    assert (config["lr"], config["epochs"], config["max_len"]) == (2e-5, 4, 96)
    assert config["model_name"] == before["model_name"]
    assert config["output_dir"].endswith("arabert_hparam/lr3e5_e3_bs8_len128")
    assert config["max_train_samples"] is None and config["max_eval_samples"] is None


def test_build_run_config_rejects_an_impossible_trial() -> None:
    from src.approach_3_arabert import hparam_search

    with pytest.raises(ValueError):
        hparam_search.build_run_config(base_config(), _trial(lr=0.0))


def test_build_summary_picks_the_best_validation_score() -> None:
    from src.approach_3_arabert import hparam_search

    rows = [
        {**_trial(name="lr2e5_e3_bs8_len128", lr=2e-5), "precision": 0.88, "recall": 0.89, "f1": 0.885},
        {**_trial(name="lr5e5_e3_bs8_len128", lr=5e-5), "precision": 0.90, "recall": 0.91, "f1": 0.905},
    ]
    failures = [{**_trial(name="lr9e5_e3_bs8_len128"), "error": "OutOfMemoryError: test"}]
    summary = hparam_search.build_summary(rows, failures)

    assert summary["selection_split"] == "validation"
    assert summary["best"]["name"] == "lr5e5_e3_bs8_len128"
    assert summary["failures"] == failures
    assert "lr=5e-05" in summary["next_step"]

    markdown = hparam_search.render_markdown(summary)
    assert "lr5e5_e3_bs8_len128" in markdown
    assert "validation" in markdown
    assert "OutOfMemoryError" in markdown
