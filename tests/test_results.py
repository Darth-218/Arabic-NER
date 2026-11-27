"""Tests for the canonical results writer and README table (issue 2.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.results as results_module
from src.results import read_metrics, results_table, update_readme_table, write_metrics


def test_write_metrics_split_specific_default_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(results_module, "METRICS_DIR", tmp_path)
    test_path = write_metrics(
        model="mbert", split="test", precision=0.5, recall=0.5, f1=0.5
    )
    validation_path = write_metrics(
        model="mbert", split="validation", precision=0.4, recall=0.4, f1=0.4
    )
    assert test_path.name == "mbert.json"
    assert validation_path.name == "mbert_validation.json"
    assert read_metrics(validation_path)["split"] == "validation"


def test_write_metrics_schema_is_canonical(tmp_path: Path) -> None:
    path = write_metrics(
        model="marbert",
        split="test",
        precision=0.5,
        recall=0.4,
        f1=0.44,
        per_entity={"PER": {"precision": 0.6, "f1": 0.5, "support": 10}},
        config={"model": "marbert"},
        approach="advanced_arabic_transformer",
        owner="member_4",
        out=tmp_path / "marbert.json",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {
        "model",
        "approach",
        "owner",
        "split",
        "precision",
        "recall",
        "f1",
        "per_entity",
        "config",
        "timestamp",
    }
    assert set(data["per_entity"]) == {"PER", "ORG", "LOC"}
    assert data["per_entity"]["PER"]["support"] == 10


def test_write_metrics_rejects_unknown_model_and_missing_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        write_metrics(
            model="not-a-model",
            split="test",
            precision=0.1,
            recall=0.1,
            f1=0.1,
            out=tmp_path / "bad.json",
        )
    path = tmp_path / "missing.json"
    path.write_text(json.dumps({"model": "mbert"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        read_metrics(path)


def test_results_table_marks_missing_models_and_reads_valid(tmp_path: Path) -> None:
    assert "| Multilingual | mBERT | - | - | - |" in results_table(tmp_path)
    write_metrics(
        model="mbert",
        split="test",
        precision=0.1,
        recall=0.2,
        f1=0.3,
        out=tmp_path / "mbert.json",
    )
    assert "| Multilingual | mBERT | 0.1000 | 0.2000 | 0.3000 |" in results_table(tmp_path)


def test_update_readme_table_requires_exactly_one_table(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(results_module, "METRICS_DIR", tmp_path / "empty_metrics")
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n\n"
        "| Approach | Model | Precision | Recall | F1-Score |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| Classical | CRF | - | - | - |\n",
        encoding="utf-8",
    )
    update_readme_table(readme)
    text = readme.read_text(encoding="utf-8")
    assert "| Classical | CRF | - | - | - |" in text
    assert "| Advanced Arabic | MARBERT |" in text
