"""Exploratory data analysis for Arabic WikiANN (issue 1.2).

Reproducible script (no notebook) per TL guidance:
    python data/eda.py

Writes `reports/eda/summary.md`, `stats.json`, and figures under
`reports/eda/figures/`. Loads data only through `data/load_data.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data import EXPECTED_SPLITS, load_arabic_wikiann
from src.common.config import ID2LABEL, LABELS, REPORTS_DIR

DEFAULT_OUT_DIR = REPORTS_DIR / "eda"
ENTITY_NAMES = ("PER", "ORG", "LOC")


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def length_stats(lengths: Sequence[int]) -> dict[str, float | int]:
    return {
        "count": len(lengths),
        "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "median": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths, default=0),
    }


def label_counts(tag_sequences: Iterable[Sequence[int]]) -> dict[str, int]:
    counts = Counter(label for tags in tag_sequences for label in (ID2LABEL[int(t)] for t in tags))
    return {label: counts.get(label, 0) for label in LABELS}


def entity_span_counts(tag_sequences: Iterable[Sequence[int]]) -> dict[str, int]:
    counts = {name: 0 for name in ENTITY_NAMES}
    for tags in tag_sequences:
        for tag in tags:
            label = ID2LABEL[int(tag)]
            if label.startswith("B-"):
                counts[label[2:]] = counts.get(label[2:], 0) + 1
    return counts


def vocab_stats(train_tokens: Sequence[Sequence[str]], other: Sequence[Sequence[str]]) -> dict[str, Any]:
    vocabulary = Counter(token for sentence in train_tokens for token in sentence)
    total = sum(vocabulary.values())
    other_tokens = [token for sentence in other for token in sentence]
    oov = sum(1 for token in other_tokens if token not in vocabulary)
    rare = sum(count for count in vocabulary.values() if count == 1)
    return {
        "train_vocab_size": len(vocabulary),
        "train_tokens": total,
        "tokens": len(other_tokens),
        "oov_tokens": oov,
        "oov_rate": round(oov / len(other_tokens), 4) if other_tokens else 0.0,
        "rare_tokens": rare,
        "rare_share": round(rare / total, 4) if total else 0.0,
    }


def sequence_overlap(exact_sets: dict[str, set[tuple[str, ...]]]) -> dict[str, int]:
    pairs = {}
    names = list(exact_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pairs[f"{left}<->{right}"] = len(exact_sets[left] & exact_sets[right])
    return pairs


def word_overlap(vocabularies: dict[str, set[str]]) -> dict[str, float]:
    pairs = {}
    names = list(vocabularies)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            union = vocabularies[left] | vocabularies[right]
            intersection = vocabularies[left] & vocabularies[right]
            pairs[f"{left}<->{right}"] = round(
                len(intersection) / len(union), 4
            ) if union else 0.0
    return pairs


def plot_label_distribution(distributions: dict[str, dict[str, int]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(distributions)
    width = 0.8 / max(1, len(splits))
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for offset, split in enumerate(splits):
        values = [distributions[split][label] for label in LABELS]
        positions = [index + offset * width for index in range(len(LABELS))]
        axis.bar(positions, values, width=width, label=split)
    axis.set_xticks([index + width * (len(splits) - 1) / 2 for index in range(len(LABELS))])
    axis.set_xticklabels(LABELS)
    axis.set_ylabel("Tag count")
    axis.set_title("WikiANN-ar label distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)


def plot_lengths(lengths: dict[str, list[int]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for split, values in lengths.items():
        axis.hist(values, bins=50, alpha=0.55, label=split, density=True)
    axis.set_xlabel("Words per sequence")
    axis.set_ylabel("Density")
    axis.set_title("WikiANN-ar sequence length distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)


def plot_entity_spans(entities: dict[str, dict[str, int]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(entities)
    width = 0.8 / max(1, len(splits))
    figure, axis = plt.subplots(figsize=(8, 4.2))
    for offset, split in enumerate(splits):
        values = [entities[split][name] for name in ENTITY_NAMES]
        positions = [index + offset * width for index in range(len(ENTITY_NAMES))]
        axis.bar(positions, values, width=width, label=split)
    axis.set_xticks([index + width * (len(splits) - 1) / 2 for index in range(len(ENTITY_NAMES))])
    axis.set_xticklabels(ENTITY_NAMES)
    axis.set_ylabel("Entity spans")
    axis.set_title("WikiANN-ar entity type distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)


def plot_top_tokens(vocabulary: Counter[str], out: Path, top: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    common = vocabulary.most_common(top)[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh([f"#{rank}" for rank in range(len(common), 0, -1)], [count for _, count in common])
    axis.set_xlabel("Frequency")
    axis.set_title(f"Top {top} train tokens (rank order; see summary.md)")
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run WikiANN-ar EDA")
    parser.add_argument("--out-dir", default=None, help="default: reports/eda")
    arguments = parser.parse_args(argv)
    out_dir = Path(arguments.out_dir) if arguments.out_dir else DEFAULT_OUT_DIR
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_arabic_wikiann()

    labels = {split: label_counts(dataset[split]["ner_tags"]) for split in dataset}
    entities = {split: entity_span_counts(dataset[split]["ner_tags"]) for split in dataset}
    lengths = {
        split: [len(tokens) for tokens in dataset[split]["tokens"]]
        for split in dataset
    }
    vocabulary = Counter(
        token for sentence in dataset["train"]["tokens"] for token in sentence
    )
    oov = {
        split: vocab_stats(dataset["train"]["tokens"], dataset[split]["tokens"])
        for split in dataset
        if split != "train"
    }
    exact_sets = {
        split: {tuple(tokens) for tokens in dataset[split]["tokens"]}
        for split in dataset
    }
    vocabularies = {
        split: set(vocabulary) if split == "train" else {
            token for sentence in dataset[split]["tokens"] for token in sentence
        }
        for split in dataset
    }

    stats = {
        "splits": {
            split: {
                "sequences": len(dataset[split]),
                "tokens": sum(lengths[split]),
                "expected_sequences": EXPECTED_SPLITS.get(split),
                "label_counts": labels[split],
                "entity_spans": entities[split],
                "length": length_stats(lengths[split]),
            }
            for split in dataset
        },
        "vocabulary": {
            "train_size": len(vocabulary),
            "oov": oov,
            "rare": {
                "count": sum(1 for count in vocabulary.values() if count == 1),
                "share": round(
                    sum(1 for count in vocabulary.values() if count == 1) / len(vocabulary),
                    4,
                ) if vocabulary else 0.0,
            },
            "top_tokens": vocabulary.most_common(20),
        },
        "exact_sequence_overlap": sequence_overlap(exact_sets),
        "word_jaccard": word_overlap(vocabularies),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    plot_label_distribution(labels, figures_dir / "label_distribution.png")
    plot_lengths(lengths, figures_dir / "sequence_lengths.png")
    plot_entity_spans(entities, figures_dir / "entity_distribution.png")
    plot_top_tokens(vocabulary, figures_dir / "top_tokens.png")

    total_tags = sum(labels["train"].values())
    o_share = labels["train"]["O"] / total_tags if total_tags else 0.0
    entity_totals = {name: entities["train"][name] for name in ENTITY_NAMES}
    leakage = stats["exact_sequence_overlap"]

    lines = [
        "# WikiANN-ar EDA summary",
        "",
        "Generated by `python data/eda.py` (script-based reproducibility; no notebook).",
        "",
        "## Split overview",
        "",
        "| Split | Sequences | Tokens | Expected | Mean length | p95 | Max |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, values in stats["splits"].items():
        length = values["length"]
        lines.append(
            f"| {split} | {values['sequences']} | {values['tokens']} | "
            f"{values['expected_sequences']} | {length['mean']} | {length['p95']} | {length['max']} |"
        )

    lines += [
        "",
        "## Label distribution (train)",
        "",
        "| Label | Count | Share |",
        "| :--- | ---: | ---: |",
    ]
    for label in LABELS:
        count = labels["train"][label]
        lines.append(f"| {label} | {count} | {count / total_tags:.4f} |")

    lines += [
        "",
        "## Entity spans (B- tags)",
        "",
        "| Split | PER | ORG | LOC |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for split, values in entities.items():
        lines.append(
            f"| {split} | {values['PER']} | {values['ORG']} | {values['LOC']} |"
        )

    lines += [
        "",
        "## Vocabulary and OOV",
        "",
        f"- Train vocabulary: **{len(vocabulary):,}** types over "
        f"**{total_tags:,}** token positions.",
        f"- Rare train types (frequency 1): **{stats['vocabulary']['rare']['count']:,}** "
        f"({stats['vocabulary']['rare']['share']:.2%} of the vocabulary).",
    ]
    for split, values in oov.items():
        lines.append(
            f"- {split} OOV rate: **{values['oov_rate']:.2%}** "
            f"({values['oov_tokens']:,} / {values['tokens']:,} tokens)"
        )

    lines += [
        "",
        "## Cross-split overlap",
        "",
        "- Exact duplicate sequences (deduplicated sets): "
        + ", ".join(f"{pair}={count}" for pair, count in leakage.items()),
        "- Word-vocabulary Jaccard: "
        + ", ".join(f"{pair}={value:.4f}" for pair, value in stats["word_jaccard"].items()),
        "- Top train tokens: "
        + ", ".join(f"`{token}` ({count})" for token, count in stats["vocabulary"]["top_tokens"][:10]),
        "",
        "## Findings",
        "",
        f"- `O` dominates the tag distribution at **{o_share:.2%}** of train tags.",
        f"- Entity imbalance: PER={entity_totals['PER']}, ORG={entity_totals['ORG']}, "
        f"LOC={entity_totals['LOC']} spans in train.",
        f"- Sequence length is concentrated below the p95 of **{stats['splits']['train']['length']['p95']:.0f}** "
        f"words (max {stats['splits']['train']['length']['max']}).",
        f"- Test OOV is **{oov['test']['oov_rate']:.2%}**, so subword models should still see "
        "unseen surface forms.",
        "- Cross-split leakage is quantified in `reports/eda/data_quality.md`.",
        "",
        "## Figures",
        "",
        "- `figures/label_distribution.png`",
        "- `figures/sequence_lengths.png`",
        "- `figures/entity_distribution.png`",
        "- `figures/top_tokens.png`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote EDA artifacts to {out_dir}")


if __name__ == "__main__":
    main()
