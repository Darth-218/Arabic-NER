"""Data integrity and train/test leakage report (issue 1.6).

    python data/quality_report.py

Checks duplicate sequences (exact and normalized) within and across splits,
all-O sequences, BIO transition validity, and label coverage. Writes
`reports/eda/data_quality.md` + `.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data import load_arabic_wikiann
from src.common.config import ID2LABEL, LABEL2ID, REPORTS_DIR
from src.data_utils import normalize_arabic

DEFAULT_MD = REPORTS_DIR / "eda" / "data_quality.md"
DEFAULT_JSON = REPORTS_DIR / "eda" / "data_quality.json"


def group_duplicates(
    sequences_per_split: dict[str, list[Sequence[str]]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    """Group sequences by key and summarize within/cross-split duplicates."""
    groups: dict[tuple[str, ...], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for split, sequences in sequences_per_split.items():
        for index, sequence in enumerate(sequences):
            groups[tuple(sequence)][split].append(index)

    within = {
        split: {"duplicate_groups": 0, "rows_in_duplicate_groups": 0}
        for split in sequences_per_split
    }
    cross: dict[str, dict[str, Any]] = {}
    for key, by_split in groups.items():
        for split, indices in by_split.items():
            if len(indices) > 1:
                within[split]["duplicate_groups"] += 1
                within[split]["rows_in_duplicate_groups"] += len(indices)
        splits = list(by_split)
        for i, left in enumerate(splits):
            for right in splits[i + 1:]:
                pair = f"{left}<->{right}"
                entry = cross.setdefault(
                    pair, {"groups": 0, "pairs": 0, "examples": []}
                )
                entry["groups"] += 1
                entry["pairs"] += len(by_split[left]) * len(by_split[right])
                if len(entry["examples"]) < 5:
                    entry["examples"].append(
                        {
                            left: by_split[left][0],
                            right: by_split[right][0],
                            "tokens": list(key),
                        }
                    )
    return within, cross


def bio_errors(tag_sequences: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for index, tags in enumerate(tag_sequences):
        previous = None
        for position, tag in enumerate(tags):
            label = ID2LABEL[int(tag)]
            if label.startswith("I-"):
                entity = label[2:]
                if previous not in {f"B-{entity}", f"I-{entity}"}:
                    errors.append(
                        {
                            "index": index,
                            "position": position,
                            "tag": label,
                            "previous": previous,
                        }
                    )
            previous = label
    return errors


def summarize_split(dataset: Any, split: str) -> dict[str, Any]:
    tokens = dataset[split]["tokens"]
    tags = dataset[split]["ner_tags"]
    lengths = [len(sequence) for sequence in tokens]
    entity_presence = {name: 0 for name in ("PER", "ORG", "LOC")}
    for sequence in tags:
        seen = {ID2LABEL[int(tag)][2:] for tag in sequence if int(tag) != LABEL2ID["O"]}
        for name in entity_presence:
            entity_presence[name] += int(name in seen)
    return {
        "sequences": len(tokens),
        "tokens": sum(lengths),
        "all_O_sequences": sum(
            1 for sequence in tags if all(int(tag) == LABEL2ID["O"] for tag in sequence)
        ),
        "token_tag_mismatches": sum(
            1 for words, labels in zip(tokens, tags) if len(words) != len(labels)
        ),
        "max_length": max(lengths, default=0),
        "entity_presence": entity_presence,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report WikiANN-ar data quality")
    parser.add_argument("--md", default=None, help="default: reports/eda/data_quality.md")
    parser.add_argument("--json", default=None, help="default: reports/eda/data_quality.json")
    parser.add_argument(
        "--fail-on-leakage",
        action="store_true",
        help="exit non-zero if exact cross-split duplicates exist",
    )
    arguments = parser.parse_args(argv)
    md_path = Path(arguments.md) if arguments.md else DEFAULT_MD
    json_path = Path(arguments.json) if arguments.json else DEFAULT_JSON
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_arabic_wikiann()
    splits = list(dataset)

    exact_tokens = {split: [list(seq) for seq in dataset[split]["tokens"]] for split in splits}
    normalized_tokens = {
        split: [
            [normalize_arabic(token) for token in sequence]
            for sequence in exact_tokens[split]
        ]
        for split in splits
    }

    exact_within, exact_cross = group_duplicates(exact_tokens)
    normalized_within, normalized_cross = group_duplicates(normalized_tokens)

    integrity = {split: summarize_split(dataset, split) for split in splits}
    bio = {
        split: bio_errors(dataset[split]["ner_tags"])
        for split in splits
    }

    exact_leak = sum(entry["pairs"] for entry in exact_cross.values())
    normalized_leak = sum(entry["pairs"] for entry in normalized_cross.values())

    payload = {
        "splits": integrity,
        "duplicates": {
            "exact": {"within": exact_within, "cross": exact_cross},
            "normalized": {"within": normalized_within, "cross": normalized_cross},
        },
        "bio_errors": {split: len(errors) for split, errors in bio.items()},
        "leakage": {
            "exact_cross_split_pairs": exact_leak,
            "normalized_cross_split_pairs": normalized_leak,
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def example_text(entry: dict[str, Any]) -> str:
        tokens = entry["tokens"]
        text = " ".join(tokens[:12]) + (" …" if len(tokens) > 12 else "")
        splits_text = ", ".join(
            f"{key} #{value}" for key, value in entry.items() if key != "tokens"
        )
        return f"`{text}` ({splits_text})"

    lines = [
        "# WikiANN-ar data quality & leakage report",
        "",
        "Generated by `python data/quality_report.py`.",
        "",
        "## Verdict",
        "",
    ]
    if exact_leak or normalized_leak:
        lines.append(
            f"**Leakage detected:** {exact_leak} exact and {normalized_leak} "
            "normalized cross-split duplicate sequence pairs. Documented below; "
            "de-duplicate or exclude overlaps before treating scores as an "
            "unseen-test estimate."
        )
    else:
        lines.append(
            "**No cross-split duplicate sequences detected (exact or normalized).**"
        )

    lines += [
        "",
        "## Integrity summary",
        "",
        "| Split | Sequences | Tokens | All-O seqs | Max len | Token/tag mismatches | Exact dup groups (within) | BIO errors |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in splits:
        values = integrity[split]
        lines.append(
            f"| {split} | {values['sequences']} | {values['tokens']} | "
            f"{values['all_O_sequences']} | {values['max_length']} | "
            f"{values['token_tag_mismatches']} | "
            f"{exact_within[split]['duplicate_groups']} | {len(bio[split])} |"
        )

    lines += [
        "",
        "## Entity presence (sequences containing the type)",
        "",
        "| Split | PER | ORG | LOC |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for split in splits:
        presence = integrity[split]["entity_presence"]
        lines.append(
            f"| {split} | {presence['PER']} | {presence['ORG']} | {presence['LOC']} |"
        )

    lines += [
        "",
        "## Cross-split duplicates",
        "",
        "| Pair | Exact groups | Exact pairs | Normalized groups | Normalized pairs |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    pairs = sorted(set(exact_cross) | set(normalized_cross))
    for pair in pairs:
        exact = exact_cross.get(pair, {"groups": 0, "pairs": 0})
        normalized = normalized_cross.get(pair, {"groups": 0, "pairs": 0})
        lines.append(
            f"| {pair} | {exact['groups']} | {exact['pairs']} | "
            f"{normalized['groups']} | {normalized['pairs']} |"
        )

    if exact_cross:
        lines += ["", "### Exact-overlap examples", ""]
        for pair, entry in exact_cross.items():
            for example in entry["examples"]:
                lines.append(f"- {pair}: {example_text(example)}")

    if bio["train"]:
        lines += ["", "### BIO transition errors (first 5)", ""]
        for error in bio["train"][:5]:
            lines.append(
                f"- train[{error['index']}] index {error['position']}: "
                f"{error['tag']} after {error['previous']}"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- Normalization uses the project default (diacritics, tatweel, alef/ya).",
        "- All-O sequences are valid WikiANN examples but carry no entity signal.",
        "- BIO errors count `I-X` tags without a preceding `B-X`/`I-X`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path} and {json_path}")
    print(f"Leakage: exact={exact_leak}, normalized={normalized_leak}")

    if arguments.fail_on_leakage and exact_leak:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
