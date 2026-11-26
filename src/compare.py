"""TEMPORARY PLACEHOLDER — OWNED BY MEMBER 1 (PM).

Minimal aggregator so `make compare` works before Epic 7 is implemented.
Reads only canonical metrics files and regenerates the README table.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/compare.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results import results_table, update_readme_table


def main() -> None:
    print(results_table())
    path = update_readme_table()
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
