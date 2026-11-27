"""Aggregate canonical metrics and regenerate the README table (issue 7.1).

Reads only validated `results/metrics/<model>.json` files, so incomplete or
malformed artifacts fail loudly instead of producing a misleading table.
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
