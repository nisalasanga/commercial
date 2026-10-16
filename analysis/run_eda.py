"""Launcher script for running EDA from the analysis folder."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.eda import run_eda


def main() -> None:
    """Run EDA with default dataset and output locations."""
    data_path = PROJECT_ROOT / "ecommerce_data.csv"
    plots_dir = PROJECT_ROOT / "analysis" / "plots"
    run_eda(str(data_path), str(plots_dir))


if __name__ == "__main__":
    main()
