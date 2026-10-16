"""Pipeline script to generate the training master table."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.master_table.one import build_and_save_master_table


def main() -> None:
	"""Build and persist the seller-level training master table."""
	input_path = PROJECT_ROOT / "ecommerce_data.csv"
	output_path = PROJECT_ROOT / "artifacts" / "master_table.csv"

	master_df = build_and_save_master_table(input_path, output_path)
	print(f"Master table written to: {output_path}")
	print(f"Rows: {master_df.shape[0]:,}, Columns: {master_df.shape[1]}")


if __name__ == "__main__":
	main()
