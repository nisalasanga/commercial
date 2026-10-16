"""Single entry point to build the training master table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.master_table.aggregate_features import build_aggregate_features
from src.master_table.derived_features import build_derived_features
from src.master_table.lag_features import build_lag_features
from src.utils import load_data

MIN_HISTORY_DAYS = 90
FEATURE_WINDOW_DAYS = 90
TARGET_HORIZON_DAYS = 90

REQUIRED_COLUMNS = [
    "seller_id",
    "date",
    "daily_gmv",
    "refund_amount",
    "platform_fee",
    "promo_flag",
    "is_default",
]


def _prepare_input(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw input data."""
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = data.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["seller_id", "date"]).sort_values(
        ["seller_id", "date"]
    )

    return working


def _add_history_index(data: pd.DataFrame) -> pd.DataFrame:
    """Add per-seller history day index."""
    working = data.copy()
    working["history_day_index"] = working.groupby("seller_id").cumcount() + 1
    return working


def _build_target_table(data: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Build forward-looking NPL target for each seller-date row.

    The target is 1 if any default is observed in the next `horizon_days` rows
    for the same seller, excluding the current row.
    """

    def _future_max(series: pd.Series) -> pd.Series:
        shifted = series.shift(-1)
        # Reverse rolling computes forward window in original order.
        return (
            shifted.iloc[::-1]
            .rolling(window=horizon_days, min_periods=1)
            .max()
            .iloc[::-1]
        )

    working = data.sort_values(["seller_id", "date"]).copy()
    working["target_npl"] = (
        working.groupby("seller_id")["is_default"].transform(_future_max).fillna(0.0)
    )

    seller_counts = working.groupby("seller_id")["seller_id"].transform("size")
    working["future_rows_available"] = seller_counts - working["history_day_index"]

    return working[["seller_id", "date", "target_npl", "future_rows_available"]]


def _eligible_sellers(data: pd.DataFrame, min_history_days: int) -> pd.Series:
    """Return seller IDs that satisfy the minimum history threshold."""
    history_counts = data.groupby("seller_id").size()
    eligible = history_counts[history_counts >= min_history_days].index
    return pd.Series(eligible)


def create_master_table(data: pd.DataFrame) -> pd.DataFrame:
    """Create seller-date master table for NPL/default model training.

    Args:
        data: Raw daily ecommerce transaction dataframe.

    Returns:
        Seller-date level master table with engineered features and target label.
    """
    base = _prepare_input(data)
    base = _add_history_index(base)

    eligible_ids = _eligible_sellers(base, min_history_days=MIN_HISTORY_DAYS)
    if eligible_ids.empty:
        return pd.DataFrame(columns=["seller_id", "date", "target_npl"])

    eligible_data = base[base["seller_id"].isin(eligible_ids)].copy()

    labels = _build_target_table(eligible_data, horizon_days=TARGET_HORIZON_DAYS)
    feature_blocks = [
        build_aggregate_features(eligible_data),
        build_lag_features(eligible_data),
        build_derived_features(eligible_data),
    ]

    master = eligible_data[["seller_id", "date", "history_day_index"]].copy()
    master = master.merge(labels, on=["seller_id", "date"], how="left")

    for block in feature_blocks:
        master = master.merge(block, on=["seller_id", "date"], how="left")

    # Apply row-level threshold and ensure target horizon is available.
    master = master[master["history_day_index"] >= FEATURE_WINDOW_DAYS]
    master = master[master["future_rows_available"] >= TARGET_HORIZON_DAYS]

    # Keep a clean training table.
    master = master.drop(columns=["future_rows_available"])

    master = master.fillna(0.0)
    master["target_npl"] = master["target_npl"].astype(int)
    return master


def create_master_table_from_csv(input_csv_path: str | Path) -> pd.DataFrame:
    """Load CSV and return engineered master table."""
    raw = load_data(str(input_csv_path))
    return create_master_table(raw)


def build_and_save_master_table(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
) -> pd.DataFrame:
    """Build master table from CSV and save it to disk."""
    master = create_master_table_from_csv(input_csv_path)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    return master


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "ecommerce_data.csv"
    output_path = project_root / "artifacts" / "master_table.csv"
    master_df = build_and_save_master_table(input_path, output_path)
    print(f"Master table saved: {output_path}")
    print(f"Shape: {master_df.shape}")
