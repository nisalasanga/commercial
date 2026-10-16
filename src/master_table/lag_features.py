"""Lag-based feature set for seller-date master table."""

from __future__ import annotations

import pandas as pd

LAG_NUMERIC_COLUMNS = ["daily_gmv", "refund_amount", "platform_fee"]


def build_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build lag and lag-delta features at seller-date level.

    Args:
        data: Daily transaction data sorted by seller/date.

    Returns:
        Seller-date level lag feature table.
    """
    working = data.sort_values(["seller_id", "date"]).copy()

    for col in LAG_NUMERIC_COLUMNS:
        working[f"{col}_lag_1"] = working.groupby("seller_id")[col].shift(1)
        working[f"{col}_lag_7"] = working.groupby("seller_id")[col].shift(7)
        working[f"{col}_lag_30"] = working.groupby("seller_id")[col].shift(30)
        working[f"{col}_lag_90"] = working.groupby("seller_id")[col].shift(90)
        working[f"{col}_delta_1"] = working[col] - working[f"{col}_lag_1"]
        working[f"{col}_delta_7"] = working[col] - working[f"{col}_lag_7"]

    keep_columns = ["seller_id", "date"]
    keep_columns += [
        col
        for col in working.columns
        if any(
            token in col
            for token in ["_lag_", "_delta_"]
        )
    ]

    return working[keep_columns].fillna(0.0)
