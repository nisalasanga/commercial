"""Lead-based feature set for seller-date master table."""

from __future__ import annotations

import pandas as pd

LEAD_NUMERIC_COLUMNS = ["daily_gmv", "refund_amount", "platform_fee"]


def build_lead_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build lead and lead-delta features at seller-date level.

    Note:
        Lead features use future rows inside each seller history. For strict
        production training, ensure the cutoff logic prevents leakage.

    Args:
        data: Daily transaction data sorted by seller/date.

    Returns:
        Seller-date level lead feature table.
    """
    working = data.sort_values(["seller_id", "date"]).copy()

    for col in LEAD_NUMERIC_COLUMNS:
        working[f"{col}_lead_1"] = working.groupby("seller_id")[col].shift(-1)
        working[f"{col}_lead_7"] = working.groupby("seller_id")[col].shift(-7)
        working[f"{col}_lead_30"] = working.groupby("seller_id")[col].shift(-30)
        working[f"{col}_lead_90"] = working.groupby("seller_id")[col].shift(-90)
        working[f"{col}_lead_delta_1"] = working[f"{col}_lead_1"] - working[col]
        working[f"{col}_lead_delta_7"] = working[f"{col}_lead_7"] - working[col]

    keep_columns = ["seller_id", "date"]
    keep_columns += [col for col in working.columns if "_lead_" in col]

    return working[keep_columns].fillna(0.0)
