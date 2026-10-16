"""Derived interaction feature set for seller-date master table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build derived interaction features at seller-date level.

    Args:
        data: Daily transaction data.

    Returns:
        Seller-date level derived feature table.
    """
    working = data.sort_values(["seller_id", "date"]).copy()

    working["net_gmv"] = (
        working["daily_gmv"] - working["refund_amount"] - working["platform_fee"]
    )
    working["refund_to_fee"] = np.where(
        working["platform_fee"] > 0,
        working["refund_amount"] / working["platform_fee"],
        0.0,
    )

    grouped_net = working.groupby("seller_id")["net_gmv"]
    grouped_rtf = working.groupby("seller_id")["refund_to_fee"]

    working["net_gmv_mean_30d"] = grouped_net.transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    working["net_gmv_mean_90d"] = grouped_net.transform(
        lambda s: s.rolling(window=90, min_periods=1).mean()
    )
    working["refund_to_fee_mean_30d"] = grouped_rtf.transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    working["refund_to_fee_mean_90d"] = grouped_rtf.transform(
        lambda s: s.rolling(window=90, min_periods=1).mean()
    )

    promo_gmv = working["daily_gmv"].where(working["promo_flag"] == 1)
    nonpromo_gmv = working["daily_gmv"].where(working["promo_flag"] == 0)
    working["promo_gmv_mean_90d"] = promo_gmv.groupby(working["seller_id"]).transform(
        lambda s: s.rolling(window=90, min_periods=1).mean()
    )
    working["nonpromo_gmv_mean_90d"] = nonpromo_gmv.groupby(
        working["seller_id"]
    ).transform(lambda s: s.rolling(window=90, min_periods=1).mean())
    working["promo_gmv_lift_90d"] = np.where(
        working["nonpromo_gmv_mean_90d"] > 0,
        (working["promo_gmv_mean_90d"] - working["nonpromo_gmv_mean_90d"])
        / working["nonpromo_gmv_mean_90d"],
        0.0,
    )

    feature_columns = [
        "seller_id",
        "date",
        "net_gmv",
        "refund_to_fee",
        "net_gmv_mean_30d",
        "net_gmv_mean_90d",
        "refund_to_fee_mean_30d",
        "refund_to_fee_mean_90d",
        "promo_gmv_mean_90d",
        "nonpromo_gmv_mean_90d",
        "promo_gmv_lift_90d",
    ]

    return working[feature_columns].fillna(0.0)
