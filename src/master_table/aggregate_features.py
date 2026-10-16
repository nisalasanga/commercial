"""Aggregate feature set for seller-level master table."""

import numpy as np
import pandas as pd

WINDOWS = {
    "1d": 1,
    "7d": 7,
    "30d": 30,
    "90d": 90,
}


def _aggregate_for_window(
    data: pd.DataFrame,
    window_size: int,
    suffix: str,
) -> pd.DataFrame:
    """Create rolling aggregate features per seller-date row.

    Args:
        data: Input dataframe sorted by seller/date.
        window_size: Rolling lookback window size.
        suffix: Feature suffix to indicate the window.

    Returns:
        Feature dataframe with one row per seller-date.
    """
    working = data[["seller_id", "date"]].copy()

    rolling_specs = {
        "daily_gmv": ["sum", "mean", "median", "min", "max", "std"],
        "refund_amount": ["sum", "mean", "median", "std"],
        "platform_fee": ["sum", "mean", "median", "std"],
        "refund_ratio": ["mean", "median", "max"],
        "fee_ratio": ["mean", "median"],
    }

    for column, stats in rolling_specs.items():
        grouped = data.groupby("seller_id")[column]
        for stat_name in stats:
            feature_name = f"{column}_{stat_name}_{suffix}"
            working[feature_name] = grouped.transform(
                lambda s, w=window_size, stat=stat_name: s.rolling(
                    window=w,
                    min_periods=1,
                ).agg(stat)
            )

    working[f"obs_count_{suffix}"] = data.groupby("seller_id")["daily_gmv"].transform(
        lambda s, w=window_size: s.rolling(window=w, min_periods=1).count()
    )
    working[f"promo_day_ratio_{suffix}"] = data.groupby("seller_id")["promo_flag"].transform(
        lambda s, w=window_size: s.rolling(window=w, min_periods=1).mean()
    )

    return working.fillna(0.0)


def build_aggregate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build rolling aggregate features at seller-date granularity.

    Args:
        data: Daily transaction data with seller_id and numeric columns.

    Returns:
        Seller-date level aggregate feature table.
    """
    working = data.sort_values(["seller_id", "date"]).copy()

    working["refund_ratio"] = np.where(
        working["daily_gmv"] > 0,
        working["refund_amount"] / working["daily_gmv"],
        0.0,
    )
    working["fee_ratio"] = np.where(
        working["daily_gmv"] > 0,
        working["platform_fee"] / working["daily_gmv"],
        0.0,
    )

    features = working[["seller_id", "date"]].copy()
    for suffix, window_size in WINDOWS.items():
        block = _aggregate_for_window(working, window_size=window_size, suffix=suffix)
        keep_cols = [
            col for col in block.columns if col not in {"seller_id", "date"}
        ]
        features = features.merge(
            block[["seller_id", "date"] + keep_cols],
            on=["seller_id", "date"],
            how="left",
        )

    return features.fillna(0.0)
