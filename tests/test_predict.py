from __future__ import annotations

import numpy as np
import pandas as pd

from src.train_predict.predict import (
    build_latest_seller_features,
    predict_latest_sellers,
)


class FixedProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile([0.2, 0.8], (len(features), 1))


def make_seller_history(seller_id: str, number_of_days: int) -> list[dict]:
    return [
        {
            "seller_id": seller_id,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
            "daily_gmv": 100.0 + day,
            "refund_amount": 5.0,
            "platform_fee": 15.0,
            "promo_flag": day % 2,
        }
        for day in range(number_of_days)
    ]


def test_build_latest_seller_features_selects_latest_eligible_seller_rows():
    data = pd.DataFrame(
        make_seller_history("eligible", 100)
        + make_seller_history("too_short", 89)
    )

    latest = build_latest_seller_features(data)

    assert latest["seller_id"].tolist() == ["eligible"]
    assert latest.iloc[0]["date"] == pd.Timestamp("2024-04-09")
    assert latest.iloc[0]["history_day_index"] == 100


def test_predict_latest_sellers_scores_latest_rows():
    data = pd.DataFrame(make_seller_history("seller", 90))
    feature_columns = [
        "history_day_index",
        "daily_gmv_sum_1d",
        "daily_gmv_mean_7d",
    ]

    predictions = predict_latest_sellers(
        data,
        FixedProbabilityModel(),
        feature_columns,
    )

    assert predictions[["seller_id", "date"]].to_dict("records") == [
        {"seller_id": "seller", "date": pd.Timestamp("2024-03-30")}
    ]
    assert predictions.iloc[0]["npl_probability"] == 0.8
    assert predictions.iloc[0]["predicted_npl"] == 1
