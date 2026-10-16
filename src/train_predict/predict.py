"""Batch prediction utilities for the seller NPL model."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.master_table.aggregate_features import build_aggregate_features
from src.master_table.derived_features import build_derived_features
from src.master_table.lag_features import build_lag_features
from src.master_table.one import MIN_HISTORY_DAYS

PREDICTION_INPUT_COLUMNS = [
    "seller_id",
    "date",
    "daily_gmv",
    "refund_amount",
    "platform_fee",
    "promo_flag",
]


def build_latest_seller_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the model features for the latest eligible row for each seller."""
    missing_columns = [
        column for column in PREDICTION_INPUT_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required prediction columns: {missing_columns}")
    if data.empty:
        raise ValueError("Prediction input data is empty.")

    working = data.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["seller_id", "date"]).sort_values(
        ["seller_id", "date"]
    )
    if working.empty:
        raise ValueError("Prediction input has no rows with a valid seller_id and date.")

    working["history_day_index"] = working.groupby("seller_id").cumcount() + 1
    feature_blocks = [
        build_aggregate_features(working),
        build_lag_features(working),
        build_derived_features(working),
    ]

    features = working[["seller_id", "date", "history_day_index"]].copy()
    for block in feature_blocks:
        features = features.merge(block, on=["seller_id", "date"], how="left")

    latest = (
        features.sort_values(["seller_id", "date"])
        .drop_duplicates("seller_id", keep="last")
    )
    latest = latest[latest["history_day_index"] >= MIN_HISTORY_DAYS].copy()
    if latest.empty:
        raise ValueError(
            f"No sellers have the minimum {MIN_HISTORY_DAYS}-day history required "
            "by the trained model."
        )

    return latest.fillna(0.0)


def load_prediction_artifacts(
    model_path: str | Path,
    feature_columns_path: str | Path,
) -> tuple[RandomForestClassifier, list[str]]:
    """Load the trained classifier and its expected input feature names."""
    with Path(model_path).open("rb") as model_file:
        model = pickle.load(model_file)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(f"Expected a RandomForestClassifier in {model_path}.")

    with Path(feature_columns_path).open(encoding="utf-8") as columns_file:
        feature_columns = json.load(columns_file)
    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        raise ValueError(f"Invalid feature-column metadata in {feature_columns_path}.")
    if not feature_columns:
        raise ValueError(f"Feature-column metadata is empty: {feature_columns_path}.")
    if {"seller_id", "date"}.intersection(feature_columns):
        raise ValueError("Feature metadata must not include seller_id or date.")

    return model, feature_columns


def predict_latest_sellers(
    data: pd.DataFrame,
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Score each seller's latest feature row with the trained NPL classifier."""
    latest = build_latest_seller_features(data)
    missing_features = [
        column for column in feature_columns if column not in latest.columns
    ]
    if missing_features:
        raise ValueError(
            f"Generated prediction features do not match the model; "
            f"missing columns: {missing_features}"
        )

    X = latest.loc[:, feature_columns].fillna(0.0)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The loaded model has no positive (class 1) probability.")
    positive_class_index = classes.index(1)
    probabilities = model.predict_proba(X)[:, positive_class_index]

    predictions = latest[["seller_id", "date"]].copy()
    predictions["npl_probability"] = probabilities
    predictions["predicted_npl"] = (probabilities >= 0.5).astype(int)
    return predictions.sort_values("npl_probability", ascending=False).reset_index(
        drop=True
    )


def run_prediction(
    input_path: str | Path,
    model_path: str | Path,
    feature_columns_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Load input and model artifacts, score sellers, and save the result."""
    data = pd.read_csv(input_path)
    model, feature_columns = load_prediction_artifacts(
        model_path,
        feature_columns_path,
    )
    predictions = predict_latest_sellers(data, model, feature_columns)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)
    return predictions
