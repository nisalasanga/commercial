"""Training pipeline for NPL prediction using master table features."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	average_precision_score,
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.master_table.one import build_and_save_master_table

MASTER_TABLE_PATH = PROJECT_ROOT / "artifacts" / "master_table.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "random_forest_npl.pkl"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "train_metrics.json"
FEATURES_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.json"

TARGET_COLUMN = "target_npl"
DROP_COLUMNS = {"seller_id", "date", TARGET_COLUMN}


def load_or_create_master_table(
	master_table_path: Path,
	source_csv_path: Path,
) -> pd.DataFrame:
	"""Load existing master table or build it from source CSV.

	Args:
		master_table_path: Existing master table artifact path.
		source_csv_path: Raw input CSV path to rebuild master table if needed.

	Returns:
		Seller-date level master table.
	"""
	if master_table_path.exists():
		return pd.read_csv(master_table_path)

	return build_and_save_master_table(source_csv_path, master_table_path)


def prepare_features(master_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""Prepare training matrix and target from master table.

	Args:
		master_table: Seller-date level feature table.

	Returns:
		Tuple of features dataframe and target series.
	"""
	if TARGET_COLUMN not in master_table.columns:
		raise ValueError(f"Required target column missing: {TARGET_COLUMN}")

	frame = master_table.copy()
	if "date" in frame.columns:
		frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

	feature_columns = [col for col in frame.columns if col not in DROP_COLUMNS]
	X = frame[feature_columns].fillna(0.0)
	y = frame[TARGET_COLUMN].astype(int)
	return X, y


def time_based_split(
	frame: pd.DataFrame,
	test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Create train/test split using a date cutoff.

	Args:
		frame: Master table containing date column.
		test_ratio: Fraction of latest dates reserved for test.

	Returns:
		Train and test dataframes.
	"""
	if "date" not in frame.columns:
		raise ValueError("Master table requires date column for time-based split.")

	data = frame.copy()
	data["date"] = pd.to_datetime(data["date"], errors="coerce")
	data = data.dropna(subset=["date"]).sort_values("date")

	unique_dates = np.array(sorted(data["date"].dropna().unique()))
	if len(unique_dates) < 2:
		raise ValueError("Insufficient date diversity for train/test split.")

	split_index = int(len(unique_dates) * (1 - test_ratio))
	split_index = min(max(split_index, 1), len(unique_dates) - 1)
	cutoff_date = unique_dates[split_index - 1]

	train_df = data[data["date"] <= cutoff_date].copy()
	test_df = data[data["date"] > cutoff_date].copy()

	if train_df.empty or test_df.empty:
		raise ValueError("Time split produced empty train/test set.")

	return train_df, test_df


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
	"""Train class-imbalance aware Random Forest model.

	Args:
		X_train: Train feature matrix.
		y_train: Train target vector.

	Returns:
		Trained RandomForestClassifier.
	"""
	model = RandomForestClassifier(
		n_estimators=400,
		max_depth=10,
		min_samples_leaf=20,
		random_state=42,
		n_jobs=-1,
		class_weight="balanced_subsample",
	)
	model.fit(X_train, y_train)
	return model


def evaluate_model(
	model: RandomForestClassifier,
	X: pd.DataFrame,
	y_true: pd.Series,
	label: str = "test",
) -> dict[str, float]:
	"""Evaluate classifier performance for a specific split.

	Args:
		model: Trained model.
		X: Feature matrix for the requested split.
		y_true: Target labels for that split.
		label: Split label used in metrics output.

	Returns:
		Metrics dictionary with the split name included.
	"""
	y_prob = model.predict_proba(X)[:, 1]
	y_pred = (y_prob >= 0.5).astype(int)
	true_negative, false_positive, false_negative, true_positive = confusion_matrix(
		y_true,
		y_pred,
		labels=[0, 1],
	).ravel()

	metrics = {
		"split": label,
		"roc_auc": float(roc_auc_score(y_true, y_prob)),
		"pr_auc": float(average_precision_score(y_true, y_prob)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"positive_rate": float(y_true.mean()),
		"confusion_matrix": {
			"true_negative": int(true_negative),
			"false_positive": int(false_positive),
			"false_negative": int(false_negative),
			"true_positive": int(true_positive),
		},
	}
	return metrics


def save_artifacts(
	model: RandomForestClassifier,
	metrics: dict[str, float] | dict[str, dict[str, float]],
	feature_columns: list[str],
) -> None:
	"""Save trained model and metadata artifacts."""
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

	with open(MODEL_PATH, "wb") as f:
		pickle.dump(model, f)

	with open(METRICS_PATH, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	with open(FEATURES_PATH, "w", encoding="utf-8") as f:
		json.dump(feature_columns, f, indent=2)


def main() -> None:
	"""Run model training pipeline."""
	source_csv = PROJECT_ROOT / "ecommerce_data.csv"
	master_df = load_or_create_master_table(MASTER_TABLE_PATH, source_csv)

	train_df, test_df = time_based_split(master_df, test_ratio=0.2)

	X_train, y_train = prepare_features(train_df)
	X_test, y_test = prepare_features(test_df)

	model = train_random_forest(X_train, y_train)
	train_metrics = evaluate_model(model, X_train, y_train, label="train")
	test_metrics = evaluate_model(model, X_test, y_test, label="test")
	metrics = {"train": train_metrics, "test": test_metrics}

	save_artifacts(model, metrics, X_train.columns.tolist())

	print("Training completed.")
	print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
	print(f"Model: {MODEL_PATH}")
	print(f"Metrics: {METRICS_PATH}")
	print("Train metrics:")
	print(json.dumps(train_metrics, indent=2))
	print("Test metrics:")
	print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
	main()
