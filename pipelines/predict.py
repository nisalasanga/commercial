"""Run batch NPL predictions for the latest available day per seller."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_predict.predict import run_prediction

MODEL_PATH = PROJECT_ROOT / "artifacts" / "random_forest_npl.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.json"
PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "predictions.csv"


def main() -> None:
    """Load the trained model and write latest-seller predictions."""
    input_path = PROJECT_ROOT / "ecommerce_data.csv"
    predictions = run_prediction(
        input_path=input_path,
        model_path=MODEL_PATH,
        feature_columns_path=FEATURE_COLUMNS_PATH,
        output_path=PREDICTIONS_PATH,
    )

    print(f"Predictions written to: {PREDICTIONS_PATH}")
    print(f"Scored sellers: {len(predictions):,}")
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()
