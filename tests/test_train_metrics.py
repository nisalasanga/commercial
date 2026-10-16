from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.train import evaluate_model


class FixedProbabilityModel:
	def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
		return np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])


def test_evaluate_model_includes_confusion_matrix_and_precision_recall():
	features = pd.DataFrame({"feature": [1, 2, 3, 4]})
	labels = pd.Series([0, 0, 1, 1])

	metrics = evaluate_model(FixedProbabilityModel(), features, labels)

	assert metrics["confusion_matrix"] == {
		"true_negative": 1,
		"false_positive": 1,
		"false_negative": 1,
		"true_positive": 1,
	}
	assert metrics["precision"] == 0.5
	assert metrics["recall"] == 0.5
