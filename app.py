"""Simple fintech lending app for ecommerce seller default prediction.

This app covers three assignment parts:
1) Feature engineering + predictive modeling.
2) Deployment and MLOps architecture summary.
3) Agentic AI + RAG extension design notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import streamlit as st
except ImportError:
    st = None


@dataclass
class ModelArtifacts:
    """Container for model outputs used by the UI.

    Attributes:
        pipeline: Trained sklearn pipeline.
        metrics: Validation metrics.
        feature_importance: Logistic regression coefficients by feature.
        scored_sellers: Seller-level table with risk scores.
    """

    pipeline: Pipeline
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    scored_sellers: pd.DataFrame


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load raw ecommerce transactions.

    Args:
        csv_path: Path to input CSV.

    Returns:
        Raw dataframe with parsed dates.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _trend_slope(values: pd.Series) -> float:
    """Return slope of a sequence using simple linear fit."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = values.to_numpy(dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def build_seller_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create seller-level features from daily transaction history.

    Args:
        df: Daily transaction-level input data.

    Returns:
        Seller-level features with one row per seller.
    """
    working = df.copy()
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
    working["high_refund_day"] = (working["refund_ratio"] > 0.2).astype(int)

    base = (
        working.groupby("seller_id", as_index=False)
        .agg(
            total_gmv=("daily_gmv", "sum"),
            avg_daily_gmv=("daily_gmv", "mean"),
            std_daily_gmv=("daily_gmv", "std"),
            avg_refund_amount=("refund_amount", "mean"),
            avg_refund_ratio=("refund_ratio", "mean"),
            avg_fee_ratio=("fee_ratio", "mean"),
            promo_day_ratio=("promo_flag", "mean"),
            high_refund_day_ratio=("high_refund_day", "mean"),
            is_default=("is_default", "first"),
        )
        .fillna(0.0)
    )

    recent = (
        working.sort_values("date")
        .groupby("seller_id", as_index=False)
        .tail(30)
        .groupby("seller_id", as_index=False)
        .agg(recent_30d_avg_gmv=("daily_gmv", "mean"))
    )

    previous = (
        working.sort_values("date")
        .groupby("seller_id", as_index=False)
        .apply(lambda x: x.iloc[-60:-30])
        .reset_index(drop=True)
        .groupby("seller_id", as_index=False)
        .agg(prev_30d_avg_gmv=("daily_gmv", "mean"))
    )

    slope = (
        working.sort_values("date")
        .groupby("seller_id", as_index=False)
        .agg(gmv_trend_slope=("daily_gmv", _trend_slope))
    )

    features = base.merge(recent, on="seller_id", how="left")
    features = features.merge(previous, on="seller_id", how="left")
    features = features.merge(slope, on="seller_id", how="left")

    features["recent_gmv_change"] = np.where(
        features["prev_30d_avg_gmv"] > 0,
        (features["recent_30d_avg_gmv"] - features["prev_30d_avg_gmv"])
        / features["prev_30d_avg_gmv"],
        0.0,
    )

    return features.fillna(0.0)


def train_model(feature_df: pd.DataFrame) -> ModelArtifacts:
    """Train a default risk model and return artifacts for display.

    Args:
        feature_df: Seller-level feature table.

    Returns:
        ModelArtifacts object with model, metrics, and score table.
    """
    input_df = feature_df.copy()
    y = input_df.pop("is_default")
    sellers = input_df.pop("seller_id")
    X = input_df

    X_train, X_test, y_train, y_test, seller_train, seller_test = train_test_split(
        X,
        y,
        sellers,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    coef = pipeline.named_steps["classifier"].coef_[0]
    importance = pd.DataFrame(
        {"feature": X.columns, "coefficient": coef}
    ).sort_values("coefficient", ascending=False)

    scored = feature_df[["seller_id", "is_default"]].copy()
    scored["default_probability"] = pipeline.predict_proba(X)[:, 1]
    scored["predicted_label"] = (scored["default_probability"] >= 0.5).astype(int)

    # Mark test split membership for transparent evaluation in UI.
    scored["is_in_test_split"] = scored["seller_id"].isin(seller_test).astype(int)

    return ModelArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        feature_importance=importance,
        scored_sellers=scored.sort_values("default_probability", ascending=False),
    )


def render_modeling_tab(raw_df: pd.DataFrame) -> None:
    """Render Part 1 UI: features, model, and risk scores."""
    st.subheader("Part 1: Feature Engineering + Predictive Modeling")

    feature_df = build_seller_features(raw_df)
    artifacts = train_model(feature_df)

    metric_cols = st.columns(5)
    metric_cols[0].metric("ROC-AUC", f"{artifacts.metrics['roc_auc']:.3f}")
    metric_cols[1].metric("Accuracy", f"{artifacts.metrics['accuracy']:.3f}")
    metric_cols[2].metric("Precision", f"{artifacts.metrics['precision']:.3f}")
    metric_cols[3].metric("Recall", f"{artifacts.metrics['recall']:.3f}")
    metric_cols[4].metric("F1", f"{artifacts.metrics['f1']:.3f}")

    st.markdown("#### Top risk sellers")
    st.dataframe(artifacts.scored_sellers.head(15), use_container_width=True)

    st.markdown("#### Feature impact (logistic coefficients)")
    st.dataframe(artifacts.feature_importance, use_container_width=True)

    selected_seller = st.selectbox(
        "Inspect seller risk score",
        options=artifacts.scored_sellers["seller_id"].tolist(),
    )
    seller_row = artifacts.scored_sellers.loc[
        artifacts.scored_sellers["seller_id"] == selected_seller
    ].iloc[0]

    st.write(
        {
            "seller_id": selected_seller,
            "predicted_default_probability": round(
                float(seller_row["default_probability"]), 4
            ),
            "predicted_label": int(seller_row["predicted_label"]),
            "actual_label": int(seller_row["is_default"]),
        }
    )


def render_mlop_tab() -> None:
    """Render Part 2 summary for deployment and MLOps."""
    st.subheader("Part 2: Deployment + MLOps Architecture")
    st.markdown(
        """
1. Daily ingestion pipeline loads transaction data from partner APIs or storage.
2. Feature job computes seller-level aggregates and writes a feature table.
3. Batch scoring service predicts risk and publishes lending recommendations.
4. Analysts view scores in internal dashboard and can override with reason.
5. Monitoring tracks drift, model performance, and failed pipeline steps.
6. Retraining job runs monthly or when drift threshold is crossed.

Recommended stack (simple):
- Orchestration: Airflow or Azure Data Factory
- Storage: Data Lake + SQL warehouse
- Model registry: MLflow
- Serving: FastAPI batch endpoint
- Monitoring: Evidently + dashboard alerts
        """
    )


def render_rag_tab() -> None:
    """Render Part 3 summary for agentic AI + RAG extension."""
    st.subheader("Part 3: Agentic AI + RAG Extension")
    st.markdown(
        """
Goal: let credit analysts ask questions about uploaded PDF financial reports.

Basic design:
1. Analyst uploads PDF files.
2. Backend extracts text and splits into chunks.
3. Chunks are embedded and stored in a vector database.
4. Chat query retrieves top relevant chunks.
5. LLM answers with citations and confidence hints.
6. Final recommendation combines model score + report insights.

Guardrails:
- Show source citations for each answer.
- Restrict access by seller/case ID.
- Keep audit log for every prompt and answer.
- Add fallback: "insufficient evidence" when retrieval is weak.
        """
    )


def main() -> None:
    """Run Streamlit app."""
    if st is None:
        raise ImportError(
            "streamlit is required to run the UI. Install dependencies from requirements.txt"
        )

    st.set_page_config(page_title="FinTech Lending Decision App", layout="wide")
    st.title("FinTech Lending Decision App")
    st.caption("Simple and readable implementation for the assignment")

    default_path = Path("ecommerce_data.csv")
    csv_path = st.text_input("CSV path", value=str(default_path))

    if not Path(csv_path).exists():
        st.error("CSV path does not exist. Please provide a valid file path.")
        return

    raw_df = load_data(csv_path)
    st.write(
        f"Loaded {len(raw_df):,} rows for {raw_df['seller_id'].nunique():,} sellers."
    )

    with st.expander("Preview raw data"):
        st.dataframe(raw_df.head(20), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Modeling", "MLOps", "Agentic AI + RAG"])

    with tab1:
        render_modeling_tab(raw_df)

    with tab2:
        render_mlop_tab()

    with tab3:
        render_rag_tab()


if __name__ == "__main__":
    main()
