from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from customer_retention_intelligence_platform.api.schemas import ChurnPrediction, CustomerFeatures
from customer_retention_intelligence_platform.utils.config import settings
from customer_retention_intelligence_platform.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    logger.info("Loading model from %s", settings.model_path)
    return joblib.load(settings.model_path)


@lru_cache(maxsize=1)
def load_metrics_payload() -> dict[str, Any]:
    metrics_path = Path(settings.metrics_path)
    if not metrics_path.exists():
        return {}

    try:
        return json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Metrics file is invalid JSON: %s", metrics_path)
        return {}


def classify_segment(probability: float) -> str:
    if probability >= 0.75:
        return "critical"
    if probability >= 0.5:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"


def model_readiness() -> dict[str, Any]:
    model_exists = Path(settings.model_path).exists()
    payload = load_metrics_payload()
    model_version = payload.get("metadata", {}).get("artifact_version")
    return {
        "status": "ok" if model_exists else "degraded",
        "model_loaded": model_exists,
        "model_path": settings.model_path,
        "model_version": model_version,
    }


def _predict_one(model: Pipeline, features: CustomerFeatures) -> ChurnPrediction:
    payload = pd.DataFrame([features.model_dump()])
    churn_probability = float(model.predict_proba(payload)[0, 1])
    predicted_churn = churn_probability >= settings.threshold

    return ChurnPrediction(
        churn_probability=round(churn_probability, 4),
        predicted_churn=predicted_churn,
        risk_segment=classify_segment(churn_probability),
        decision_threshold=settings.threshold,
    )


def predict_churn(features: CustomerFeatures) -> ChurnPrediction:
    model = load_model()
    return _predict_one(model, features)


def predict_churn_batch(features_list: list[CustomerFeatures]) -> list[ChurnPrediction]:
    model = load_model()
    return [_predict_one(model, item) for item in features_list]
