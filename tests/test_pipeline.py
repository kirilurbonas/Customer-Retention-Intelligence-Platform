from pathlib import Path

import json

from customer_retention_intelligence_platform.pipeline.data_generation import build_synthetic_customer_dataset
from customer_retention_intelligence_platform.pipeline.training import persist_artifacts, train_and_evaluate


def test_training_pipeline_produces_metrics():
    df = build_synthetic_customer_dataset(n_rows=2000, seed=123)
    model, metrics = train_and_evaluate(df)

    assert model is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["recall_class_1"] >= 0.5


def test_persist_artifacts_writes_metadata(tmp_path: Path):
    df = build_synthetic_customer_dataset(n_rows=800, seed=7)
    model, metrics = train_and_evaluate(df)

    model_path = tmp_path / "churn_model.joblib"
    persist_artifacts(
        model=model,
        metrics=metrics,
        model_path=str(model_path),
        training_rows=800,
        seed=7,
    )

    assert model_path.exists()
    metrics_path = model_path.with_suffix(".metrics.json")
    assert metrics_path.exists()

    payload = json.loads(metrics_path.read_text())
    assert payload["metadata"]["artifact_version"] == "1"
    assert payload["metadata"]["training_rows"] == 800
    assert payload["metrics"]["roc_auc"] >= 0.0
