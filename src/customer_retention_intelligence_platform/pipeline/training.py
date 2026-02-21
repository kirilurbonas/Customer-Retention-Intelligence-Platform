from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "payment_failures",
    "used_mobile_app",
    "has_family_plan",
    "region",
]
TARGET = "churned"



def build_training_pipeline() -> Pipeline:
    numeric_features = [
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "payment_failures",
        "used_mobile_app",
        "has_family_plan",
    ]
    categorical_features = ["region"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )

    pipeline = build_training_pipeline()
    pipeline.fit(x_train, y_train)

    probs = pipeline.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "roc_auc": round(float(auc), 4),
        "precision_class_1": round(float(report["1"]["precision"]), 4),
        "recall_class_1": round(float(report["1"]["recall"]), 4),
        "f1_class_1": round(float(report["1"]["f1-score"]), 4),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
    }
    return pipeline, metrics


def persist_artifacts(
    model: Pipeline,
    metrics: dict[str, Any],
    model_path: str,
    training_rows: int,
    seed: int,
) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    payload = {
        "metadata": {
            "artifact_version": "1",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "features": FEATURES,
            "target": TARGET,
            "training_rows": training_rows,
            "seed": seed,
            "model_class": "LogisticRegression",
        },
        "metrics": metrics,
    }

    metrics_path = path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(payload, indent=2))
