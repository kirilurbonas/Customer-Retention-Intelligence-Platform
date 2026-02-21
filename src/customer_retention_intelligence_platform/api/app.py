from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from customer_retention_intelligence_platform.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnPrediction,
    CustomerFeatures,
    HealthResponse,
)
from customer_retention_intelligence_platform.api.service import model_readiness, predict_churn, predict_churn_batch
from customer_retention_intelligence_platform.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    readiness = model_readiness()
    if readiness["model_loaded"]:
        logger.info("Model artifacts detected at startup")
    else:
        logger.warning("Model artifact missing at startup. Run training before predictions.")
    yield


app = FastAPI(
    title="Customer Retention Intelligence Platform API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    readiness = model_readiness()
    return HealthResponse(**readiness)


@app.get("/ready", response_model=HealthResponse)
def ready_check() -> HealthResponse:
    readiness = model_readiness()
    if not readiness["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model artifact missing. Service not ready.")
    return HealthResponse(**readiness)


@app.post("/predict", response_model=ChurnPrediction)
def predict(features: CustomerFeatures) -> ChurnPrediction:
    try:
        return predict_churn(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact missing. Train model first.") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        predictions = predict_churn_batch(payload.items)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact missing. Train model first.") from exc

    return BatchPredictionResponse(predictions=predictions, total=len(predictions))
