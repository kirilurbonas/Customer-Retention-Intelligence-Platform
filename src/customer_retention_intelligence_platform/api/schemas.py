from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_spend: float = Field(..., ge=0, le=10000)
    support_tickets: int = Field(..., ge=0, le=100)
    payment_failures: int = Field(..., ge=0, le=20)
    used_mobile_app: int = Field(..., ge=0, le=1)
    has_family_plan: int = Field(..., ge=0, le=1)
    region: str = Field(..., examples=["north", "south", "east", "west"])


class ChurnPrediction(BaseModel):
    churn_probability: float
    predicted_churn: bool
    risk_segment: str
    decision_threshold: float


class BatchPredictionRequest(BaseModel):
    items: list[CustomerFeatures] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[ChurnPrediction]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_version: str | None = None
