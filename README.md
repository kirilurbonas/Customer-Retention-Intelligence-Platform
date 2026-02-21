# Customer Retention Intelligence Platform

Customer Retention Intelligence Platform is an end-to-end ML service that predicts customer churn risk for subscription businesses and exposes real-time/batch inference APIs.

## Overview
The project includes:
- Reproducible training pipeline (`scripts/train.py`) with persisted model + metadata.
- REST API (`FastAPI`) with health/readiness endpoints.
- Single and batch prediction endpoints.
- CI checks for linting and tests.
- Docker setup for containerized deployment.

## Use cases
- Trigger retention workflows when users enter `high`/`critical` risk segments.
- Score customer lists in batch for weekly campaign planning.
- Integrate real-time scoring into billing, CRM, or support systems.

## Architecture
1. Data generation/training pipeline builds a scikit-learn model.
2. Artifacts are persisted to `models/`:
   - `churn_model.joblib`
   - `churn_model.metrics.json` (includes model metadata)
3. API loads artifacts at runtime and serves predictions.
4. `/health` and `/ready` provide operational status for orchestration.

## Repository structure
```text
.
├── .github/workflows/ci.yml
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── scripts/
│   └── train.py
├── src/customer_retention_intelligence_platform/
│   ├── api/
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── service.py
│   ├── pipeline/
│   │   ├── data_generation.py
│   │   └── training.py
│   └── utils/
│       ├── config.py
│       └── logging.py
└── tests/
```

## Prerequisites
- Python 3.10+
- `pip`
- Optional: Docker 24+

## Configuration
Copy and edit environment settings:

```bash
cp .env.example .env
```

Available variables:
- `APP_ENV`: `dev|staging|prod|test`
- `LOG_LEVEL`: `DEBUG|INFO|WARNING|ERROR`
- `MODEL_PATH`: model artifact path
- `THRESHOLD`: decision threshold for predicted churn (`0 < threshold < 1`)

## Local development
1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

2. Train the model:

```bash
make train
```

Optional training flags:

```bash
python scripts/train.py --rows 10000 --seed 11 --raw-output data/raw/customers.csv
```

3. Run API:

```bash
make serve
```

4. Run checks:

```bash
make lint
make test
```

## API reference
Base URL: `http://localhost:8000`

### `GET /health`
Liveness + model artifact visibility.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/churn_model.joblib",
  "model_version": "1"
}
```

### `GET /ready`
Readiness probe for orchestration. Returns `503` if model artifact is missing.

### `POST /predict`
Single prediction.

Request:

```json
{
  "tenure_months": 5,
  "monthly_spend": 130,
  "support_tickets": 4,
  "payment_failures": 2,
  "used_mobile_app": 0,
  "has_family_plan": 0,
  "region": "east"
}
```

Response:

```json
{
  "churn_probability": 0.8271,
  "predicted_churn": true,
  "risk_segment": "critical",
  "decision_threshold": 0.5
}
```

### `POST /predict/batch`
Batch prediction for up to 500 records per request.

Request:

```json
{
  "items": [
    {
      "tenure_months": 12,
      "monthly_spend": 70,
      "support_tickets": 1,
      "payment_failures": 0,
      "used_mobile_app": 1,
      "has_family_plan": 1,
      "region": "north"
    }
  ]
}
```

## Docker deployment
Build image:

```bash
make docker-build
```

Run container:

```bash
make docker-run
```

Or with compose:

```bash
docker compose up --build
```

## Production recommendations
- Replace synthetic dataset with warehouse ingestion.
- Pin dependencies with lock files and enable vulnerability scanning.
- Add authentication/rate-limiting in front of `/predict` endpoints.
- Add model monitoring (drift, performance decay) and scheduled retraining.
- Export metrics/logs to your observability stack.

## Troubleshooting
- `503 Model artifact missing`: run `make train` first.
- Import errors when running scripts: ensure virtual environment is active.
- Unexpected threshold behavior: verify `THRESHOLD` in `.env`.
