# Customer Retention Intelligence Platform

Customer Retention Intelligence Platform is an end-to-end ML service that predicts customer churn risk for subscription businesses and exposes real-time and batch inference APIs.

## What is included
- Reproducible training pipeline (`scripts/train.py`) with persisted model + metrics metadata.
- FastAPI service with health and readiness probes.
- Single and batch prediction endpoints.
- Test suite and lint checks for CI.
- Dockerfile + Compose support.
- Committed test dataset under `data/test/` for deterministic local testing.

## Repository structure
```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
│       ├── customers_test.csv
│       └── batch_prediction_payload.json
├── models/
├── scripts/
│   └── train.py
├── src/customer_retention_intelligence_platform/
│   ├── api/
│   ├── pipeline/
│   └── utils/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

## Prerequisites
- Python 3.10+
- `pip`
- Optional: Docker 24+

## Setup
1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -e '.[dev]'
```

3. Create environment file:
```bash
cp .env.example .env
```

## Configuration
Environment variables:
- `APP_ENV`: `dev|staging|prod|test`
- `LOG_LEVEL`: `DEBUG|INFO|WARNING|ERROR`
- `MODEL_PATH`: model artifact path (default expected: `models/churn_model.joblib`)
- `THRESHOLD`: decision threshold for predicted churn (`0 < threshold < 1`)

## Quick start
1. Train a model:
```bash
make train
```

2. Start API:
```bash
make serve
```

3. Check service:
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready
```

4. Run tests:
```bash
make test
```

## Test dataset
Two committed fixtures are available:
- `data/test/customers_test.csv`: 20 labeled rows with full training schema (`churned` included).
- `data/test/batch_prediction_payload.json`: ready-to-send API payload (`items` only, no `churned`).

### Use the test payload with batch prediction
1. Start API (`make serve`)
2. Send fixture payload:
```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d @data/test/batch_prediction_payload.json
```

### Use the CSV fixture for manual checks
```bash
head -n 5 data/test/customers_test.csv
```

## Training
Default training (synthetic data generation + artifact persistence):
```bash
make train
```

Custom training data generation parameters:
```bash
python scripts/train.py --rows 10000 --seed 11 --raw-output data/raw/customers_seed11.csv
```

Artifacts written:
- Model: `models/churn_model.joblib`
- Metrics/metadata: `models/churn_model.metrics.json`

## API reference
Base URL: `http://localhost:8000`

### `GET /health`
Liveness and model visibility.

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
Readiness probe. Returns `503` if model artifact is missing.

### `POST /predict`
Single prediction.

Request example:
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

### `POST /predict/batch`
Batch prediction for up to 500 records.

Request shape:
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

## Developer commands
- `make install`: install editable package + dev dependencies.
- `make train`: train and persist artifacts.
- `make serve`: run API with Uvicorn on port `8000`.
- `make test`: run `pytest -q`.
- `make lint`: run Ruff lint checks.
- `make format`: run Ruff autofix.
- `make docker-build`: build container image.
- `make docker-run`: run container on `localhost:8000`.

## Docker
Build image:
```bash
make docker-build
```

Run container:
```bash
make docker-run
```

Or run with compose:
```bash
docker compose up --build
```

## Troubleshooting
- `503 Model artifact missing`: run `make train` first.
- Import errors in scripts: confirm virtual environment is active.
- Threshold behavior unexpected: verify `THRESHOLD` in `.env`.
- API starts but predictions fail: ensure `models/churn_model.joblib` and `models/churn_model.metrics.json` both exist.

## Production recommendations
- Replace synthetic generation with warehouse/feature-store ingestion.
- Add auth and rate limiting to prediction endpoints.
- Add monitoring for drift and retraining triggers.
- Track online/offline model performance over time.
