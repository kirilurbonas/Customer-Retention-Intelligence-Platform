.PHONY: install train serve test lint format docker-build docker-run

install:
	python -m pip install -e '.[dev]'

train:
	python scripts/train.py

serve:
	uvicorn customer_retention_intelligence_platform.api.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .

format:
	ruff check . --fix

docker-build:
	docker build -t customer-retention-intelligence-platform:latest .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env customer-retention-intelligence-platform:latest
