from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_path: str = os.getenv("MODEL_PATH", "models/churn_model.joblib")
    threshold: float = float(os.getenv("THRESHOLD", "0.50"))

    @property
    def metrics_path(self) -> str:
        return str(Path(self.model_path).with_suffix(".metrics.json"))

    def validate(self) -> None:
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("THRESHOLD must be between 0 and 1.")
        if self.app_env not in {"dev", "staging", "prod", "test"}:
            raise ValueError("APP_ENV must be one of: dev, staging, prod, test.")


settings = Settings()
settings.validate()
