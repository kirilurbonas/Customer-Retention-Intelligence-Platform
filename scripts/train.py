from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from customer_retention_intelligence_platform.pipeline.data_generation import build_synthetic_customer_dataset
from customer_retention_intelligence_platform.pipeline.training import persist_artifacts, train_and_evaluate
from customer_retention_intelligence_platform.utils.config import settings
from customer_retention_intelligence_platform.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn model and persist artifacts.")
    parser.add_argument("--rows", type=int, default=6000, help="Number of synthetic rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible datasets.")
    parser.add_argument(
        "--raw-output",
        type=str,
        default="data/raw/customers.csv",
        help="Path to save generated training data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating dataset rows=%s seed=%s", args.rows, args.seed)
    dataset = build_synthetic_customer_dataset(n_rows=args.rows, seed=args.seed)
    dataset.to_csv(raw_path, index=False)

    logger.info("Training model")
    model, metrics = train_and_evaluate(dataset)
    persist_artifacts(
        model=model,
        metrics=metrics,
        model_path=settings.model_path,
        training_rows=args.rows,
        seed=args.seed,
    )

    logger.info("Saved training data to %s", raw_path)
    logger.info("Saved model to %s", settings.model_path)
    logger.info("Metrics: %s", metrics)
