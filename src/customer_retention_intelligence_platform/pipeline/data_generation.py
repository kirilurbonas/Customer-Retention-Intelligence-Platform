from __future__ import annotations

import numpy as np
import pandas as pd


def build_synthetic_customer_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, n_rows)
    monthly_spend = rng.normal(75, 25, n_rows).clip(15, 220)
    support_tickets = rng.poisson(1.8, n_rows).clip(0, 12)
    payment_failures = rng.binomial(4, 0.12, n_rows)
    used_mobile_app = rng.binomial(1, 0.72, n_rows)
    has_family_plan = rng.binomial(1, 0.35, n_rows)
    region = rng.choice(["north", "south", "east", "west"], n_rows, p=[0.26, 0.24, 0.25, 0.25])

    churn_signal = (
        -0.045 * tenure_months
        + 0.018 * monthly_spend
        + 0.31 * support_tickets
        + 0.62 * payment_failures
        - 0.85 * used_mobile_app
        - 0.55 * has_family_plan
        + rng.normal(0, 0.8, n_rows)
    )

    region_bias = {"north": -0.1, "south": 0.1, "east": 0.18, "west": -0.07}
    region_term = np.vectorize(region_bias.get)(region)

    logits = churn_signal + region_term
    churn_probability = 1.0 / (1.0 + np.exp(-logits))
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend.round(2),
            "support_tickets": support_tickets,
            "payment_failures": payment_failures,
            "used_mobile_app": used_mobile_app,
            "has_family_plan": has_family_plan,
            "region": region,
            "churned": churned,
        }
    )
