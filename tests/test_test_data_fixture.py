from pathlib import Path

import pandas as pd


def test_customers_test_dataset_has_expected_schema():
    dataset_path = Path("data/test/customers_test.csv")
    assert dataset_path.exists()

    df = pd.read_csv(dataset_path)
    assert len(df) >= 10
    assert list(df.columns) == [
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "payment_failures",
        "used_mobile_app",
        "has_family_plan",
        "region",
        "churned",
    ]

    assert set(df["churned"].unique()).issubset({0, 1})
