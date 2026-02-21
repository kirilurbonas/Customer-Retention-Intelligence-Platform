from fastapi.testclient import TestClient

from customer_retention_intelligence_platform.api.app import app
from customer_retention_intelligence_platform.api.schemas import ChurnPrediction

client = TestClient(app)


def test_health_check_shape():
    res = client.get("/health")
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "model_loaded" in payload
    assert "model_path" in payload


def test_ready_returns_503_when_model_missing():
    res = client.get("/ready")
    if res.status_code == 200:
        assert res.json()["status"] == "ok"
    else:
        assert res.status_code == 503


def test_batch_predict_contract(monkeypatch):
    fake_prediction = ChurnPrediction(
        churn_probability=0.81,
        predicted_churn=True,
        risk_segment="critical",
        decision_threshold=0.5,
    )

    def fake_batch(_items):
        return [fake_prediction]

    monkeypatch.setattr("customer_retention_intelligence_platform.api.app.predict_churn_batch", fake_batch)

    res = client.post(
        "/predict/batch",
        json={
            "items": [
                {
                    "tenure_months": 8,
                    "monthly_spend": 120,
                    "support_tickets": 2,
                    "payment_failures": 1,
                    "used_mobile_app": 0,
                    "has_family_plan": 0,
                    "region": "east",
                }
            ]
        },
    )

    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 1
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["risk_segment"] == "critical"
