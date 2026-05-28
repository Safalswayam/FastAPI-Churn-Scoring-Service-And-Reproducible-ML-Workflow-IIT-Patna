"""API Tests — D2C Churn Scoring Service"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["MODEL_PATH"] = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.pkl")

import pytest
from fastapi.testclient import TestClient
from app.main import app

# Use context manager so lifespan (model load) fires
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

VALID_PAYLOAD = {
    "customer_id": "CUST00001",
    "recency_days": 45, "frequency_180d": 3, "monetary_180d": 1200.0,
    "return_rate_180d": 0.1, "avg_discount_pct_180d": 0.2, "avg_rating_180d": 4.0,
    "category_diversity_180d": 2, "ticket_count_90d": 1, "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 5.0, "days_since_signup": 300,
    "city_tier": "Tier 1", "age_group": "25-34", "acquisition_channel": "Instagram",
    "loyalty_tier": "Silver", "preferred_category": "Skin Care", "marketing_consent": "Yes",
    "sessions_30d": 8, "product_views_30d": 20, "cart_adds_30d": 3, "wishlist_adds_30d": 2,
    "abandoned_carts_30d": 1, "email_opens_30d": 5, "campaign_clicks_30d": 2,
    "last_visit_days_ago": 3,
}

HIGH_RISK_PAYLOAD = {
    **VALID_PAYLOAD, "customer_id": "CUST99999",
    "recency_days": 120, "frequency_180d": 1, "monetary_180d": 200.0,
    "ticket_count_90d": 4, "negative_ticket_rate_90d": 0.75,
    "sessions_30d": 0, "last_visit_days_ago": 30, "loyalty_tier": None,
}

# ── Health ────────────────────────────────────────────────
def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok"
    assert b["model_loaded"] is True
    assert b["version"] == "1.0.0"

# ── Valid predictions ─────────────────────────────────────
def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    b = r.json()
    assert b["customer_id"] == "CUST00001"
    assert 0.0 <= b["churn_probability"] <= 1.0
    assert b["predicted_class"] in [0, 1]
    assert b["risk_level"] in ["Very Low","Low","Medium","High","Very High"]
    assert b["confidence"] in ["Low","Medium","High"]
    assert isinstance(b["risk_explanation"], str) and len(b["risk_explanation"]) > 0
    assert b["threshold_used"] > 0

def test_predict_high_risk_customer(client):
    r = client.post("/predict", json=HIGH_RISK_PAYLOAD)
    assert r.status_code == 200
    b = r.json()
    assert b["churn_probability"] >= 0.5
    assert b["predicted_class"] == 1
    assert b["risk_level"] in ["High","Very High"]

def test_predict_retained_customer(client):
    low_risk = {**VALID_PAYLOAD, "customer_id": "CUST_SAFE",
                "recency_days": 5, "frequency_180d": 8, "monetary_180d": 4500.0,
                "sessions_30d": 20, "ticket_count_90d": 0,
                "last_visit_days_ago": 1, "loyalty_tier": "Gold"}
    r = client.post("/predict", json=low_risk)
    assert r.status_code == 200
    assert r.json()["churn_probability"] < 0.5

# ── Validation errors ─────────────────────────────────────
def test_predict_missing_required_field(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "recency_days"}
    assert client.post("/predict", json=bad).status_code == 422

def test_predict_invalid_city_tier(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "city_tier": "Metro"}).status_code == 422

def test_predict_invalid_age_group(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "age_group": "teen"}).status_code == 422

def test_predict_out_of_range_return_rate(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "return_rate_180d": 1.5}).status_code == 422

def test_predict_negative_recency(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "recency_days": -5}).status_code == 422

def test_predict_invalid_loyalty_tier(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "loyalty_tier": "Bronze"}).status_code == 422

def test_predict_null_loyalty_tier_accepted(client):
    r = client.post("/predict", json={**VALID_PAYLOAD, "loyalty_tier": None})
    assert r.status_code == 200

# ── Batch ─────────────────────────────────────────────────
def test_batch_predict_valid(client):
    r = client.post("/batch_predict", json={"customers": [VALID_PAYLOAD, HIGH_RISK_PAYLOAD]})
    assert r.status_code == 200
    b = r.json()
    assert b["total_customers"] == 2
    assert len(b["predictions"]) == 2
    assert "high_risk_count" in b
    assert b["processing_time_ms"] > 0
    for pred in b["predictions"]:
        assert 0.0 <= pred["churn_probability"] <= 1.0
        assert pred["predicted_class"] in [0, 1]

def test_batch_predict_single_customer(client):
    r = client.post("/batch_predict", json={"customers": [VALID_PAYLOAD]})
    assert r.status_code == 200
    assert r.json()["total_customers"] == 1

def test_batch_predict_empty_list(client):
    assert client.post("/batch_predict", json={"customers": []}).status_code == 422

def test_batch_high_risk_count_matches(client):
    r = client.post("/batch_predict", json={"customers": [VALID_PAYLOAD, HIGH_RISK_PAYLOAD]})
    b = r.json()
    computed = sum(1 for p in b["predictions"] if p["predicted_class"] == 1)
    assert b["high_risk_count"] == computed

# ── Schema integrity ──────────────────────────────────────
def test_predict_all_fields_present(client):
    b = client.post("/predict", json=VALID_PAYLOAD).json()
    for field in ["customer_id","churn_probability","predicted_class",
                  "risk_level","risk_explanation","confidence","threshold_used"]:
        assert field in b, f"Missing: {field}"

def test_predict_probability_is_float(client):
    b = client.post("/predict", json=VALID_PAYLOAD).json()
    assert isinstance(b["churn_probability"], float)
