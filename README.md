# Part 4 — FastAPI Churn Scoring Service
## D2C Customer Churn Intelligence & Retention API

> **Tests: 17/17 passing** | **Endpoints: 3** | **Docker: included**

---

## Overview

Internal REST API that scores customers for 60-day churn probability using a trained XGBoost model. CRM tools call `/predict` with a customer's behavioral features and receive a churn probability, risk level, and plain-English explanation.

---

## Folder Structure

```
part4_api/
├── app/
│   └── main.py               # FastAPI application (3 endpoints)
├── artifacts/
│   ├── model.pkl             # Pre-trained XGBoost model
│   └── metrics.json          # Saved evaluation metrics
├── capstone_data/            # Raw CSVs used to train model (optional)
├── tests/
│   └── test_api.py           # 17 pytest test cases (all passing)
├── train_model.py            # Standalone training script (run if no model.pkl)
├── monitoring_plan.md        # Post-deployment monitoring & responsible-use guide
├── Dockerfile                # Container deployment
├── metrics.json              # Example runtime metrics
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option A — Use Pre-trained Model (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API (model.pkl already included)
uvicorn app.main:app --reload --port 8000
```

### Option B — Train Model from Scratch

```bash
pip install -r requirements.txt

# Train and save model.pkl from raw dataset CSVs
python train_model.py --data-dir /path/to/capstone_data

# Then start the API
uvicorn app.main:app --reload --port 8000
```

### Option C — Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Swagger UI: http://localhost:8000/docs

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check + model load status |
| `POST` | `/predict` | Score a single customer |
| `POST` | `/batch_predict` | Score up to 500 customers |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Sample Requests & Responses

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "artifacts/model.pkl",
  "version": "1.0.0"
}
```

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST00042",
    "recency_days": 95,
    "frequency_180d": 2,
    "monetary_180d": 480.0,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.3,
    "avg_rating_180d": 3.5,
    "category_diversity_180d": 1,
    "ticket_count_90d": 2,
    "negative_ticket_rate_90d": 0.5,
    "avg_resolution_hours_90d": 18.0,
    "days_since_signup": 450,
    "city_tier": "Tier 2",
    "age_group": "25-34",
    "acquisition_channel": "Instagram",
    "loyalty_tier": null,
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes",
    "sessions_30d": 0,
    "product_views_30d": 0,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 1,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 25
  }'
```

```json
{
  "customer_id": "CUST00042",
  "churn_probability": 0.8712,
  "predicted_class": 1,
  "risk_level": "Very High",
  "risk_explanation": "Key risk signals: no purchase in 95 days; zero web activity in last 30 days; 2 support tickets recently; high negative sentiment in tickets; not enrolled in loyalty programme; last visit 25 days ago.",
  "confidence": "High",
  "threshold_used": 0.27
}
```

### POST /batch_predict

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"customers": [<payload_1>, <payload_2>]}'
```

```json
{
  "predictions": [...],
  "total_customers": 2,
  "high_risk_count": 1,
  "processing_time_ms": 12.4
}
```

---

## Input Validation Rules

| Field | Type | Constraints |
|-------|------|-------------|
| `recency_days` | int | 0 – 1000 |
| `return_rate_180d` | float | 0.0 – 1.0 |
| `city_tier` | str | "Tier 1", "Tier 2", "Tier 3" |
| `age_group` | str | "18-24", "25-34", "35-44", "45+" |
| `loyalty_tier` | str or null | "Silver", "Gold", "Platinum", null |
| `marketing_consent` | str | "Yes" or "No" |

Invalid inputs return `HTTP 422 Unprocessable Entity` with field-level error details.

---

## Run Tests

```bash
cd part4_api
pytest tests/test_api.py -v
```

Expected: **17 passed** covering:
- Health endpoint
- Valid single prediction
- High-risk customer detection
- Low-risk customer detection
- 6 validation error cases (missing field, bad city_tier, bad age_group, out-of-range, negative recency, invalid loyalty_tier)
- Null loyalty_tier accepted
- Batch prediction structure
- Batch high_risk_count accuracy
- Response schema completeness
- Probability type check

---

## Responsible Use

See `monitoring_plan.md` for the full guide. In brief:

Use to **prioritise** CRM outreach lists by churn risk  
Use risk explanation to **brief** customer-facing agents  
Use batch scores for **weekly high-risk reporting**

Do NOT auto-apply discounts based on score alone  
Do NOT use for credit, hiring, or eligibility decisions  
Do NOT skip human review for Very High risk (≥0.80) customers

---

## Tech Stack

Python 3.10+ | FastAPI | Pydantic v2 | Uvicorn | XGBoost | scikit-learn | joblib | pytest | Docker
