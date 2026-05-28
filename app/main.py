"""
Part 4 — FastAPI Churn Scoring Service
D2C Customer Churn Intelligence & Retention API
"""

import os
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "artifacts/metrics.json")
_artifacts = None
_metrics_cache = None

def load_model():
    global _artifacts
    if _artifacts is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        _artifacts = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    return _artifacts


def load_metrics() -> Optional[Dict[str, Any]]:
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache
    try:
        with open(METRICS_PATH, "r") as f:
            _metrics_cache = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Metrics file not found at {METRICS_PATH}")
        _metrics_cache = None
    except Exception as e:
        logger.warning(f"Failed to load metrics: {e}")
        _metrics_cache = None
    return _metrics_cache

# ── Pydantic schemas ──────────────────────────────────────

class CustomerFeatures(BaseModel):
    customer_id: str

    recency_days:              int   = Field(..., ge=0, le=1000)
    frequency_180d:            int   = Field(..., ge=0, le=200)
    monetary_180d:             float = Field(..., ge=0, le=200000)
    return_rate_180d:          float = Field(..., ge=0.0, le=1.0)
    avg_discount_pct_180d:     float = Field(..., ge=0.0, le=1.0)
    avg_rating_180d:           Optional[float] = Field(None, ge=1.0, le=5.0)
    category_diversity_180d:   int   = Field(..., ge=0, le=10)
    ticket_count_90d:          int   = Field(..., ge=0, le=50)
    negative_ticket_rate_90d:  float = Field(..., ge=0.0, le=1.0)
    avg_resolution_hours_90d:  float = Field(..., ge=0.0, le=500.0)
    days_since_signup:         int   = Field(..., ge=0, le=3000)
    city_tier:                 str
    age_group:                 str
    acquisition_channel:       str
    loyalty_tier:              Optional[str] = None
    preferred_category:        str
    marketing_consent:         str
    sessions_30d:              int   = Field(..., ge=0, le=500)
    product_views_30d:         int   = Field(..., ge=0, le=2000)
    cart_adds_30d:             int   = Field(..., ge=0, le=500)
    wishlist_adds_30d:         int   = Field(..., ge=0, le=500)
    abandoned_carts_30d:       int   = Field(..., ge=0, le=200)
    email_opens_30d:           int   = Field(..., ge=0, le=500)
    campaign_clicks_30d:       int   = Field(..., ge=0, le=200)
    last_visit_days_ago:       int   = Field(..., ge=0, le=365)

    @field_validator("city_tier")
    @classmethod
    def val_city(cls, v):
        if v not in {"Tier 1", "Tier 2", "Tier 3"}:
            raise ValueError("city_tier must be 'Tier 1', 'Tier 2', or 'Tier 3'")
        return v

    @field_validator("age_group")
    @classmethod
    def val_age(cls, v):
        if v not in {"18-24", "25-34", "35-44", "45+"}:
            raise ValueError("age_group must be one of: 18-24, 25-34, 35-44, 45+")
        return v

    @field_validator("marketing_consent")
    @classmethod
    def val_consent(cls, v):
        if v not in {"Yes", "No"}:
            raise ValueError("marketing_consent must be 'Yes' or 'No'")
        return v

    @field_validator("loyalty_tier")
    @classmethod
    def val_loyalty(cls, v):
        if v is not None and v not in {"Silver", "Gold", "Platinum"}:
            raise ValueError("loyalty_tier must be Silver, Gold, Platinum, or null")
        return v


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str
    confidence: str
    threshold_used: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_path: str
    version: str
    artifacts_loaded: List[str]
    metrics: Optional[Dict[str, Any]] = None


# ── Inference ─────────────────────────────────────────────

def build_feature_row(d: dict) -> dict:
    """Build a flat feature dict matching the model's expected columns exactly."""
    recency_days = d["recency_days"]
    sessions     = d["sessions_30d"]

    recency_bucket = (
        "≤7d"   if recency_days <= 7  else
        "8-30d" if recency_days <= 30 else
        "31-60d" if recency_days <= 60 else
        "61-90d" if recency_days <= 90 else ">90d"
    )

    return {
        # Base numerical (from rfm_modeling_snapshot)
        "recency_days":              recency_days,
        "frequency_180d":            d["frequency_180d"],
        "monetary_180d":             d["monetary_180d"],
        "return_rate_180d":          d["return_rate_180d"],
        "avg_discount_pct_180d":     d["avg_discount_pct_180d"],
        "avg_rating_180d":           d["avg_rating_180d"] if d["avg_rating_180d"] is not None else 0.0,
        "category_diversity_180d":   d["category_diversity_180d"],
        "ticket_count_90d":          d["ticket_count_90d"],
        "negative_ticket_rate_90d":  d["negative_ticket_rate_90d"],
        "avg_resolution_hours_90d":  d["avg_resolution_hours_90d"],
        "days_since_signup":         d["days_since_signup"],
        "sessions_30d":              sessions,
        "product_views_30d":         d["product_views_30d"],
        "cart_adds_30d":             d["cart_adds_30d"],
        "wishlist_adds_30d":         d["wishlist_adds_30d"],
        "abandoned_carts_30d":       d["abandoned_carts_30d"],
        "email_opens_30d":           d["email_opens_30d"],
        "campaign_clicks_30d":       d["campaign_clicks_30d"],
        "last_visit_days_ago":       d["last_visit_days_ago"],
        # Derived numerical
        "engagement_score": (
            sessions * 2 +
            d["product_views_30d"] * 0.5 +
            d["email_opens_30d"] +
            d["campaign_clicks_30d"] * 1.5
        ),
        "cart_abandon_rate": (
            d["abandoned_carts_30d"] / max(sessions, 1) if sessions > 0 else 0.0
        ),
        "complaint_intensity": d["ticket_count_90d"] * (1 + d["negative_ticket_rate_90d"]),
        "has_loyalty": 1 if d["loyalty_tier"] else 0,
        # Categorical
        "city_tier":           d["city_tier"],
        "age_group":           d["age_group"],
        "acquisition_channel": d["acquisition_channel"],
        "loyalty_tier":        d["loyalty_tier"] if d["loyalty_tier"] else "Unknown",
        "preferred_category":  d["preferred_category"],
        "marketing_consent":   d["marketing_consent"],
        "recency_bucket":      recency_bucket,
    }


def make_prediction(features: CustomerFeatures, artifacts: dict) -> PredictionResponse:
    preprocessor = artifacts["preprocessor"]
    model        = artifacts["xgb_model"]
    threshold    = artifacts["threshold"]
    ALL_NUM      = artifacts["feature_names"]["num"]
    ALL_CAT      = artifacts["feature_names"]["cat"]

    row = build_feature_row(features.model_dump())
    df  = pd.DataFrame([row])

    # Ensure correct column order and types
    df[ALL_NUM] = df[ALL_NUM].fillna(0).astype(float)
    df[ALL_CAT] = df[ALL_CAT].fillna("Unknown").astype(str)

    X    = df[ALL_NUM + ALL_CAT]
    X_pp = preprocessor.transform(X)
    proba = float(model.predict_proba(X_pp)[0, 1])
    pred  = int(proba >= threshold)

    risk = (
        "Very High" if proba >= 0.80 else
        "High"      if proba >= 0.60 else
        "Medium"    if proba >= 0.40 else
        "Low"       if proba >= 0.20 else
        "Very Low"
    )
    distance   = abs(proba - threshold)
    confidence = "High" if distance >= 0.20 else "Medium" if distance >= 0.10 else "Low"

    d = features.model_dump()
    reasons = []
    if d["recency_days"] > 60:
        reasons.append(f"no purchase in {d['recency_days']} days")
    if d["sessions_30d"] == 0:
        reasons.append("zero web activity in last 30 days")
    if d["ticket_count_90d"] >= 2:
        reasons.append(f"{d['ticket_count_90d']} support tickets recently")
    if d["negative_ticket_rate_90d"] >= 0.5:
        reasons.append("high negative sentiment in tickets")
    if not d["loyalty_tier"]:
        reasons.append("not enrolled in loyalty programme")
    if d["last_visit_days_ago"] > 14:
        reasons.append(f"last visit {d['last_visit_days_ago']} days ago")

    explanation = (
        "Key risk signals: " + "; ".join(reasons) + "."
        if reasons else
        "No strong individual risk signals detected. Risk driven by combined behavioral patterns."
    )

    return PredictionResponse(
        customer_id=features.customer_id,
        churn_probability=round(proba, 4),
        predicted_class=pred,
        risk_level=risk,
        risk_explanation=explanation,
        confidence=confidence,
        threshold_used=threshold,
    )


# ── App ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="D2C Churn Scoring API",
    description=(
        "Internal CRM churn prediction service for a D2C personal-care brand. "
        "Predicts 60-day churn probability per customer based on behavioral, "
        "transactional, and support signals."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Health"], include_in_schema=False)
def root():
    """Redirect root to Swagger UI for easy testing."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness check — returns model load status."""
    loaded = _artifacts is not None
    artifacts_loaded = list(_artifacts.keys()) if loaded else []
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        model_path=MODEL_PATH,
        version="1.0.0",
        artifacts_loaded=artifacts_loaded,
        metrics=load_metrics(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Score a single customer",
    description=(
        "Accepts one customer feature payload and returns churn probability, "
        "risk level, and a plain-English explanation."
    ),
    responses={
        200: {
            "description": "Churn score for one customer",
            "content": {
                "application/json": {
                    "example": {
                        "customer_id": "CUST00042",
                        "churn_probability": 0.72,
                        "predicted_class": 1,
                        "risk_level": "High",
                        "risk_explanation": "Key risk signals: low recent activity; 2 support tickets recently.",
                        "confidence": "Medium",
                        "threshold_used": 0.27
                    }
                }
            },
        }
    },
)
def predict(payload: CustomerFeatures):
    """Score a single customer for churn probability."""
    try:
        arts = load_model()
        return make_prediction(payload, arts)
    except Exception as e:
        logger.error(f"Prediction error for {payload.customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/batch_predict",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Score multiple customers",
    description="Accepts multiple customer payloads and returns predictions for each.",
    responses={
        200: {
            "description": "Churn scores for a batch of customers",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {
                                "customer_id": "CUST00042",
                                "churn_probability": 0.72,
                                "predicted_class": 1,
                                "risk_level": "High",
                                "risk_explanation": "Key risk signals: low recent activity; 2 support tickets recently.",
                                "confidence": "Medium",
                                "threshold_used": 0.27
                            },
                            {
                                "customer_id": "CUST00043",
                                "churn_probability": 0.12,
                                "predicted_class": 0,
                                "risk_level": "Low",
                                "risk_explanation": "No strong individual risk signals detected. Risk driven by combined behavioral patterns.",
                                "confidence": "High",
                                "threshold_used": 0.27
                            }
                        ],
                        "total_customers": 2,
                        "high_risk_count": 1,
                        "processing_time_ms": 12.4
                    }
                }
            },
        }
    },
)
def batch_predict(payload: BatchRequest):
    """Score multiple customers (max 500) in one request."""
    t0 = time.time()
    try:
        arts = load_model()
        predictions = [make_prediction(c, arts) for c in payload.customers]
        elapsed_ms  = (time.time() - t0) * 1000
        high_risk   = sum(1 for p in predictions if p.predicted_class == 1)
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk,
            processing_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
