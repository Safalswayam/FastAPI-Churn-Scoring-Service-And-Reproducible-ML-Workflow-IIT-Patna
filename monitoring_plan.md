# Monitoring Plan — D2C Churn Scoring API

> **Service:** `/predict` and `/batch_predict` endpoints  
> **Model:** XGBoost Churn Classifier v1.0  
> **Owner:** Data Intelligence Team  
> **Review Cadence:** Weekly dashboard, monthly deep-dive

---

## 1. What to Monitor

### 1.1 Data Drift

Drift occurs when the distribution of incoming prediction requests diverges from the training data. This degrades model accuracy silently — predictions become unreliable without any error being raised.

| Feature | Monitoring Method | Alert Threshold |
|---------|------------------|-----------------|
| `recency_days` | KS test vs. training distribution | p-value < 0.05 |
| `sessions_30d` | PSI (Population Stability Index) | PSI > 0.20 |
| `monetary_180d` | Mean shift > 20% from training mean | > 20% shift |
| `ticket_count_90d` | Distribution histogram weekly | Visual + PSI > 0.20 |
| `loyalty_tier` (null rate) | % null per week | > 5% change from baseline |

**Tool recommendations:** Evidently AI, WhyLogs, or a custom weekly pandas comparison script.

---

### 1.2 Prediction Distribution

Monitor the distribution of churn probability scores to catch silent model degradation.

| Metric | Baseline (Training) | Alert Condition |
|--------|--------------------|-----------------| 
| Mean churn probability | ~0.47 | Shifts by > ±0.08 over 2 weeks |
| % predictions > 0.80 (Very High Risk) | Establish on first 30 days | Changes by > 15% |
| % predictions < 0.20 (Very Low Risk) | Establish on first 30 days | Changes by > 15% |

A sudden spike in high-risk predictions could mean:
- Genuine churn wave (investigate business context)
- Data quality issue in incoming features
- Model miscalibration

---

### 1.3 Business Outcomes (Ground Truth)

Every 60 days after a prediction, check actual purchase behavior against predicted churn.

| Metric | Target | Action if Breached |
|--------|--------|-------------------|
| ROC-AUC on rolling 60-day window | ≥ 0.75 | Trigger retraining |
| Recall on churned customers | ≥ 0.65 | Review threshold; consider retraining |
| Precision on predicted churners | ≥ 0.65 | Check for feature drift |
| % High-risk customers retained after intervention | Establish baseline | If < 10%: review intervention quality |

---

### 1.4 API Performance

| Metric | Target | Alert |
|--------|--------|-------|
| `/predict` p95 latency | < 100ms | > 200ms |
| `/batch_predict` (100 customers) latency | < 500ms | > 1,000ms |
| HTTP 5xx error rate | < 0.5% | > 2% |
| HTTP 422 validation error rate | < 5% | > 10% (may indicate upstream schema change) |

**Tool recommendations:** Prometheus + Grafana, or Datadog APM.

---

### 1.5 Model-Specific Drift Signals

| Signal | Definition | Alert |
|--------|-----------|-------|
| PSI (overall) | Population Stability Index across all numerical features | > 0.25 |
| Feature importance shift | Top 5 features change ranking significantly | Re-evaluate monthly |
| Calibration drift | Predicted probability vs. actual churn rate | > 0.10 absolute difference in decile bins |

---

## 2. Retraining Triggers

Retrain the model when any of the following occur:

1. **Scheduled:** Monthly rolling retrain on the most recent 9 months of data (regardless of metrics)
2. **Performance-triggered:** ROC-AUC drops below 0.75 on any 60-day ground-truth window
3. **Drift-triggered:** PSI > 0.25 on 3+ key features simultaneously
4. **Business event-triggered:** Major product launch, pricing change, or market disruption
5. **Data schema change:** New features added or existing features change definition

**Retraining pipeline:**
1. Pull latest data snapshot
2. Re-run feature engineering with same leakage rules
3. Re-train with same hyperparameter search
4. Validate on holdout: must beat current model by ≥ 0.01 ROC-AUC
5. Shadow-deploy for 1 week before full cutover

---

## 3. Logging Requirements

Every prediction request should log:

```json
{
  "timestamp": "2025-10-01T09:15:32Z",
  "customer_id": "CUST00001",
  "churn_probability": 0.7234,
  "predicted_class": 1,
  "risk_level": "High",
  "threshold_used": 0.48,
  "api_version": "1.0.0",
  "latency_ms": 18.4
}
```

Store logs in a time-series store (e.g., BigQuery, ClickHouse) for batch analysis.

---

## 4. Dashboard Checklist (Weekly)

- [ ] Mean churn probability this week vs. last week
- [ ] % high-risk customers flagged
- [ ] API error rate (5xx + 422)
- [ ] p95 latency
- [ ] PSI for top 5 features
- [ ] Ground truth comparison (for batches with 60-day outcome available)
- [ ] Intervention conversion rate (if CRM system reports back)

---

## 5. Responsible Use of Model Output

### ✅ Appropriate Uses

- **Prioritizing outreach lists:** Use the churn score to rank customers for the CRM team's weekly call/email queue. Focus human effort on the top 20% by risk.
- **Campaign targeting:** Serve different retention message types based on risk tier (Very High vs. Medium).
- **Reporting:** Aggregate churn risk trends for leadership dashboards and product decisions.

### ❌ Inappropriate Uses

| Misuse | Risk | Correct Alternative |
|--------|------|---------------------|
| Automatically applying discounts based on churn score alone | Trains customers to game the system; erodes margins | Use score to trigger human review, not automatic discounts |
| Denying service/upgrades to low-score customers | Not what the model is designed for; potential discrimination | Model is retention-only; do not repurpose for credit/eligibility |
| Treating churn score as definitive truth | Model has ~22% error rate at threshold | Always review high-stakes cases with additional context |
| Using for hiring, lending, or legal decisions | Model was not validated for these use cases | Use only for internal retention targeting |

### Human Oversight Requirements

- **Score ≥ 0.80 (Very High):** CRM agent must personally review before deciding on intervention type
- **High-value customers (LTV > ₹10,000):** Regardless of score, require manager sign-off on retention offer
- **Customers with recent unresolved complaints:** Route to support team first; do not send marketing offers

---

*Last updated: October 2025 | Model Version: 1.0.0*
