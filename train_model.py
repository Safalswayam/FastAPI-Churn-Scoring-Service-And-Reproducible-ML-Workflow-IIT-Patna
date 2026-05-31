"""
train_model.py — Standalone model training script for Part 4
D2C Customer Churn Intelligence & Retention API

Run this script FIRST if you do not have a pre-trained model.pkl:
    python train_model.py

This trains the XGBoost churn model from the raw dataset and saves
artifacts/model.pkl ready for the FastAPI service.

Usage:
    python train_model.py --data-dir ./capstone_data
    python train_model.py --data-dir /path/to/capstone_data
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
import xgboost as xgb

SNAPSHOT = pd.Timestamp("2025-09-30")

CAT_COLS = ["city_tier","age_group","acquisition_channel","loyalty_tier",
            "preferred_category","marketing_consent"]
NUM_BASE = ["recency_days","frequency_180d","monetary_180d","return_rate_180d",
            "avg_discount_pct_180d","avg_rating_180d","category_diversity_180d",
            "ticket_count_90d","negative_ticket_rate_90d","avg_resolution_hours_90d",
            "days_since_signup","sessions_30d","product_views_30d","cart_adds_30d",
            "wishlist_adds_30d","abandoned_carts_30d","email_opens_30d",
            "campaign_clicks_30d","last_visit_days_ago"]
NUM_DERIVED = ["engagement_score","cart_abandon_rate","complaint_intensity","has_loyalty"]
CAT_DERIVED = ["recency_bucket"]
ALL_NUM = NUM_BASE + NUM_DERIVED
ALL_CAT = CAT_COLS + CAT_DERIVED

FN_COST = 500   # ₹ cost of missing a churner (lost LTV)
FP_COST = 30    # ₹ cost of false alarm (wasted intervention)


def engineer(df):
    df = df.copy()
    df["engagement_score"] = (
        df["sessions_30d"] * 2 +
        df["product_views_30d"] * 0.5 +
        df["email_opens_30d"] +
        df["campaign_clicks_30d"] * 1.5
    )
    df["cart_abandon_rate"] = np.where(
        df["sessions_30d"] > 0,
        df["abandoned_carts_30d"] / df["sessions_30d"].clip(lower=1), 0
    )
    df["complaint_intensity"] = df["ticket_count_90d"] * (1 + df["negative_ticket_rate_90d"])
    df["has_loyalty"] = df["loyalty_tier"].notna().astype(int)
    df["recency_bucket"] = pd.cut(
        df["recency_days"], bins=[-1, 7, 30, 60, 90, 999],
        labels=["≤7d","8-30d","31-60d","61-90d",">90d"]
    ).astype(str)
    return df


def get_Xy(df, target="churn_next_60d"):
    X = df[ALL_NUM + ALL_CAT].copy()
    X[ALL_NUM] = X[ALL_NUM].fillna(0).astype(float)
    X[ALL_CAT] = X[ALL_CAT].fillna("Unknown").astype(str)
    y = df[target].astype(int)
    return X, y


def train(data_dir: str, output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load snapshot ──────────────────────────────────────────────────────────
    snap_path = os.path.join(data_dir, "rfm_modeling_snapshot.csv")
    if not os.path.exists(snap_path):
        print(f"ERROR: Cannot find {snap_path}")    
        print("Download the dataset from the project Google Drive and place CSVs in --data-dir")
        sys.exit(1)

    print(f"Loading snapshot from: {snap_path}")
    snap = pd.read_csv(snap_path)

    # Leakage verification
    assert snap["snapshot_date"].iloc[0] == "2025-09-30", "Snapshot date mismatch — leakage risk!"
    print(" Snapshot date verified: 2025-09-30 - no post-snapshot data used")

    train_df = snap[snap["split"] == "train"].copy()
    val_df   = snap[snap["split"] == "validation"].copy()
    test_df  = snap[snap["split"] == "test"].copy()

    print(f"  Train: {len(train_df):,}  | Val: {len(val_df):,}  | Test: {len(test_df):,}")
    print(f"  Churn rates - Train: {train_df['churn_next_60d'].mean():.1%}  Val: {val_df['churn_next_60d'].mean():.1%}  Test: {test_df['churn_next_60d'].mean():.1%}")

    # ── Feature engineering ────────────────────────────────────────────────────
    train_df = engineer(train_df)
    val_df   = engineer(val_df)
    test_df  = engineer(test_df)

    X_train, y_train = get_Xy(train_df)
    X_val,   y_val   = get_Xy(val_df)
    X_test,  y_test  = get_Xy(test_df)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ALL_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ALL_CAT),
    ])
    X_train_pp = preprocessor.fit_transform(X_train, y_train)
    X_val_pp   = preprocessor.transform(X_val)
    X_test_pp  = preprocessor.transform(X_test)

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    print("\nTraining Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced",
                            solver="lbfgs", random_state=42)
    lr.fit(X_train_pp, y_train)
    lr_proba = lr.predict_proba(X_val_pp)[:, 1]
    lr_pred  = (lr_proba >= 0.5).astype(int)

    lr_metrics = {
        "model": "Logistic Regression (Baseline)",
        "split": "validation",
        "roc_auc": round(roc_auc_score(y_val, lr_proba), 4),
        "pr_auc": round(average_precision_score(y_val, lr_proba), 4),
        "f1": round(f1_score(y_val, lr_pred), 4),
        "precision": round(precision_score(y_val, lr_pred), 4),
        "recall": round(recall_score(y_val, lr_pred), 4),
    }
    print("  Logistic Regression (Baseline) — Validation")
    for k, v in lr_metrics.items():
        print(f"    {k:<12} {v}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("Training XGBoost model...")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        use_label_encoder=False, eval_metric="auc",
        early_stopping_rounds=30, random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train_pp, y_train, eval_set=[(X_val_pp, y_val)], verbose=False)
    xgb_val_proba = xgb_model.predict_proba(X_val_pp)[:, 1]
    xgb_val_pred  = (xgb_val_proba >= 0.5).astype(int)
    xgb_metrics = {
        "model": "XGBoost",
        "split": "validation",
        "roc_auc": round(roc_auc_score(y_val, xgb_val_proba), 4),
        "pr_auc": round(average_precision_score(y_val, xgb_val_proba), 4),
        "f1": round(f1_score(y_val, xgb_val_pred), 4),
        "precision": round(precision_score(y_val, xgb_val_pred), 4),
        "recall": round(recall_score(y_val, xgb_val_pred), 4),
    }
    print("  XGBoost — Validation")
    for k, v in xgb_metrics.items():
        print(f"    {k:<12} {v}")
    print(f"\n  Improvement over baseline: ROC-AUC +{xgb_metrics['roc_auc']-lr_metrics['roc_auc']:.4f}")

    # ── Business cost-aware threshold ─────────────────────────────────────────
    print("Selecting business-optimal threshold...")
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_val, xgb_val_proba)
    results = []
    for t in thresh_arr:
        pred = (xgb_val_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        cost = fn * FN_COST + fp * FP_COST
        results.append({
            "threshold": round(float(t), 3),
            "fn": int(fn),
            "fp": int(fp),
            "tp": int(tp),
            "recall": recall_score(y_val, pred),
            "precision": precision_score(y_val, pred),
            "f1": f1_score(y_val, pred),
            "total_cost": cost,
        })
    res_df = pd.DataFrame(results)
    best_idx = res_df["total_cost"].idxmin()
    best_thresh = float(res_df.loc[best_idx, "threshold"])
    print(f"  Selected threshold: {best_thresh}  (minimises INR FN*{FN_COST} + FP*{FP_COST})")

    # ── Test set evaluation ────────────────────────────────────────────────────
    xgb_test_proba = xgb_model.predict_proba(X_test_pp)[:, 1]
    xgb_test_pred  = (xgb_test_proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, xgb_test_pred).ravel()

    test_metrics = {
        "model": "XGBoost (Final)",
        "split": "test",
        "threshold": best_thresh,
        "roc_auc": round(roc_auc_score(y_test, xgb_test_proba), 4),
        "pr_auc": round(average_precision_score(y_test, xgb_test_proba), 4),
        "f1": round(f1_score(y_test, xgb_test_pred), 4),
        "precision": round(precision_score(y_test, xgb_test_pred), 4),
        "recall": round(recall_score(y_test, xgb_test_pred), 4),
    }
    print(f"\n  Test ROC-AUC: {test_metrics['roc_auc']:.4f}  |  F1: {test_metrics['f1']:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    artifacts = {
        "preprocessor":  preprocessor,
        "xgb_model":     xgb_model,
        "lr_pipeline":   lr,
        "feature_names": {"num": ALL_NUM, "cat": ALL_CAT},
        "threshold":     best_thresh,
    }
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(artifacts, model_path)
    # Also save to root for direct API loading
    joblib.dump(artifacts, "artifacts/model.pkl")

    metrics = {
        "baseline_logistic_regression": lr_metrics,
        "xgboost_validation": xgb_metrics,
        "xgboost_test_final": {**test_metrics},
        "confusion_matrix_test": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "business_cost_at_threshold": {
            "fn_cost_per_customer": FN_COST,
            "fp_cost_per_customer": FP_COST,
            "total_test_cost": int(fn * FN_COST + fp * FP_COST),
        },
        "feature_count": len(ALL_NUM) + len(ALL_CAT),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nYou can now start the API:")
    print("  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train D2C Churn model")
    parser.add_argument("--data-dir", default="capstone_data/",
                        help="Path to directory containing the CSV dataset files")
    parser.add_argument("--output-dir", default="artifacts",
                        help="Directory to save model.pkl")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
