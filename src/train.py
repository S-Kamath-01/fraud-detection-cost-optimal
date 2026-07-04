"""
train.py

Trains a class-weighted logistic regression baseline (untuned, reference
point only) and an XGBoost model (production model — this is what flows
into threshold.py, explain.py, and main.py).

Class imbalance is handled via class weighting, not resampling:
  - No synthetic data invented (as SMOTE would).
  - No real data discarded (as undersampling would) - train has 2.65M rows,
    no practical need to throw legitimate examples away.
  - Keeps output probabilities honest/uncalibrated-by-resampling, which
    matters downstream: threshold.py needs real probabilities to search
    over, and explain.py's SHAP values need to reflect the actual model,
    not a model trained on a rebalanced synthetic distribution.

Only threshold-agnostic metrics (ROC-AUC, PR-AUC) are reported here, plus
a 0.5-threshold confusion matrix labeled explicitly as provisional. The
real decision threshold is chosen in threshold.py from a cost matrix, not
here.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb
import json
from pathlib import Path

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
CHECKPOINT_DIR = Path("checkpoints")

FEATURE_COLUMNS = ["amount", "oldbalanceOrg", "newbalanceOrig", "is_transfer"]


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV and prepare X, y for training or validation."""
    df = pd.read_csv(path)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    X = df[FEATURE_COLUMNS]
    y = df["isFraud"]
    return X, y


def evaluate(name: str, y_true: pd.Series, y_proba: np.ndarray):
    """
    Evaluate a model on a validation set, printing ROC-AUC, PR-AUC, and a provisional confusion matrix. 
    """
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    y_pred_provisional = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_provisional)

    print(f"--- {name} (val set) ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}  "
          f"(more informative here — base rate is {y_true.mean():.4%}, "
          f"so ROC-AUC alone can look deceptively strong)")
    print("Confusion matrix at 0.5 threshold (PROVISIONAL — real threshold "
          "chosen in threshold.py from a cost matrix, not here):")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print()

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def train_baseline(X_train, y_train):
    """
    Train a class-weighted logistic regression baseline model. This is a
    reference point only, not the production model.
    """
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train a class-weighted XGBoost model. This is the production model that
    flows into threshold.py, explain.py, and main.py.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight computed from train: {scale_pos_weight:.2f} "
          f"({n_neg} negative / {n_pos} positive)\n")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    X_train, y_train = load_and_prepare(TRAIN_PATH)
    X_val, y_val = load_and_prepare(VAL_PATH)

    print(f"Train: {X_train.shape}, fraud rate {y_train.mean():.4%}")
    print(f"Val:   {X_val.shape}, fraud rate {y_val.mean():.4%}\n")

    # --- Baseline ---
    baseline = train_baseline(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_val)[:, 1]
    baseline_metrics = evaluate("Logistic Regression (baseline)", y_val, baseline_proba)

    # --- Production model ---
    xgb_model = train_xgboost(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
    xgb_metrics = evaluate("XGBoost (production)", y_val, xgb_proba)

    print("--- Comparison ---")
    print(f"{'Model':<30} {'ROC-AUC':<10} {'PR-AUC':<10}")
    print(f"{'Logistic Regression':<30} {baseline_metrics['roc_auc']:<10.4f} "
          f"{baseline_metrics['pr_auc']:<10.4f}")
    print(f"{'XGBoost':<30} {xgb_metrics['roc_auc']:<10.4f} "
          f"{xgb_metrics['pr_auc']:<10.4f}")


    model_path = CHECKPOINT_DIR / "xgb_model.json"
    xgb_model.save_model(model_path)
    print(f"\nSaved XGBoost model to {model_path}")


    with open(CHECKPOINT_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    drift_reference = X_train.sample(n=min(50_000, len(X_train)), random_state=42)
    drift_reference_path = CHECKPOINT_DIR / "drift_reference.csv"
    drift_reference.to_csv(drift_reference_path, index=False)
    print(f"Saved drift reference sample ({len(drift_reference)} rows) to "
          f"{drift_reference_path}")


if __name__ == "__main__":
    main()