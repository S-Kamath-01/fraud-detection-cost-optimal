"""
explain.py

SHAP explainability for the trained XGBoost model.

Two purposes:
  1. Global summary - which features matter most overall, run once here,
     saved as a plot for the README/portfolio writeup.
  2. explain_prediction() - a reusable per-row explanation function that
     main.py imports and calls for every /predict request, so every
     decision the API makes comes with a human-readable reason, not just
     a probability.

Uses shap.TreeExplainer, not the generic KernelExplainer - TreeExplainer
computes exact SHAP values for tree ensembles and is fast enough to run
per-request in the API; KernelExplainer is model-agnostic but slow and
approximate, and there's no reason to use it when the model is XGBoost.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
import matplotlib.pyplot as plt
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")
IMG_DIR = Path("assets/img")
MODEL_PATH = CHECKPOINT_DIR / "xgb_model.json"
FEATURES_PATH = CHECKPOINT_DIR / "feature_columns.json"
TRAIN_PATH = "data/processed/train.csv"

SUMMARY_SAMPLE_SIZE = 5000

FEATURE_LABELS = {
    "amount": "transaction amount",
    "oldbalanceOrg": "sender's balance before the transaction",
    "newbalanceOrig": "sender's balance after the transaction",
    "is_transfer": "transaction is a TRANSFER (vs. CASH_OUT)",
}


def load_model_and_features():
    with open(FEATURES_PATH) as f:
        feature_columns = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model, feature_columns


def build_explainer(model):
    return shap.TreeExplainer(model)


def global_summary(model, feature_columns):
    """
    Computes SHAP values on a sample of train data and saves a summary
    bar plot showing average feature importance. Sampled, not run on the
    full 2.65M rows, purely for speed - this is a one-time diagnostic,
    not something that runs per-request.
    """
    df = pd.read_csv(TRAIN_PATH)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    sample = df[feature_columns].sample(
        n=min(SUMMARY_SAMPLE_SIZE, len(df)), random_state=42
    )

    explainer = build_explainer(model)
    shap_values = explainer.shap_values(sample)

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_columns
    ).sort_values(ascending=False)

    print("--- Global feature importance (mean |SHAP value|, sampled train) ---")
    print(mean_abs_shap.to_string())

    plt.figure(figsize=(8, 4))
    mean_abs_shap.sort_values().plot(kind="barh")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global feature importance")
    plt.tight_layout()
    plot_path = IMG_DIR / "shap_summary.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved summary plot to {plot_path}")

    return mean_abs_shap


def explain_prediction(model, explainer, feature_columns: list, row: dict) -> dict:
    """
    Explains a single prediction. `row` is a dict of {feature: value} for
    exactly the features in feature_columns. Returns the probability, the
    per-feature SHAP contributions, and a plain-language explanation
    string ranking the top contributing factors.

    This is the function main.py will import and call for every
    /predict request.
    """
    X = pd.DataFrame([row])[feature_columns]
    proba = float(model.predict_proba(X)[:, 1][0])

    shap_values = explainer.shap_values(X)[0]
    contributions = dict(zip(feature_columns, shap_values.tolist()))

    # Rank by absolute contribution, describe top 3 in plain language
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_factors = []
    for feature, value in ranked[:3]:
        direction = "increased" if value > 0 else "decreased"
        label = FEATURE_LABELS.get(feature, feature)
        top_factors.append(f"{label} {direction} fraud probability "
                            f"(contribution: {value:+.3f})")

    explanation_text = (
        f"Predicted fraud probability: {proba:.4f}. "
        f"Top contributing factors: " + "; ".join(top_factors) + "."
    )

    return {
        "probability": proba,
        "shap_contributions": contributions,
        "explanation": explanation_text,
    }


def main():
    model, feature_columns = load_model_and_features()

    global_summary(model, feature_columns)

    # Demo: explain a single example row so the output is inspectable
    # without needing main.py built yet.
    explainer = build_explainer(model)
    df = pd.read_csv(TRAIN_PATH)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    example_fraud = df[df["isFraud"] == 1].iloc[0]
    example_row = {col: example_fraud[col] for col in feature_columns}

    print("\n--- Example: explaining a single known-fraud row ---")
    result = explain_prediction(model, explainer, feature_columns, example_row)
    print(result["explanation"])
    print("\nFull SHAP contributions:")
    for feature, value in result["shap_contributions"].items():
        print(f"  {feature}: {value:+.4f}")

    print("\nExample row feature values:")
    print(example_row)  

if __name__ == "__main__":
    main()