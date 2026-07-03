"""
threshold.py

Finds the classification threshold that minimizes total expected business
cost on the validation set, rather than defaulting to 0.5 or optimizing a
threshold-agnostic metric like F1.

Cost model (configurable below):
  - False positive: fixed cost. Blocking a legitimate transaction costs
    roughly the same in support/review overhead regardless of the
    transaction's size - the friction is the same shape whether the
    blocked transaction was 1,000 or 1,00,000.
  - False negative: fixed cost + the transaction amount itself. Missing
    fraud always costs some fixed investigation/write-off overhead, plus
    the actual money lost - which scales directly with amount.

The threshold is searched on the VALIDATION set only. Test is reserved
for final, one-time reporting after the threshold is locked in - searching
on test would let the threshold choice leak information from the set
we're supposed to be evaluating on.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path

VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
CHECKPOINT_DIR = Path("checkpoints")
MODEL_PATH = CHECKPOINT_DIR / "xgb_model.json"
FEATURES_PATH = CHECKPOINT_DIR / "feature_columns.json"

# --- Cost matrix (configurable) ---
FP_COST = 500          # fixed cost of blocking a legitimate transaction
FN_FIXED_COST = 2000    # fixed investigation/write-off overhead per missed fraud
# Total FN cost for a given row = FN_FIXED_COST + amount

THRESHOLD_RANGE = np.arange(0.01, 1.00, 0.01)


def load_with_predictions(path: str):
    """
    Loads a CSV file, adds a column of predicted probabilities from the
    pre-trained XGBoost model, and returns the DataFrame.
    """
    df = pd.read_csv(path)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)

    with open(FEATURES_PATH) as f:
        feature_columns = json.load(f)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    X = df[feature_columns]
    df["proba"] = model.predict_proba(X)[:, 1]
    return df


def cost_at_threshold(df: pd.DataFrame, threshold: float) -> dict:
    """
    Computes the total expected business cost at a given classification threshold.
    """
    predicted_fraud = df["proba"] >= threshold

    fp_mask = predicted_fraud & (df["isFraud"] == 0)
    fn_mask = (~predicted_fraud) & (df["isFraud"] == 1)

    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()

    fp_total_cost = fp_count * FP_COST
    fn_total_cost = (FN_FIXED_COST * fn_count) + df.loc[fn_mask, "amount"].sum()

    total_cost = fp_total_cost + fn_total_cost

    return {
        "threshold": threshold,
        "fp_count": int(fp_count),
        "fn_count": int(fn_count),
        "fp_cost": float(fp_total_cost),
        "fn_cost": float(fn_total_cost),
        "total_cost": float(total_cost),
    }


def search(df: pd.DataFrame) -> pd.DataFrame:
    """
    Searches over a range of thresholds and computes the total expected business
    cost for each threshold. Returns a DataFrame with the results.
    """
    results = [cost_at_threshold(df, t) for t in THRESHOLD_RANGE]
    return pd.DataFrame(results)


def main():
    print(f"Cost matrix: FP_COST={FP_COST} (fixed), "
          f"FN_COST=amount + {FN_FIXED_COST} (fixed)\n")

    # --- Search on VAL only ---
    val_df = load_with_predictions(VAL_PATH)
    results = search(val_df)

    best = results.loc[results["total_cost"].idxmin()]
    chosen_threshold = float(best["threshold"])

    print("--- Cost by threshold on VAL (every 10th step shown) ---")
    print(results.iloc[::10].to_string(index=False))
    print()

    print("--- Threshold chosen on VAL (frozen from here on) ---")
    print(f"Threshold: {chosen_threshold:.2f}")
    print(f"VAL total cost at this threshold: {best['total_cost']:,.0f}")

    naive_val = cost_at_threshold(val_df, 0.5)
    savings_val = naive_val["total_cost"] - best["total_cost"]
    pct_val = (savings_val / naive_val["total_cost"]) * 100
    print(f"VAL total cost at naive 0.5: {naive_val['total_cost']:,.0f}")
    print(f"VAL savings from cost-optimal threshold: "
          f"{savings_val:,.0f} ({pct_val:.1f}%)\n")

    # --- Final, one-time evaluation on untouched TEST set ---
    # The threshold above is now frozen. Test is only ever used here, to
    # report a final honest number - never to search for a better threshold.
    print("--- FINAL evaluation on TEST (threshold frozen, not re-searched) ---")
    test_df = load_with_predictions(TEST_PATH)

    test_optimal = cost_at_threshold(test_df, chosen_threshold)
    test_naive = cost_at_threshold(test_df, 0.5)

    print(f"TEST fraud rate: {test_df['isFraud'].mean():.4%}")
    print(f"TEST cost at naive 0.5:                  {test_naive['total_cost']:,.0f} "
          f"(FP={test_naive['fp_count']}, FN={test_naive['fn_count']})")
    print(f"TEST cost at frozen threshold ({chosen_threshold:.2f}):     "
          f"{test_optimal['total_cost']:,.0f} "
          f"(FP={test_optimal['fp_count']}, FN={test_optimal['fn_count']})")

    test_savings = test_naive["total_cost"] - test_optimal["total_cost"]
    test_pct = (test_savings / test_naive["total_cost"]) * 100
    print(f"TEST savings from cost-optimal threshold: "
          f"{test_savings:,.0f} ({test_pct:.1f}%)")
    print("\n^ This is the headline number: unseen-data validation that "
          "choosing the threshold from business cost, rather than "
          "defaulting to 0.5, produces a real reduction in expected cost.")

    # Save chosen threshold + cost matrix + test results as deployment artifacts
    output = {
        "threshold": chosen_threshold,
        "fp_cost": FP_COST,
        "fn_fixed_cost": FN_FIXED_COST,
        "test_cost_at_threshold": test_optimal["total_cost"],
        "test_cost_at_naive_0.5": test_naive["total_cost"],
        "test_savings_pct": test_pct,
    }
    with open(CHECKPOINT_DIR / "threshold_config.json", "w") as f:
        json.dump(output, f, indent=2)

    results.to_csv(CHECKPOINT_DIR / "threshold_search_results.csv", index=False)
    print(f"\nSaved threshold_config.json (deployment artifact) and "
          f"threshold_search_results.csv to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()