"""
features.py
 
Diagnostic script for feature selection. Does NOT output a final feature
set automatically — prints correlation and group-mean evidence so the
final feature list can be chosen deliberately (see train.py once written),
with reasoning documented rather than a blind correlation threshold.
 
Runs on data/processed/train.csv ONLY. Val/test are never touched during
feature selection - using them here would be a (mild, but real) form of
leakage into a decision that affects the final model.
"""

import pandas as pd

TRAIN_PATH = "data/processed/train.csv"
CANDIDATE_NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_delta",
    "dest_balance_delta",
    "orig_balance_mismatch",
    "dest_balance_mismatch",
    "is_merchant_dest",
]

def load_train():
    df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded train data: {df.shape}, fraud rate: {df['isFraud'].mean():.4f}\n")
    return df

def correlation_table(df: pd.DataFrame):
    """Print a table of correlations between candidate numeric features and the target."""
    print("--- Point-biserial correlation with isFraud (train only) ---")
    corrs = df[CANDIDATE_NUMERIC_FEATURES + ["isFraud"]].corr()["isFraud"]
    corrs = corrs.drop("isFraud").sort_values(key=abs, ascending=False)
    print(corrs.to_string())
    print()

def group_means(df: pd.DataFrame, cols):
    """Print a table of group means for candidate numeric features, grouped by the target."""
    print("--- Fraud vs non-fraud group means ---")
    grouped = df.groupby("isFraud")[cols].mean().T
    grouped.columns = ["mean_non_fraud", "mean_fraud"]
    grouped["ratio_fraud_over_nonfraud"] = (
        grouped["mean_fraud"] / grouped["mean_non_fraud"].replace(0, float("nan"))
    )
    print(grouped.to_string())
    print()

def check_destination_side(df: pd.DataFrame):
    """Check if the destination is a merchant (sid=0) or not (sid!=0)."""
    print("--- Destination-side deep dive (suspected simulator artifact) ---")
    cols = ["oldbalanceDest", "newbalanceDest", "dest_balance_delta",
            "dest_balance_mismatch"]
    group_means(df, cols)
 
    # How often is newbalanceDest exactly 0 for fraud vs non-fraud?
    zero_dest_by_class = df.groupby("isFraud")["newbalanceDest"].apply(
        lambda s: (s == 0).mean()
    )
    print("Fraction of rows where newbalanceDest == 0, by class:")
    print(zero_dest_by_class.to_string())
    print()

def check_merchant_flag(df: pd.DataFrame):
    """Check if the destination is a merchant (is_merchant_dest=1) or not (is_merchant_dest=0)."""
    print("--- is_merchant_dest sanity check (within TRANSFER/CASH_OUT only) ---")
    print(df["is_merchant_dest"].value_counts(normalize=True).to_string())
    print()

def check_type_split(df: pd.DataFrame):
    """Check fraud rates by transaction type."""
    print("--- type vs isFraud (within scoped data) ---")
    print(pd.crosstab(df["type"], df["isFraud"], normalize="index").to_string())
    print()

def check_skew(df: pd.DataFrame):
    """Check skew of candidate numeric features."""
    print("--- Skew of raw amount/balance columns (flag for log-transform "
          "consideration in train.py, not applied here) ---")
    cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"]
    print(df[cols].skew().to_string())
    print()

def main():
    df = load_train()
    correlation_table(df)
    group_means(df, CANDIDATE_NUMERIC_FEATURES)
    check_destination_side(df)
    check_merchant_flag(df)
    check_type_split(df)
    check_skew(df)
 
 
if __name__ == "__main__":
    main()