"""
preprocess.py
 
Takes raw PaySim data and produces clean, temporally-split train/val/test
CSVs ready for feature selection (features.py) and training (train.py).
 
Design decisions (see README.md for more context):
  - Scoped to TRANSFER and CASH_OUT only: fraud is structurally impossible
    in PAYMENT, CASH_IN, DEBIT in this dataset (confirmed via EDA).
  - isFlaggedFraud is removed from the feature set entirely. It's PaySim's
    own naive rule-based flag (a hardcoded amount threshold) and is kept
    aside only to compute a "naive baseline vs our model" comparison later.
  - orig_balance_delta / dest_balance_delta and their mismatch flags are
    engineered here as candidates. features.py decides which survive.
  - Split is temporal, cut on `step` value (not row count, not random),
    70/15/15 train/val/test. Val is reserved for cost-optimal threshold
    search (threshold.py); test is only ever touched for final reporting.
  - No scaling/normalization here — deferred to train.py so any scaler
    is fit on train only, never on val/test.
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/paysim.csv")
PROCESSED_DIR = Path("data/processed")

FRAUD_POSSIBLE_TYPES = ["TRANSFER", "CASH_OUT"]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# the remaining 0.15 is for test

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw PaySim CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape}")
    return df

def scope_to_fraud_possible_types(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the DataFrame to only rows where fraud is possible."""
    before = len(df)
    df = df[df["type"].isin(FRAUD_POSSIBLE_TYPES)].copy()
    print(f"Scoped to {FRAUD_POSSIBLE_TYPES}: {before} -> {len(df)} rows")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features for the model."""
    # Origin-side balance reconciliation
    # Expected: oldbalanceOrg - amount == newbalanceOrig if books are clean
    df["orig_balance_delta"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["orig_balance_mismatch"] = (df["orig_balance_delta"] != 0).astype(int)

    # Destination-side balance reconciliation
    # Expected: oldbalanceDest + amount == newbalanceDest if books are clean
    df["dest_balance_delta"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )
    df["dest_balance_mismatch"] = (df["dest_balance_delta"] != 0).astype(int)

    # Merchant destinations are tracked differently in the simulator
    # (ofter zero balances by construction) and, per PaySim's own
    # documentation, never appear as a fraud destinations. Worth keeping
    # explicityly as a feature rather than relying on the model to infer
    # it indirectly from balance columns.

    df["is_merchant_dest"] = df["nameDest"].str.startswith("M").astype(int)

    return df

def split_off_baseline_column(df:pd.DataFrame):
    """
    isFlaggedFraud is PaySim's own naive rule. We keep it aside, indexed
    the same as the main dataframe, purely to compute a basaeline
    recall/precision comparison later. It must never enter X.
    """

    baseline = df[["isFraud", "isFlaggedFraud"]].copy()
    df = df.drop(columns = ["isFlaggedFraud"])
    return df, baseline

def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are identifiers, not features."""
    # nameOrig/nameDest are hihg cardinality identifiers. Everything useful that
    # can be derived from them (is_merchant_dest) already has been engineered. They are not features, and they will overfit the model if left in.
    return df.drop(columns=["nameOrig", "nameDest"])

def temporal_split(df: pd.DataFrame):
    """
    Split the dataframe into train/val/test based on the `step` column.
    70% train, 15% val, 15% test.
    """
    min_step , max_step = df["step"].min(), df["step"].max()
    step_range = max_step - min_step

    train_cutoff = min_step + step_range * TRAIN_FRAC
    val_cutoff = min_step + step_range * (TRAIN_FRAC + VAL_FRAC)

    train = df[df["step"] <= train_cutoff].copy()
    val = df[(df["step"] > train_cutoff) & (df["step"] <= val_cutoff)].copy()
    test = df[df["step"] > val_cutoff].copy()

    print(f"Step range: {min_step}-{max_step}")
    print(f"Train cutoff step: {train_cutoff:.1f} -> {len(train)} rows, "
          f"fraud rate {train['isFraud'].mean():.4%}")
    print(f"Val   cutoff step: {val_cutoff:.1f} -> {len(val)} rows, "
          f"fraud rate {val['isFraud'].mean():.4%}")
    print(f"Test range: > {val_cutoff:.1f}   -> {len(test)} rows, "
          f"fraud rate {test['isFraud'].mean():.4%}")
 
    return train, val, test

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = scope_to_fraud_possible_types(df)
    df = engineer_features(df)
    df, baseline = split_off_baseline_column(df)
    df = drop_identifier_columns(df)

    train, val, test = temporal_split(df)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    baseline.to_csv(PROCESSED_DIR / "isFlaggedFraud_baseline.csv", index=False)

    print(f"\nSaved train.csv, val.csv, test.csv, isFlaggedFraud_baseline.csv " f"to {PROCESSED_DIR}/")

if __name__ == "__main__":
    main()