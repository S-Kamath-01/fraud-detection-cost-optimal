"""
drift.py

Distribution drift detection using PSI (Population Stability Index) and
the KS (Kolmogorov-Smirnov) statistic.

No live prediction log exists yet (that's main.py + Postgres, not built).
As a stand-in, this compares TRAIN (reference distribution) against TEST
(current distribution) - which usefully doubles as a real test of the
volume/fraud-rate drift already found and documented during preprocessing,
rather than a synthetic demo.

drift_report() is written generically (reference_df, current_df) so it can
be reused unchanged later: main.py will eventually call it with train vs.
"last N days of logged Postgres predictions" instead of train vs test.

PSI interpretation (standard industry thresholds):
  < 0.10           no significant shift
  0.10 - 0.25      moderate shift, worth investigating
  > 0.25           significant shift, model likely needs retraining

Important: PSI/KS on isFraud rate is a misleading signal on its own in this
dataset, since fraud COUNT is roughly constant over time while transaction
VOLUME collapses (see README Findings). Drift is checked per-FEATURE
instead, since feature distribution shift is what actually invalidates a
model's assumptions, independent of the volume artifact.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

FEATURE_COLUMNS = ["amount", "oldbalanceOrg", "newbalanceOrig", "is_transfer"]

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25

N_BINS = 10


def load_with_encoding(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    return df


def calculate_psi(reference: pd.Series, current: pd.Series, n_bins: int = N_BINS) -> float:
    """
    PSI for a single feature. Bins are defined by quantiles of the
    REFERENCE distribution only (train), then both reference and current
    are binned using those same edges - this is standard practice, since
    PSI measures how much the current distribution has shifted relative
    to the reference's own shape, not relative to some new binning of
    itself.
    """
    # Quantile-based bin edges from reference, deduplicated in case of
    # heavy ties (e.g. is_transfer is binary, quantiles may collapse).
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(reference.quantile(quantiles).values)

    if len(bin_edges) < 3:
        # Not enough distinct values to bin meaningfully (e.g. a binary
        # feature) - fall back to using the distinct values as bins.
        bin_edges = np.unique(np.concatenate([[-np.inf], bin_edges, [np.inf]]))

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Avoid log(0) / division by zero for empty bins
    epsilon = 1e-6
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def interpret_psi(psi: float) -> str:
    if psi < PSI_MODERATE_THRESHOLD:
        return "no significant shift"
    elif psi < PSI_SIGNIFICANT_THRESHOLD:
        return "MODERATE shift - worth investigating"
    else:
        return "SIGNIFICANT shift - retraining likely warranted"


def drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                  feature_columns: list) -> pd.DataFrame:
    """
    Generic drift report between any two dataframes on the given feature
    columns. Reusable as-is once main.py exists: call with
    (train_df, recent_predictions_df, FEATURE_COLUMNS).
    """
    rows = []
    for feature in feature_columns:
        psi = calculate_psi(reference_df[feature], current_df[feature])
        ks_stat, ks_pvalue = ks_2samp(reference_df[feature], current_df[feature])

        rows.append({
            "feature": feature,
            "psi": psi,
            "psi_interpretation": interpret_psi(psi),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
        })

    return pd.DataFrame(rows)


def main():
    train = load_with_encoding(TRAIN_PATH)
    test = load_with_encoding(TEST_PATH)

    print(f"Reference (train): {len(train)} rows")
    print(f"Current (test):    {len(test)} rows\n")

    report = drift_report(train, test, FEATURE_COLUMNS)
    print("--- Drift report: train (reference) vs test (current) ---")
    print(report.to_string(index=False))

    print("\nNote: fraud rate itself is NOT included as a drift feature here "
          "- fraud count is roughly constant over time in this dataset while "
          "transaction volume collapses (see README), so raw fraud-rate "
          "drift would conflate a known volume artifact with genuine "
          "feature distribution shift. Drift is measured on the model's "
          "actual input features instead.")

    print("\nNote on KS p-values: with reference/current sample sizes this "
          "large (train ~2.65M, test ~38K), the KS test has enough "
          "statistical power to flag even trivially small distributional "
          "differences as 'significant' (p near 0) regardless of whether "
          "the difference is practically meaningful. The KS STATISTIC "
          "(effect size, 0-1) is the number that matters here, not the "
          "p-value - and PSI, which measures magnitude directly rather "
          "than statistical significance, is the more reliable signal at "
          "this scale.")

    any_significant = (report["psi"] >= PSI_SIGNIFICANT_THRESHOLD).any()
    if any_significant:
        print("\n>>> At least one feature shows SIGNIFICANT drift between "
              "train and test. This is expected here: test is a "
              "later, lower-volume slice of simulated time than train, "
              "and demonstrates the monitor correctly detecting the same "
              "shift already found during preprocessing.")


if __name__ == "__main__":
    main()