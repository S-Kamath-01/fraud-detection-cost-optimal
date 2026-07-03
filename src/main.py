"""
main.py

FastAPI app tying together the trained model, the frozen cost-optimal
threshold, and SHAP explainability. Every /predict call is logged to
Postgres, which /drift-report reads from to compare recent live traffic
against the training distribution.

Model, threshold, feature list, and SHAP explainer are all loaded ONCE at
startup (module-level, via a lifespan handler) - not per-request, since
constructing the SHAP explainer and loading the model are non-trivial costs
that shouldn't repeat on every call.
"""

import os
import json
from contextlib import asynccontextmanager

import pandas as pd
import xgboost as xgb
import shap
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.drift import drift_report, FEATURE_COLUMNS as DRIFT_FEATURE_COLUMNS

CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "xgb_model.json")
FEATURES_PATH = os.path.join(CHECKPOINT_DIR, "feature_columns.json")
THRESHOLD_CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "threshold_config.json")
TRAIN_PATH = "data/processed/train.csv"

MODEL_VERSION = "xgb-v1"

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/fraud_detection"
)

FEATURE_LABELS = {
    "amount": "transaction amount",
    "oldbalanceOrg": "sender's balance before the transaction",
    "newbalanceOrig": "sender's balance after the transaction",
    "is_transfer": "transaction is a TRANSFER (vs. CASH_OUT)",
}

# Populated at startup by the lifespan handler
state = {
    "model": None,
    "explainer": None,
    "feature_columns": None,
    "threshold": None,
    "db_connected": False,
}


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    conn = get_db_connection()
    with conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                amount DOUBLE PRECISION NOT NULL,
                old_balance_org DOUBLE PRECISION NOT NULL,
                new_balance_orig DOUBLE PRECISION NOT NULL,
                is_transfer INTEGER NOT NULL,
                predicted_probability DOUBLE PRECISION NOT NULL,
                decision TEXT NOT NULL,
                threshold_used DOUBLE PRECISION NOT NULL,
                model_version TEXT NOT NULL,
                shap_contributions JSONB
            )
        """)
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load model, threshold, feature list, explainer, DB ---
    with open(FEATURES_PATH) as f:
        state["feature_columns"] = json.load(f)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    state["model"] = model
    state["explainer"] = shap.TreeExplainer(model)

    with open(THRESHOLD_CONFIG_PATH) as f:
        threshold_config = json.load(f)
    state["threshold"] = threshold_config["threshold"]

    try:
        init_db()
        state["db_connected"] = True
    except Exception as e:
        print(f"WARNING: could not connect to database at startup: {e}")
        state["db_connected"] = False

    yield
    # --- Shutdown: nothing to clean up explicitly (connections are
    # opened/closed per-request, not held open) ---


app = FastAPI(title="Fraud Detection API", version=MODEL_VERSION, lifespan=lifespan)


class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    type: str = Field(..., description="TRANSFER or CASH_OUT")


class PredictionResponse(BaseModel):
    probability: float
    decision: str
    threshold_used: float
    model_version: str
    explanation: str
    top_factors: dict


@app.get("/")
def root():
    return {
        "service": "fraud-detection-api",
        "version": MODEL_VERSION,
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "model_loaded": state["model"] is not None,
        "threshold_loaded": state["threshold"] is not None,
        "database_connected": state["db_connected"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    if transaction.type not in ("TRANSFER", "CASH_OUT"):
        raise HTTPException(
            status_code=400,
            detail="type must be TRANSFER or CASH_OUT — fraud is not "
                   "possible in other transaction types in this model's "
                   "training scope (see README).",
        )

    row = {
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "is_transfer": 1 if transaction.type == "TRANSFER" else 0,
    }

    feature_columns = state["feature_columns"]
    X = pd.DataFrame([row])[feature_columns]

    proba = float(state["model"].predict_proba(X)[:, 1][0])
    threshold = state["threshold"]
    decision = "block" if proba >= threshold else "allow"

    shap_values = state["explainer"].shap_values(X)[0]
    contributions = dict(zip(feature_columns, [float(v) for v in shap_values]))
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_factors = dict(ranked[:3])

    factor_sentences = []
    for feature, value in ranked[:3]:
        direction = "increased" if value > 0 else "decreased"
        label = FEATURE_LABELS.get(feature, feature)
        factor_sentences.append(f"{label} {direction} fraud probability "
                                 f"(contribution: {value:+.3f})")
    explanation = (f"Predicted fraud probability: {proba:.4f}. "
                    f"Top contributing factors: " + "; ".join(factor_sentences) + ".")

    # Log to Postgres. A logging failure should not fail the prediction
    # response itself - the caller still gets a real answer even if the
    # DB is temporarily unavailable, but state["db_connected"] reflects
    # the outage for /health.
    try:
        conn = get_db_connection()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (amount, old_balance_org, new_balance_orig, is_transfer,
                     predicted_probability, decision, threshold_used,
                     model_version, shap_contributions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    transaction.amount, transaction.oldbalanceOrg,
                    transaction.newbalanceOrig, row["is_transfer"],
                    proba, decision, threshold, MODEL_VERSION,
                    json.dumps(contributions),
                ),
            )
        conn.close()
        state["db_connected"] = True
    except Exception as e:
        print(f"WARNING: failed to log prediction to database: {e}")
        state["db_connected"] = False

    return PredictionResponse(
        probability=proba,
        decision=decision,
        threshold_used=threshold,
        model_version=MODEL_VERSION,
        explanation=explanation,
        top_factors=top_factors,
    )


@app.get("/drift-report")
def get_drift_report(window: int = 1000):
    """
    Compares the most recent `window` logged predictions against the
    training distribution. Returns a status message instead of erroring
    if there isn't enough logged data yet.
    """
    try:
        conn = get_db_connection()
        query = """
            SELECT amount, old_balance_org AS "oldbalanceOrg",
                   new_balance_orig AS "newbalanceOrig", is_transfer
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT %s
        """
        current_df = pd.read_sql(query, conn, params=(window,))
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    if len(current_df) < 30:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(current_df)} logged predictions available; "
                       f"need at least 30 for a meaningful drift comparison.",
        }

    train_df = pd.read_csv(TRAIN_PATH)
    train_df["is_transfer"] = (train_df["type"] == "TRANSFER").astype(int)

    report = drift_report(train_df, current_df, DRIFT_FEATURE_COLUMNS)

    return {
        "status": "ok",
        "window_size": len(current_df),
        "report": report.to_dict(orient="records"),
    }