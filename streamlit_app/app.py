"""
streamlit_app/app.py

Dashboard frontend for the fraud detection API. Calls the FastAPI backend
over HTTP for everything - never loads the model, SHAP explainer, or any
ML library directly. This is the same architectural separation already
used in the two-tower recommender project: the frontend is a thin client,
the backend owns all inference logic.

Backend URL is configurable in the UI (not hardcoded) so the same app
works unchanged against local uvicorn, Docker, or the deployed Render URL.
"""

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "https://fraud-detection-cost-optimal.onrender.com"

with st.sidebar:
    st.header("Backend")
    st.session_state.backend_url = st.text_input(
        "API base URL",
        value=st.session_state.backend_url,
        help="Local: http://localhost:8000 · Docker: http://localhost:8000 "
             "(compose maps the port) · Deployed: your Render URL",
    )
    if st.button("Check connection"):
        try:
            resp = requests.get(f"{st.session_state.backend_url}/health", timeout=5)
            resp.raise_for_status()
            health = resp.json()
            if all(health.values()):
                st.success("Connected — model, threshold, and database all healthy.")
            else:
                st.warning(f"Connected, but not fully healthy: {health}")
        except Exception as e:
            st.error(f"Could not reach backend: {e}")

tab_predict, tab_drift, tab_about = st.tabs(
    ["Try a Prediction", "Drift Monitor", "About"]
)

# --- Tab 1: Try a Prediction ---
with tab_predict:
    st.subheader("Test a transaction")
    st.caption("Calls POST /predict on the live backend. The model, "
               "cost-optimal threshold, and SHAP explainer all run "
               "server-side.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount", min_value=0.01, value=181.0, step=1.0)
        old_balance = st.number_input("Sender's balance BEFORE transaction",
                                       min_value=0.0, value=181.0, step=1.0)
    with col2:
        new_balance = st.number_input("Sender's balance AFTER transaction",
                                       min_value=0.0, value=0.0, step=1.0)
        txn_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])

    if st.button("Run prediction", type="primary"):
        payload = {
            "amount": amount,
            "oldbalanceOrg": old_balance,
            "newbalanceOrig": new_balance,
            "type": txn_type,
        }
        try:
            resp = requests.post(
                f"{st.session_state.backend_url}/predict", json=payload, timeout=10
            )
            resp.raise_for_status()
            result = resp.json()

            decision = result["decision"]
            probability = result["probability"]

            if decision == "block":
                st.error(f"**BLOCKED** — fraud probability {probability:.2%} "
                         f"(threshold: {result['threshold_used']:.2f})")
            else:
                st.success(f"**ALLOWED** — fraud probability {probability:.2%} "
                           f"(threshold: {result['threshold_used']:.2f})")

            st.write(result["explanation"])

            factors = pd.Series(result["top_factors"])
            factors = factors.reindex(factors.abs().sort_values().index)
            st.bar_chart(factors, horizontal=True)

            st.caption(f"Model version: {result['model_version']}")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# --- Tab 2: Drift Monitor ---
with tab_drift:
    st.subheader("Feature drift vs. training distribution")
    st.caption("Calls GET /drift-report — compares the most recent logged "
               "predictions against the training reference sample using "
               "PSI and the KS statistic.")

    window = st.slider("Number of recent predictions to compare", 30, 5000, 1000)

    if st.button("Run drift report"):
        try:
            resp = requests.get(
                f"{st.session_state.backend_url}/drift-report",
                params={"window": window},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()

            if result["status"] == "insufficient_data":
                st.info(result["message"])
            else:
                st.write(f"Window size: {result['window_size']} logged predictions")
                report_df = pd.DataFrame(result["report"])
                st.dataframe(report_df, use_container_width=True)

                st.caption(
                    "PSI < 0.10: no significant shift · 0.10-0.25: moderate, "
                    "worth investigating · > 0.25: significant, retraining "
                    "likely warranted. KS p-values are sensitive to sample "
                    "size and can look 'significant' even for small, "
                    "practically meaningless shifts at scale — PSI and the "
                    "KS statistic (not its p-value) are the more reliable "
                    "signals here."
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# --- Tab 3: About ---
with tab_about:
    st.subheader("Fraud Detection with Cost-Optimal Decisioning")
    st.markdown("""
This project goes beyond a standard fraud classifier by deriving the
block/allow decision threshold from business economics rather than
defaulting to 0.5, and by pairing every prediction with a SHAP-based
explanation.

**Key results (measured on an untouched test set, not the data the
threshold was chosen on):**
- Cost-optimal threshold (0.11) reduces total expected business cost by
  **36.2%** versus a naive 0.5 threshold.
- Destination-side balance features were identified and excluded as
  target leakage — confirmed both empirically and via PaySim's own
  documentation — before any model was trained on them.
- Drift monitoring (PSI/KS) is built in and reused unchanged between
  offline analysis and the live `/drift-report` endpoint.

Built on PaySim (synthetic financial transaction data), scoped to
TRANSFER and CASH_OUT transactions — the only two types where fraud
occurs in this dataset.

See the full writeup, including the leakage investigation and threshold
derivation, in the project README.
""")