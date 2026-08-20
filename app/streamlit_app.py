import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Real-Time Credit Card Fraud Detection System")
st.markdown("Enter transaction details below to evaluate anomaly and fraud risk scores.")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select View", ["Single Prediction", "Transaction History"])

if page == "Single Prediction":
    st.subheader("Transaction Analysis")

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Transaction Time (Seconds)", min_value=0.0, value=100.0, step=1.0)
    with col2:
        amount_val = st.number_input("Transaction Amount ($)", min_value=0.0, value=250.0, step=0.5)

    with st.expander("Advanced Features (PCA V1-V28 Values) - Optional"):
        st.caption("Default value for PCA features is 0.0 unless modified.")
        pca_inputs = {}
        cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with cols[col_idx]:
                pca_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v{i}")

    if st.button("Predict Fraud Risk", type="primary"):
        payload = {"Time": time_val, "Amount": amount_val}
        payload.update(pca_inputs)

        try:
            res = requests.post(f"{API_URL}/predict", json=payload)
            if res.status_code == 200:
                data = res.json()
                
                st.divider()
                st.subheader("Prediction Result")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    if data["is_fraud"]:
                        st.error("🚨 ALERT: Fraudulent / Suspicious")
                    else:
                        st.success("✅ LEGITIMATE")
                
                with res_col2:
                    st.metric(label="Anomaly Score", value=data["anomaly_score"])
                
                with res_col3:
                    st.metric(label="DB Record ID", value=data.get("db_record_id", "N/A"))

            else:
                st.error(f"API Error: {res.json().get('error', 'Unknown Error')}")
        except Exception as e:
            st.error(f"Could not connect to Flask API backend. Ensure API is running. Error: {e}")

elif page == "Transaction History":
    st.subheader("Recent Database Logs")
    limit = st.slider("Records to fetch", min_value=5, max_value=50, value=10)

    if st.button("Refresh History"):
        try:
            res = requests.get(f"{API_URL}/history?limit={limit}")
            if res.status_code == 200:
                records = res.json().get("predictions", [])
                if records:
                    df_hist = pd.DataFrame(records)
                    st.dataframe(df_hist, use_container_width=True)
                else:
                    st.info("No records found in database.")
            else:
                st.error("Failed to retrieve history from backend API.")
        except Exception as e:
            st.error(f"Connection error: {e}")