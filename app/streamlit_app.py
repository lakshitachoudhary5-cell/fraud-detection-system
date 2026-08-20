import streamlit as st
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import FraudPredictor
from app.database import save_prediction, get_recent_predictions

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Real-Time Credit Card Fraud Detection System")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select View", ["Single Prediction", "Transaction History"])

@st.cache_resource
def load_predictor():
    return FraudPredictor()

predictor = load_predictor()

if page == "Single Prediction":
    st.header("Transaction Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Transaction Time (Seconds)", value=100.0)
    with col2:
        amount_val = st.number_input("Transaction Amount ($)", value=250.0)

    pca_inputs = {}
    with st.expander("Advanced Features (PCA V1-V28) - Optional"):
        col_a, col_b = st.columns(2)
        for i in range(1, 29):
            with col_a if i <= 14 else col_b:
                pca_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v_{i}")

    if st.button("Predict Fraud Risk", type="primary"):
        payload = {f"V{i}": 0.0 for i in range(1, 29)}
        payload.update({"Time": time_val, "Amount": amount_val})
        payload.update(pca_inputs)
        
        try:
            result = predictor.predict_single(payload)
            record_id = save_prediction(time_val, amount_val, result)
            result["db_record_id"] = record_id
            
            if result.get("is_fraud"):
                st.error(f"🚨 ALERT: Fraudulent Transaction Detected! (Score: {result.get('anomaly_score', 0):.4f})")
            else:
                st.success(f"✅ Transaction Legitimate (Score: {result.get('anomaly_score', 0):.4f})")
            
            st.write("**Prediction Details:**")
            st.json(result)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

elif page == "Transaction History":
    st.header("Recent Database Logs")
    limit = st.slider("Records to fetch", 5, 50, 10)
    
    if st.button("Refresh History"):
        try:
            records = get_recent_predictions(limit=limit)
            if records:
                df = pd.DataFrame(records)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No records found in database. Make a prediction first!")
        except Exception as e:
            st.error(f"Error fetching history: {e}")
