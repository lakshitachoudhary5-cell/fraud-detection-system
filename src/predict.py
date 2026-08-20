import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import joblib
from config.config import Config
from src.logger import logger
from src.feature_engineering import FeatureEngineer

class FraudPredictor:
    def __init__(self):
        if not Config.MODEL_PATH.exists() or not Config.SCALER_PATH.exists():
            raise FileNotFoundError("Model or Scaler binary missing. Train model first.")
        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)
        self.feature_engineer = FeatureEngineer()

    def predict_single(self, transaction_dict: dict) -> dict:
        # Convert input dictionary to DataFrame
        df = pd.DataFrame([transaction_dict])
        
        # Apply exact feature engineering
        df_feat = self.feature_engineer.transform(df)
        
        # Drop Class label if present in payload
        if 'Class' in df_feat.columns:
            df_feat = df_feat.drop(columns=['Class'])
            
        # Reorder features to match exact order learned during scaler fitting
        expected_features = list(self.scaler.feature_names_in_)
        df_feat = df_feat[expected_features]
            
        # Scale features
        scaled_features = self.scaler.transform(df_feat)
        
        # ML Model Inference
        raw_pred = self.model.predict(scaled_features)[0]
        anomaly_score = float(self.model.decision_function(scaled_features)[0])
        
        is_fraud = True if raw_pred == -1 else False
        prediction_label = "Fraudulent / Suspicious" if is_fraud else "Legitimate"
        
        return {
            "is_fraud": is_fraud,
            "prediction_label": prediction_label,
            "anomaly_score": round(anomaly_score, 5),
            "raw_prediction": int(raw_pred)
        }

if __name__ == "__main__":
    predictor = FraudPredictor()
    # Sample payload in expected field structure
    sample_data = {"Time": 100.0, "Amount": 500.0}
    for i in range(1, 29):
        sample_data[f"V{i}"] = 0.0
        
    result = predictor.predict_single(sample_data)
    print("\n--- PREDICTION RESULT ---")
    print(result)
