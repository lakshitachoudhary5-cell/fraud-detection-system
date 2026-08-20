import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from config.config import Config
from src.logger import logger

def evaluate():
    data_path = Config.DATA_PROCESSED_DIR / "featured_creditcard.csv"
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Class'])
    y_true = df['Class']
    
    logger.info("Loading saved model and scaler...")
    model = joblib.load(Config.MODEL_PATH)
    scaler = joblib.load(Config.SCALER_PATH)
    
    logger.info("Transforming data and predicting...")
    X_scaled = scaler.transform(X)
    
    # Isolation Forest outputs: 1 (normal), -1 (anomaly)
    raw_preds = model.predict(X_scaled)
    # Map to: 0 (normal), 1 (fraud/anomaly)
    y_pred = np.where(raw_preds == -1, 1, 0)
    
    # Calculate raw anomaly score (lower / more negative = higher anomaly)
    scores = model.decision_function(X_scaled)
    
    print("\n================ MODEL EVALUATION REPORT ================")
    print(classification_report(y_true, y_pred, target_names=['Legitimate (0)', 'Fraud (1)']))
    
    cm = confusion_matrix(y_true, y_pred)
    print("--- CONFUSION MATRIX ---")
    print(f"True Negatives (Legit identified as Legit): {cm[0][0]}")
    print(f"False Positives (Legit flagged as Fraud):  {cm[0][1]}")
    print(f"False Negatives (Fraud missed by model):  {cm[1][0]}")
    print(f"True Positives  (Fraud caught by model):  {cm[1][1]}")
    
    roc_auc = roc_auc_score(y_true, -scores)
    print(f"\nROC-AUC Score (Anomaly Score vs True Labels): {roc_auc:.4f}")

if __name__ == "__main__":
    evaluate()
