import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from config.config import Config
from src.logger import logger

def run_eda():
    processed_path = Config.DATA_PROCESSED_DIR / "cleaned_creditcard.csv"
    if not processed_path.exists():
        logger.error(f"Processed dataset not found at {processed_path}")
        return
    
    logger.info("Starting Exploratory Data Analysis...")
    df = pd.read_csv(processed_path)
    
    # 1. Class Distribution
    fraud_count = df['Class'].sum()
    legit_count = len(df) - fraud_count
    logger.info(f"Legitimate Transactions: {legit_count} ({legit_count/len(df)*100:.2f}%)")
    logger.info(f"Fraudulent Transactions: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    
    # 2. Transaction Amount Analysis
    print("\n--- TRANSACTION AMOUNT STATS ---")
    print("Legitimate:")
    print(df[df['Class'] == 0]['Amount'].describe())
    print("\nFraudulent:")
    print(df[df['Class'] == 1]['Amount'].describe())
    
    # 3. Top Correlated Features with Fraud
    correlations = df.corr()['Class'].sort_values()
    print("\n--- TOP 5 NEGATIVELY CORRELATED FEATURES ---")
    print(correlations.head(5))
    print("\n--- TOP 5 POSITIVELY CORRELATED FEATURES ---")
    print(correlations.tail(6).iloc[:-1])

if __name__ == "__main__":
    run_eda()
