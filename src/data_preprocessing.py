import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from config.config import Config
from src.logger import logger

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_raw_data(self) -> pd.DataFrame:
        raw_path = Config.DATA_RAW_DIR / "creditcard.csv"
        logger.info(f"Loading raw dataset from {raw_path}")
        df = pd.read_csv(raw_path)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        # Remove duplicates
        df = df.drop_duplicates().copy()
        logger.info(f"Removed {initial_len - len(df)} duplicate records. Remaining: {len(df)}")
        
        # Missing values handling
        if df.isnull().sum().sum() > 0:
            df = df.dropna()
            logger.info("Dropped missing value rows.")
        return df

    def process_and_save(self):
        df = self.load_raw_data()
        df = self.clean_data(df)
        
        # Save cleaned data
        processed_path = Config.DATA_PROCESSED_DIR / "cleaned_creditcard.csv"
        df.to_csv(processed_path, index=False)
        logger.info(f"Saved cleaned dataset to {processed_path}")
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.process_and_save()
    print(f"Cleaned Dataset Shape: {df_clean.shape}")
