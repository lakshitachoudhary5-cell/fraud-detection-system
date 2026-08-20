import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from config.config import Config
from src.logger import logger

class FeatureEngineer:
    def __init__(self):
        pass
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Applying feature engineering transformations...")
        
        # 1. Log Amount Transformation
        df['log_amount'] = np.log1p(df['Amount'])
        
        # 2. Extract Hour of Day from Time (seconds)
        df['hour_of_day'] = (df['Time'] // 3600) % 24
        
        # 3. Transaction Amount to Mean Ratio
        mean_amt = df['Amount'].mean()
        df['amount_to_mean_ratio'] = df['Amount'] / (mean_amt + 1e-5)
        
        logger.info(f"Engineered 3 new features. Total columns: {df.shape[1]}")
        return df

    def process_and_save(self):
        input_path = Config.DATA_PROCESSED_DIR / "cleaned_creditcard.csv"
        output_path = Config.DATA_PROCESSED_DIR / "featured_creditcard.csv"
        
        df = pd.read_csv(input_path)
        df_featured = self.transform(df)
        df_featured.to_csv(output_path, index=False)
        logger.info(f"Saved feature-engineered data to {output_path}")
        return df_featured

if __name__ == "__main__":
    fe = FeatureEngineer()
    df_res = fe.process_and_save()
    print(f"New Dataset Shape: {df_res.shape}")
    print("New features sample:")
    print(df_res[['Amount', 'log_amount', 'hour_of_day', 'amount_to_mean_ratio']].head(3))
