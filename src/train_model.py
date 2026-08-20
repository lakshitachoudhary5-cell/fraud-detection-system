import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from config.config import Config
from src.logger import logger

class ModelTrainer:
    def __init__(self, contamination: float = 0.0017, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

    def train_and_save(self):
        data_path = Config.DATA_PROCESSED_DIR / "featured_creditcard.csv"
        logger.info(f"Loading training data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        logger.info("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info("Training Isolation Forest model...")
        self.model.fit(X_scaled)
        
        # Save artifacts
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, Config.MODEL_PATH)
        joblib.dump(self.scaler, Config.SCALER_PATH)
        logger.info(f"Model saved to {Config.MODEL_PATH}")
        logger.info(f"Scaler saved to {Config.SCALER_PATH}")
        
        return self.model, self.scaler

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_save()
