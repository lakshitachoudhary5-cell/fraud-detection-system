import sys
from pathlib import Path

# Add project root to python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from config.config import Config
from src.logger import logger

raw_path = Config.DATA_RAW_DIR / "creditcard.csv"
if not raw_path.exists():
    logger.error(f"Dataset NOT found at {raw_path}")
    sys.exit(1)

logger.info("Loading dataset for inspection...")
df = pd.read_csv(raw_path)
print("\n================ DATASET OVERVIEW ================")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\n--- Missing Values ---")
print(f"Total Nulls: {df.isnull().sum().sum()}")
print("\n--- Duplicate Rows ---")
print(f"Duplicates: {df.duplicated().sum()}")
print("\n--- Target Class Distribution ('Class') ---")
print(df['Class'].value_counts())
print(f"Fraud Percentage: {(df['Class'].sum() / len(df)) * 100:.3f}%")
print("\n--- First 3 Rows ---")
print(df.head(3))
