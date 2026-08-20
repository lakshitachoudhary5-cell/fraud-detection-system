import os 
from pathlib import Path 
from dotenv import load_dotenv 
 
BASE_DIR = Path(__file__).resolve().parent.parent 
load_dotenv(BASE_DIR / ".env") 
 
class Config: 
    BASE_DIR = BASE_DIR 
    DATA_RAW_DIR = BASE_DIR / "data" / "raw" 
    DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed" 
    MODEL_DIR = BASE_DIR / "models" 
    MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/isolation_forest.joblib") 
    SCALER_PATH = BASE_DIR / os.getenv("SCALER_PATH", "models/scaler.joblib") 
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/fraud_db") 
    FLASK_ENV = os.getenv("FLASK_ENV", "development") 
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000)) 
