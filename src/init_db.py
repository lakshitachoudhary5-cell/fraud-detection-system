import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import psycopg2
from config.config import Config
from src.logger import logger

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_time FLOAT NOT NULL,
    amount FLOAT NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    prediction_label VARCHAR(50) NOT NULL,
    anomaly_score FLOAT NOT NULL,
    raw_prediction INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def init_database():
    try:
        logger.info(f"Connecting to PostgreSQL database at {Config.DATABASE_URL}")
        conn = psycopg2.connect(Config.DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLES_SQL)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables created / verified successfully.")
        print("Database Connection & Table Initialization: SUCCESS")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        print(f"Database Connection Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()