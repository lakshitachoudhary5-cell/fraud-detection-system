import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "fraud_detection.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            transaction_time REAL,
            amount REAL,
            anomaly_score REAL,
            is_fraud BOOLEAN,
            prediction_label TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(transaction_time, amount, prediction):
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (transaction_time, amount, anomaly_score, is_fraud, prediction_label)
            VALUES (?, ?, ?, ?, ?)
        """, (
            transaction_time,
            amount,
            float(prediction.get("anomaly_score", 0.0)),
            bool(prediction.get("is_fraud", False)),
            str(prediction.get("prediction_label", "Unknown"))
        ))
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id
    except Exception as e:
        print(f"DB Insert Error: {e}")
        return "N/A"

def get_recent_predictions(limit=10):
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, transaction_time, amount, anomaly_score, is_fraud, prediction_label
            FROM predictions ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for r in rows:
            records.append({
                "id": r[0],
                "timestamp": r[1],
                "transaction_time": r[2],
                "amount": r[3],
                "anomaly_score": r[4],
                "is_fraud": bool(r[5]),
                "prediction_label": r[6]
            })
        return records
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        return []
