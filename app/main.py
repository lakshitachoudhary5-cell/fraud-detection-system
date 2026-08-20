import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from flask import Flask, request, jsonify
from src.predict import FraudPredictor
from app.database import save_prediction, get_recent_predictions
from src.logger import logger

app = Flask(__name__)

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = FraudPredictor()
    return predictor

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Fraud Detection API",
        "model_loaded": True
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    if 'Time' not in data or 'Amount' not in data:
        return jsonify({"error": "Missing required fields: 'Time' and 'Amount'"}), 400

    try:
        payload = {f"V{i}": 0.0 for i in range(1, 29)}
        payload.update(data)

        model = get_predictor()
        result = model.predict_single(payload)

        record_id = save_prediction(
            transaction_time=float(payload['Time']),
            amount=float(payload['Amount']),
            prediction=result
        )
        result['db_record_id'] = record_id

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    limit = request.args.get('limit', default=10, type=int)
    records = get_recent_predictions(limit=limit)
    return jsonify({
        "total": len(records),
        "predictions": records
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
