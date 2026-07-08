from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# ── Load saved model and scalers ─────────────────────────────────────────────
model    = None
scaler_X = None
scaler_y = None
MODEL_MODE = "demo"

def build_model_architecture():
    """Rebuild exact model architecture from Colab — avoids version conflicts."""
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(24, 11)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear'),
    ])
    m.compile(loss='mse')
    return m

def load_models():
    global model, scaler_X, scaler_y, MODEL_MODE

    weights_path  = 'lstm_weights.weights.h5'
    scaler_x_path = 'scaler_X.pkl'
    scaler_y_path = 'scaler_y.pkl'

    missing = [f for f in [weights_path, scaler_x_path, scaler_y_path] if not os.path.exists(f)]
    if missing:
        print(f"[INFO] Missing files: {missing} — running in DEMO mode")
        return

    try:
        import joblib
        model = build_model_architecture()
        model.load_weights(weights_path)
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        MODEL_MODE = "real"
        print("[SUCCESS] Real LSTM model loaded — predictions will be REAL ✅")

    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")
        print("[INFO] Running in DEMO mode")

load_models()


# ── Helper: build features for one timestamp ──────────────────────────────────
def build_features(junction, dt, lag1, lag24, lag168, roll3, roll24):
    return np.array([
        dt.hour,
        dt.day,
        dt.month,
        dt.weekday(),
        1 if dt.weekday() >= 5 else 0,
        dt.isocalendar()[1],
        lag1, lag24, lag168, roll3, roll24
    ]).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({'mode': MODEL_MODE})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        junction = int(data['junction'])
        dt_str   = data['datetime']
        lag1     = float(data.get('lag1',  20))
        lag24    = float(data.get('lag24', 18))
        lag168   = float(data.get('lag168',17))
        roll3    = float(data.get('roll3', 19))
        roll24   = float(data.get('roll24',18))
        hours    = int(data.get('hours',   12))

        start_dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M')

        if MODEL_MODE == "real":
            predictions = real_predictions(junction, start_dt, lag1, lag24, lag168, roll3, roll24, hours)
        else:
            predictions = simulate_predictions(junction, start_dt, hours)

        return jsonify({'success': True, 'predictions': predictions, 'mode': MODEL_MODE})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def real_predictions(junction, start_dt, lag1, lag24, lag168, roll3, roll24, hours=12):
    """Run LSTM model for each of the next hours."""
    preds = []
    current_lag1 = lag1

    for i in range(hours):
        dt = start_dt + timedelta(hours=i)
        X = build_features(junction, dt, current_lag1, lag24, lag168, roll3, roll24)
        X_sc  = scaler_X.transform(X)
        # Model expects (batch, timesteps=24, features=11) — repeat single row 24 times
        X_seq = np.repeat(X_sc, 24, axis=0).reshape(1, 24, 11)
        pred_sc = model.predict(X_seq, verbose=0)
        pred    = float(scaler_y.inverse_transform(pred_sc)[0][0])
        pred    = max(0, round(pred))
        preds.append(pred)
        current_lag1 = pred

    return preds


def simulate_predictions(junction, start_dt, hours=12):
    base_map = {1: 45, 2: 14, 3: 13, 4: 7}
    base = base_map.get(junction, 15)
    preds = []
    for i in range(hours):
        hour = (start_dt.hour + i) % 24
        if 7 <= hour <= 9:
            v = base + np.random.randint(10, 20)
        elif 17 <= hour <= 19:
            v = base + np.random.randint(12, 22)
        elif 0 <= hour <= 5:
            v = max(2, base - np.random.randint(6, 10))
        else:
            v = base + np.random.randint(-3, 8)
        preds.append(max(0, int(v)))
    return preds


if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"  Traffic Forecaster — Mode: {MODEL_MODE.upper()}")
    print(f"{'='*50}\n")
    app.run(debug=True)
