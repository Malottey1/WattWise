# app/utils/forecasting.py

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

MODEL_PATH = "app/ml_models/model.pkl"
SCALER_PATH = "app/ml_models/scaler.pkl"
LOCAL_RATE = 0.15  # $ per kWh

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

def predict_next_48h(latest_data):
    """Generate 48h forecast and cost estimates."""
    future_times = [datetime.now() + timedelta(hours=i) for i in range(1, 49)]
    # X_test would be constructed using your actual preprocessing
    X_test = scaler.transform(np.array(latest_data[-48:]).reshape(-1, 1))
    preds = model.predict(X_test)

    df = pd.DataFrame({
        "time": future_times,
        "pred_kwh": preds,
        "expected_cost": preds * LOCAL_RATE
    })
    return df
