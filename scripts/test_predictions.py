# scripts/test_predictions.py

import os
import pandas as pd
import numpy as np
import joblib

from app import db
from app.models import EnergyReading, Prediction
from ml_models.model_selector import load_best_model

from sklearn.preprocessing import MinMaxScaler


def generate_predictions(user_id=1):
    """
    Pull latest readings from DB, load best model, generate predictions,
    and insert them into the predictions table.
    """
    model, model_type = load_best_model()

    # Fetch last 100 readings for this user
    readings = (
        EnergyReading.query.filter_by(user_id=user_id)
        .order_by(EnergyReading.timestamp.desc())
        .limit(100)
        .all()
    )

    if not readings:
        print("No readings found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "device_id": r.device_id,
                "timestamp": r.timestamp,
                "consumption": r.consumption,
            }
            for r in readings
        ]
    ).sort_values("timestamp")

    X = df[["consumption"]]  # features: consumption only (for now)

    # Run prediction based on model type
    if model_type == "xgboost":
        preds = model.predict(X)

    elif model_type == "sarima":
        preds = model.get_forecast(steps=len(X)).predicted_mean.values

    elif model_type == "prophet":
        future = pd.DataFrame({"ds": df["timestamp"]})
        preds = model.predict(future)["yhat"].values

    elif model_type == "lstm":
        # ---- LSTM preprocessing ----
        scaler_path = os.path.join("ml_models", "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            print("⚠️ Warning: No scaler.pkl found, predictions may be inaccurate.")
            scaler = MinMaxScaler()
            scaler.fit(X.values)

        X_scaled = scaler.transform(X.values)

        # Must match training setup
        LOOKBACK = 24  # e.g., 24 hours, change if different
        sequences = []
        for i in range(len(X_scaled) - LOOKBACK):
            seq = X_scaled[i : i + LOOKBACK]
            sequences.append(seq)

        if not sequences:
            print("⚠️ Not enough data for LSTM sequence prediction.")
            return

        X_seq = np.array(sequences).reshape(len(sequences), LOOKBACK, 1)

        preds_scaled = model.predict(X_seq)
        preds = scaler.inverse_transform(preds_scaled).flatten()

    else:
        raise ValueError("Unknown model type")

    # ---- Save into DB ----
    for i, pred in enumerate(preds):
        new_pred = Prediction(
            user_id=user_id,
            device_id=df.iloc[i]["device_id"],
            timestamp=df.iloc[i]["timestamp"],
            predicted_consumption=float(pred),
        )
        db.session.add(new_pred)

    db.session.commit()
    print(f"Inserted {len(preds)} predictions for model={model_type}.")


if __name__ == "__main__":
    # Run a test prediction for user_id=1
    generate_predictions(user_id=1)
