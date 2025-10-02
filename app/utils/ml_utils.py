import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from dotenv import load_dotenv

keras = tf.keras
load_dotenv()

MODEL_DIR = os.getenv('MODEL_DIR', './ml_models')
MODEL_FILE = os.getenv('MODEL_FILE', 'best_model.pkl')
SCALER_FILE = os.getenv('SCALER_FILE', 'scaler.pkl')
H = int(os.getenv('HORIZON', '24'))
INPUT_SEQ_LEN = int(os.getenv('INPUT_SEQ_LEN', '72'))

TARGET = "Global_active_power"

FEATURES = [
    'Global_active_power','Global_reactive_power','Voltage',
    'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'
]

# ---------------- Loaders ----------------
def load_model_and_scaler():
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)

    scaler = joblib.load(scaler_path)

    if model_path.endswith((".keras", ".h5")):
        model = keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)

    return model, scaler

# ---------------- Data Fetch ----------------
def fetch_recent_dataframe(session, hours=INPUT_SEQ_LEN+24):
    query = f"""
    SELECT timestamp, global_active_power, global_reactive_power, voltage,
           global_intensity, sub_metering_1, sub_metering_2, sub_metering_3
    FROM energy_readings
    WHERE timestamp IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT {hours}
    """
    df = pd.read_sql(query, session)
    df = df.rename(columns={
        'global_active_power': 'Global_active_power',
        'global_reactive_power': 'Global_reactive_power',
        'voltage': 'Voltage',
        'global_intensity': 'Global_intensity',
        'sub_metering_1': 'Sub_metering_1',
        'sub_metering_2': 'Sub_metering_2',
        'sub_metering_3': 'Sub_metering_3',
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # resample hourly
    df = df.resample('h').mean().ffill().bfill()
    df = df.iloc[-hours:]
    return df

# ---------------- LSTM Preprocessing ----------------
def prepare_input(df, scaler, input_seq_len=INPUT_SEQ_LEN):
    cols = [c for c in FEATURES if c in df.columns]
    arr = df[cols].values
    scaled = scaler.transform(arr)

    if scaled.shape[0] < input_seq_len:
        pad_rows = input_seq_len - scaled.shape[0]
        pad = np.repeat(scaled[0:1, :], pad_rows, axis=0)
        scaled = np.vstack([pad, scaled])

    X = scaled[-input_seq_len:, :]
    return X.reshape((1, X.shape[0], X.shape[1]))

def postprocess_predictions(preds_scaled, scaler, target_col_index=0):
    preds = np.array(preds_scaled).squeeze()
    if preds.ndim == 2:
        preds = preds[0]

    inv = []
    n_feats = scaler.n_features_in_
    for v in preds:
        dummy = np.zeros((1, n_feats))
        dummy[0, target_col_index] = v
        orig = scaler.inverse_transform(dummy)
        inv.append(orig[0, target_col_index])
    return np.array(inv)

# ---------------- XGBoost Preprocessing ----------------
def prepare_xgb_features(df):
    X = pd.DataFrame(index=df.index)
    X['target'] = df[TARGET]
    X['hour'] = df.index.hour
    X['dow'] = df.index.dayofweek
    X['is_weekend'] = (X['dow'] >= 5).astype(int)

    lags = [1,2,3,6,12,24,48,72,168]
    for lag in lags:
        X[f'lag_{lag}'] = X['target'].shift(lag)

    for w in [3,6,24]:
        X[f'roll_mean_{w}'] = X['target'].shift(1).rolling(window=w, min_periods=1).mean()
        X[f'roll_std_{w}'] = X['target'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)

    for col in FEATURES[1:]:  # skip target
        if col in df.columns:
            X[col] = df[col]
            X[f'{col}_lag1'] = df[col].shift(1)

    X = X.drop(columns=['target']).fillna(method="ffill").fillna(method="bfill")
    return X

# ---------------- Unified Prediction ----------------
def predict_from_db(engine_or_conn, model, scaler, horizon=H):
    df = fetch_recent_dataframe(engine_or_conn, hours=500)

    model_name = str(type(model)).lower()
    if "xgboost" in model_name or "sklearn" in model_name:
        X = prepare_xgb_features(df)
        X_input = X.iloc[[-1]]
        preds = model.predict(X_input)
        preds_inv = preds.flatten()
    else:
        X = prepare_input(df, scaler, input_seq_len=INPUT_SEQ_LEN)
        preds = model.predict(X)
        preds_inv = postprocess_predictions(preds, scaler)

    last_ts = df.index[-1]
    timestamps = [last_ts + pd.Timedelta(hours=i+1) for i in range(horizon)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "predicted": preds_inv[:horizon]
    })
