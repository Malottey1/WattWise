# app/simulator.py
import os
import threading
import time
import random
import joblib
import pandas as pd
from datetime import datetime, timedelta

from app import db
from app.models import EnergyReading, Device
from app.utils import compute_cost, emission_from_kwh, random_device_draw, kwh_from_active_power

# globals to store threads & control flags
_SIM_THREADS = {}
_SIM_FLAGS = {}

# model loader (simple)
def load_model_if_exists():
    """
    Try to load a Keras or joblib model from ml_models/best_model.keras or .pkl.
    Return (model_object, model_type) or (None, None).
    """
    base = os.path.join("ml_models", "best_model.keras")
    if os.path.exists(base):
        try:
            from tensorflow import keras
            model = keras.models.load_model(base)
            return model, "lstm_keras"
        except Exception:
            pass
    pkl = os.path.join("ml_models", "best_model.pkl")
    if os.path.exists(pkl):
        try:
            model = joblib.load(pkl)
            # we can't always tell; return generic object
            return model, "xgboost_or_pickle"
        except Exception:
            pass
    return None, None


def seed_from_ucidata(csv_path, user_id, device_id=None, limit=None):
    """
    Read the UCI household power CSV and insert readings into DB (demo seeding).
    csv_path: Path to UCI CSV (semicolon separated)
    device_id: optional device id to attach
    limit: max rows to import
    Returns number inserted.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, sep=';', decimal='.', header=0, na_values=['?'])
    # ensure columns exist and parse date/time
    df = df.dropna(subset=['Date','Time'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    # numeric conversion
    cols = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
            'Sub_metering_1','Sub_metering_2','Sub_metering_3']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=cols + ['datetime'])
    if limit:
        df = df.iloc[:limit]
    count = 0
    for _, row in df.iterrows():
        er = EnergyReading(
            user_id=user_id,
            device_id=device_id,
            timestamp=row['datetime'],
            energy_consumed=row['Global_active_power'],
            voltage=row.get('Voltage'),
            global_intensity=row.get('Global_intensity'),
            sub_metering_1=row.get('Sub_metering_1'),
            sub_metering_2=row.get('Sub_metering_2'),
            sub_metering_3=row.get('Sub_metering_3')
        )
        db.session.add(er)
        count += 1
        if count % 200 == 0:
            db.session.commit()
    db.session.commit()
    return count


def _sim_loop(user_id, device_id, interval_seconds=10, tariff=None, use_model=True):
    """
    Background loop that writes new EnergyReading rows every interval_seconds.
    If use_model is True and a model is available, use it to generate baseline predictions,
    else produce draws using device_type heuristic.
    Tariff is passed to compute_cost when storing metrics (not stored here).
    """
    model, model_type = load_model_if_exists() if use_model else (None, None)
    # For simplicity, model usage is naive (e.g., for LSTM you'd need sequences) — we fallback to heuristics if model cannot be applied.
    # Keep previous reading for small random walk
    last_ts = datetime.now()
    while _SIM_FLAGS.get((user_id, device_id), False):
        device = Device.query.get(device_id)
        device_type = device.device_type if device else "generic"
        # base draw from model if available — naive approach:
        draw_kw = None
        try:
            if model is not None and model_type == "xgboost_or_pickle":
                # try to call model.predict on a 1-row feature vector if possible
                # fallback: random device draw
                try:
                    # pick last reading for this device to form features
                    last_read = EnergyReading.query.filter_by(user_id=user_id, device_id=device_id).order_by(EnergyReading.timestamp.desc()).first()
                    if last_read:
                        feat = [[last_read.energy_consumed]]
                        pred = model.predict(feat)
                        if hasattr(pred, "__len__"):
                            draw_kw = float(pred[0])
                        else:
                            draw_kw = float(pred)
                except Exception:
                    draw_kw = None
            elif model is not None and model_type == "lstm_keras":
                # LSTM needs sequence; skip for now — fallback
                draw_kw = None
        except Exception:
            draw_kw = None

        if draw_kw is None:
            # heuristic draw per device-type
            draw_kw = random_device_draw(device_type)

        # add small random jitter
        noise = random.uniform(-0.1, 0.1) * abs(draw_kw)
        draw_kw_noisy = max(-5.0, draw_kw + noise)  # clamp
        # express as kWh for the interval (kW * minutes/60). We assume interval_seconds covers roughly that period.
        kwh = kwh_from_active_power(draw_kw_noisy, minutes=(interval_seconds/60.0))
        reading = EnergyReading(
            user_id=user_id,
            device_id=device_id,
            timestamp=datetime.utcnow(),
            energy_consumed=round(draw_kw_noisy, 4),
            voltage=None,
            global_intensity=None,
            sub_metering_1=None,
            sub_metering_2=None,
            sub_metering_3=None
        )
        try:
            db.session.add(reading)
            db.session.commit()
        except Exception:
            db.session.rollback()

        # sleep until next tick but check flag frequently
        for _ in range(int(interval_seconds)):
            if not _SIM_FLAGS.get((user_id, device_id), False):
                break
            time.sleep(1)
    # thread exit
    _SIM_THREADS.pop((user_id, device_id), None)
    _SIM_FLAGS.pop((user_id, device_id), None)


def start_simulation(user_id, device_id, interval_seconds=10, tariff=None, use_model=True):
    key = (user_id, device_id)
    if _SIM_FLAGS.get(key):
        return False  # already running
    _SIM_FLAGS[key] = True
    th = threading.Thread(target=_sim_loop, args=(user_id, device_id, interval_seconds, tariff, use_model), daemon=True)
    _SIM_THREADS[key] = th
    th.start()
    return True


def stop_simulation(user_id, device_id):
    key = (user_id, device_id)
    _SIM_FLAGS[key] = False
    # thread will clean up itself
    return True


def list_simulations():
    return list(_SIM_FLAGS.keys())
