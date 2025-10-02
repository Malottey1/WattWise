import os
import threading
import time
import random
import joblib
import pandas as pd
from datetime import datetime, timedelta

from app import db
from app.models import EnergyReading, Device
from app.utils.costs import compute_cost, emission_from_kwh
from app.utils.devices import random_device_draw, kwh_from_active_power

# Import create_app from the main app package
from app import create_app

# globals to store threads & control flags
_SIM_THREADS = {}
_SIM_FLAGS = {}

# model loader (simple)
def load_model_if_exists():
    """
    Try to load a Keras or joblib model from ml_models/best_model.keras or .pkl.
    Return (model_object, model_type) or (None, None).
    """
    base = os.path.join("ml_models", "best_model.pkl")
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
            return model, "xgboost_or_pickle"
        except Exception:
            pass
    return None, None


def seed_from_ucidata(csv_path, user_id, device_id=None, limit=None):
    """
    Read the UCI household power CSV and insert readings into DB (demo seeding).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    
    # Create app context for database operations
    app = create_app()
    with app.app_context():
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
    """
    # Create application context for this thread
    app = create_app()
    
    with app.app_context():
        model, model_type = load_model_if_exists() if use_model else (None, None)
        
        while _SIM_FLAGS.get((user_id, device_id), False):
            device = Device.query.get(device_id)
            device_type = device.device_type if device else "generic"
            
            # Generate energy reading (simplified)
            if device_type == "Air Conditioner":
                draw_kw = random.uniform(1.5, 3.0)
            elif device_type == "Fridge":
                draw_kw = random.uniform(0.1, 0.3)
            elif device_type == "Solar Panel":
                draw_kw = -random.uniform(0.5, 3.0)  # Negative for solar production
            else:
                draw_kw = random.uniform(0.1, 2.0)
            
            reading = EnergyReading(
                user_id=user_id,
                device_id=device_id,
                timestamp=datetime.utcnow(),
                energy_consumed=round(draw_kw, 4),
                voltage=random.uniform(220, 240),
                global_intensity=None,
                sub_metering_1=None,
                sub_metering_2=None,
                sub_metering_3=None
            )
            try:
                db.session.add(reading)
                db.session.commit()
                print(f"Simulated reading for device {device_id}: {draw_kw} kW")
            except Exception as e:
                print(f"Error saving reading: {e}")
                db.session.rollback()

            # sleep until next tick but check flag frequently
            for _ in range(int(interval_seconds)):
                if not _SIM_FLAGS.get((user_id, device_id), False):
                    break
                time.sleep(1)
        
        # thread exit
        _SIM_THREADS.pop((user_id, device_id), None)
        _SIM_FLAGS.pop((user_id, device_id), None)
        print(f"Simulation stopped for device {device_id}")


def start_simulation(user_id, device_id, interval_seconds=10, tariff=None, use_model=True):
    key = (user_id, device_id)
    if _SIM_FLAGS.get(key):
        print(f"Simulation already running for device {device_id}")
        return False  # already running
    
    _SIM_FLAGS[key] = True
    th = threading.Thread(
        target=_sim_loop, 
        args=(user_id, device_id, interval_seconds, tariff, use_model), 
        daemon=True
    )
    _SIM_THREADS[key] = th
    th.start()
    print(f"Started simulation for device {device_id}")
    return True


def stop_simulation(user_id, device_id):
    key = (user_id, device_id)
    if key in _SIM_FLAGS:
        _SIM_FLAGS[key] = False
        print(f"Stopped simulation for device {device_id}")
        return True
    else:
        print(f"No simulation found for device {device_id}")
        return False


def list_simulations():
    return list(_SIM_FLAGS.keys())