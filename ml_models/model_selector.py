import os
import joblib
from tensorflow import keras

BASE_DIR = os.path.dirname(__file__)

def load_best_model():
    """
    Loads the final best model chosen during training.
    Could be XGBoost (.pkl), SARIMA (.pkl), Prophet (.pkl), or LSTM (.h5).
    """
    # Prefer .h5 if exists (LSTM), otherwise look for .pkl
    h5_path = os.path.join(BASE_DIR, "best_model.h5")
    pkl_path = os.path.join(BASE_DIR, "best_model.pkl")

    if os.path.exists(h5_path):
        print("✅ Loaded LSTM model (best_model.h5)")
        return keras.models.load_model(h5_path), "lstm"
    elif os.path.exists(pkl_path):
        model = joblib.load(pkl_path)
        print(f"✅ Loaded model from {pkl_path}")
        # Heuristic: detect type
        model_type = type(model).__name__.lower()
        if "prophet" in model_type:
            return model, "prophet"
        elif "sarimax" in model_type:
            return model, "sarima"
        elif "multioutputregressor" in model_type or "xgb" in model_type:
            return model, "xgboost"
        else:
            return model, "unknown"
    else:
        raise FileNotFoundError("No trained model found in ml_models/")
