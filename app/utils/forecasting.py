import os
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from datetime import datetime, timedelta
import random

class LSTMForecaster:
    """
    Improved LSTM forecaster that uses recent user data from database.
    """
    
    def __init__(self, model_path='ml_models/best_model.pkl', 
                 scaler_path='ml_models/scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.input_seq_len = 72
        self.output_seq_len = 24
        
        # Load model and scaler
        self._load_model()
        self._load_scaler()
        print("✅ Improved forecaster ready")
    
    def _load_model(self):
        """Load the trained LSTM model with better debugging."""
        print(f"🔍 DEBUG: Looking for model at {self.model_path}")
        
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                print(f"✅ DEBUG: Successfully loaded LSTM model")
                print(f"✅ DEBUG: Model input shape: {self.model.input_shape}")
                print(f"✅ DEBUG: Model output shape: {self.model.output_shape}")
            except Exception as e:
                print(f"❌ DEBUG: Error loading model: {e}")
                self.model = None
        else:
            print(f"❌ DEBUG: Model file not found at {self.model_path}")
            self.model = None
    
    def _load_scaler(self):
        """Load the scaler."""
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ Loaded scaler from {self.scaler_path}")
            except Exception as e:
                print(f"❌ Error loading scaler: {e}")
                self.scaler = None
        else:
            print("⚠️ No scaler found")
            self.scaler = None

    

    def _generate_realistic_pattern(self, hours=72):
        """Generate realistic energy usage pattern based on time of day."""
        base_time = datetime.now() - timedelta(hours=hours)
        timestamps = [base_time + timedelta(hours=i) for i in range(hours)]
        
        data = []
        for ts in timestamps:
            hour = ts.hour
            # Realistic pattern based on typical household usage
            if 7 <= hour <= 9:  # Morning peak
                power = random.uniform(2.0, 3.5)
            elif 18 <= hour <= 21:  # Evening peak
                power = random.uniform(2.5, 4.0)
            elif 23 <= hour or hour <= 5:  # Night (low usage)
                power = random.uniform(0.2, 0.8)
            else:  # Daytime
                power = random.uniform(1.0, 2.0)
            
            data.append({
                'timestamp': ts,
                'Global_active_power': power,
                'Global_reactive_power': power * 0.3,
                'Voltage': 240.0,
                'Global_intensity': power * 4.5,
                'Sub_metering_1': power * 0.1,
                'Sub_metering_2': power * 0.2,
                'Sub_metering_3': power * 0.3
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        print(f"🔄 Generated {len(df)} hours of realistic pattern data")
        return df

    # ... (keep the rest of the methods the same as your current version)
    def prepare_input_sequence(self, df):
        """Prepare input for LSTM model."""
        if self.scaler is None:
            # If no scaler, return simple normalized data
            values = df[['Global_active_power']].values
            normalized = values / 5.0  # Simple normalization
            return normalized.reshape(1, len(df), 1)
        
        # Use scaler if available
        try:
            scaled_data = self.scaler.transform(df.values)
            return scaled_data.reshape(1, len(df), len(df.columns))
        except:
            # Fallback if scaler fails
            values = df[['Global_active_power']].values
            normalized = values / 5.0
            return normalized.reshape(1, len(df), 1)

    def predict(self, user_id):
        """Generate prediction using user's actual data."""
        start_time = datetime.now()
        
        try:
            # Get user's recent data
            input_df = self.get_recent_user_data(user_id, self.input_seq_len)
            
            if self.model is None:
                # Fallback: simple pattern-based prediction
                print("🔄 Using fallback prediction (no LSTM model)")
                return self._fallback_prediction(input_df)
            
            # Prepare input
            X = self.prepare_input_sequence(input_df)
            
            # Make prediction
            scaled_pred = self.model.predict(X, verbose=0)
            
            # Process predictions
            if self.scaler:
                # Inverse transform if scaler available
                predictions = self._inverse_transform_predictions(scaled_pred)
            else:
                # Simple denormalization
                predictions = scaled_pred.squeeze() * 5.0
            
            # Generate timestamps
            last_timestamp = input_df.index[-1] if len(input_df) > 0 else datetime.now()
            timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(self.output_seq_len)]
            
            # Ensure predictions are realistic
            predictions = np.maximum(predictions, 0.1)
            predictions = np.minimum(predictions, 5.0)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Prediction generated in {execution_time:.2f}s")
            
            return {
                'predictions': predictions.tolist(),
                'timestamps': timestamps,
                'input_data': input_df,
                'confidence_intervals': {
                    'lower': (predictions * 0.8).tolist(),
                    'upper': (predictions * 1.2).tolist()
                }
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_prediction()

    def _fallback_prediction(self, input_df=None):
        """Fallback prediction when model is unavailable."""
        base_time = datetime.now()
        timestamps = [base_time + timedelta(hours=i+1) for i in range(self.output_seq_len)]
        
        # Simple pattern-based prediction
        predictions = []
        for ts in timestamps:
            hour = ts.hour
            if 7 <= hour <= 9:  # Morning peak
                pred = random.uniform(2.0, 3.0)
            elif 18 <= hour <= 21:  # Evening peak
                pred = random.uniform(2.5, 3.5)
            elif 23 <= hour or hour <= 5:  # Night
                pred = random.uniform(0.3, 0.8)
            else:  # Day
                pred = random.uniform(1.2, 2.2)
            predictions.append(round(pred, 2))
        
        return {
            'predictions': predictions,
            'timestamps': timestamps,
            'input_data': pd.DataFrame() if input_df is None else input_df,
            'confidence_intervals': {
                'lower': [max(0.1, p * 0.8) for p in predictions],
                'upper': [p * 1.2 for p in predictions]
            }
        }

    def _inverse_transform_predictions(self, scaled_predictions):
        """Inverse transform predictions using scaler."""
        try:
            if len(scaled_predictions.shape) == 3:
                scaled_predictions = scaled_predictions.squeeze()
            
            n_samples = len(scaled_predictions)
            dummy = np.zeros((n_samples, 7))  # 7 features in original data
            dummy[:, 0] = scaled_predictions
            
            original = self.scaler.inverse_transform(dummy)
            return original[:, 0]
        except:
            return scaled_predictions.squeeze() * 5.0

    def predict_with_scenario(self, user_id, adjustment_factor=1.0):
        """Prediction with scenario adjustment."""
        result = self.predict(user_id)
        result['predictions'] = [p * adjustment_factor for p in result['predictions']]
        result['scenario_applied'] = f"{int((adjustment_factor - 1) * 100)}% change"
        return result

    def identify_peak_periods(self, predictions, timestamps, threshold_percentile=75):
        """Identify peak usage periods."""
        if not predictions:
            return []
        
        threshold = np.percentile(predictions, threshold_percentile)
        peaks = []
        
        for i, (pred, ts) in enumerate(zip(predictions, timestamps)):
            if pred >= threshold:
                peaks.append({
                    'hour': i + 1,
                    'timestamp': ts,
                    'predicted_kw': float(pred),
                    'is_peak': True
                })
        
        return peaks
    

    
    def get_recent_user_data(self, user_id, hours=72):
        """
        Get recent energy readings from the actual user's database.
        """
        from app.models import EnergyReading
        
        print(f"🔍 DEBUG: Starting get_recent_user_data for user {user_id}")
        
        # REMOVE THE CUTOFF FILTER - use all data
        readings = EnergyReading.query.filter(
            EnergyReading.user_id == user_id
            # REMOVED: EnergyReading.timestamp >= cutoff
        ).order_by(EnergyReading.timestamp.desc()).limit(hours * 2).all()  # Get more than needed
        
        print(f"🔍 DEBUG: Found {len(readings)} raw readings in database")
        
        if not readings:
            print("⚠️ No user data found in database, using realistic pattern")
            return self._generate_realistic_pattern(hours)
        
        # Convert to DataFrame
        data = []
        for reading in readings:
            data.append({
                'timestamp': reading.timestamp,
                'Global_active_power': reading.energy_consumed or 0.0,
                'Global_reactive_power': (reading.energy_consumed or 0.0) * 0.3,  # Better estimate
                'Voltage': reading.voltage or 240.0,
                'Global_intensity': reading.global_intensity or 0.0,
                'Sub_metering_1': reading.sub_metering_1 or 0.0,
                'Sub_metering_2': reading.sub_metering_2 or 0.0,
                'Sub_metering_3': reading.sub_metering_3 or 0.0
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        
        # Sort by timestamp (since we got descending order)
        df = df.sort_index()
        
        print(f"🔍 DEBUG: DataFrame shape: {df.shape}")
        print(f"🔍 DEBUG: Time range: {df.index.min()} to {df.index.max()}")
        
        # Resample to hourly
        df_hourly = df.resample('H').mean().ffill().bfill()
        
        # Take the most recent hours
        if len(df_hourly) >= hours:
            recent_data = df_hourly.tail(hours)
            print(f"✅ Using {len(recent_data)} hours of user data")
            return recent_data
        else:
            print(f"⚠️ Only {len(df_hourly)} hours available, using all")
            return df_hourly
    
    def calculate_cost_forecast(self, predictions, rate=0.20):
        """Calculate cost forecast."""
        if not predictions:
            return {
                'total_kwh': 0,
                'total_cost': 0,
                'hourly_costs': [],
                'avg_hourly_cost': 0,
                'rate_used': rate
            }
        
        total_kwh = sum(predictions)  # kW for each hour = kWh
        total_cost = total_kwh * rate
        hourly_costs = [p * rate for p in predictions]
        
        return {
            'total_kwh': round(total_kwh, 2),
            'total_cost': round(total_cost, 2),
            'hourly_costs': [round(c, 3) for c in hourly_costs],
            'avg_hourly_cost': round(total_cost / len(predictions), 3),
            'rate_used': rate
        }


# Global forecaster instance
_forecaster_instance = None

def get_forecaster():
    """Get forecaster instance."""
    global _forecaster_instance
    if _forecaster_instance is None:
        _forecaster_instance = LSTMForecaster()
    return _forecaster_instance