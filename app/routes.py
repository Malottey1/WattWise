# app/routes.py
import os
import random
from app.utils import ml_utils
import random
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app import db, bcrypt
from app.models import User, EnergyReading, Prediction, Recommendation, Device, SavingsLog
from app.forms import RegistrationForm, LoginForm
from app.utils.forecasting import get_forecaster
from app.utils.recommendations import generate_recommendations
from app.utils.costs import compute_cost, emission_from_kwh
from collections import deque
from datetime import datetime, timedelta
import random
import json

import time
from functools import wraps

import traceback
from sqlalchemy import func
from flask import Response, stream_with_context
from app.utils.data_stream import simulate_live_stream
from flask import Blueprint, render_template, request, jsonify


from app.simulator import start_simulation, stop_simulation, seed_from_ucidata, list_simulations

bp = Blueprint("main", __name__)

# Lazy load model + scaler
MODEL, SCALER = None, None
def get_model_and_scaler():
    global MODEL, SCALER
    if MODEL is None or SCALER is None:
        MODEL, SCALER = ml_utils.load_model_and_scaler()
    return MODEL, SCALER

# -----------------
# Home
# -----------------
@bp.route("/")
def home():
    return render_template("home.html")


# -----------------
# Authentication
# -----------------
@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")
        
        if not all([name, email, password, confirm]):
            flash("All fields are required", "danger")
            return render_template("register.html")
        
        if password != confirm:
            flash("Passwords do not match", "danger")
            return render_template("register.html")
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return render_template("register.html")
        
        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
        user = User(name=name, email=email, password_hash=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("main.login"))
    
    return render_template("register.html")
@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        if not email or not password:
            flash("Please enter both email and password", "danger")
            return render_template("login.html")
        
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for("main.dashboard"))
        else:
            flash("Login failed. Please check email and password.", "danger")
    
    return render_template("login.html")


@bp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("main.login"))


# -----------------
# Dashboard
# -----------------
@bp.route("/dashboard")
@login_required
def dashboard():
    # Load initial data from household_power_consumption.txt
    readings = load_initial_readings_from_file()
    
    # Get latest reading
    latest = readings[-1] if readings else None
    
    # Get user's devices
    devices = Device.query.filter_by(user_id=current_user.id).all()
    
    # Calculate initial metrics from file data
    total_cost = 0.0
    total_carbon = 0.0
    active_devices = 0
    usage_sum = 0.0
    
    # Calculate metrics from recent readings
    for reading in readings[-10:]:  # Last 10 readings
        if reading.energy_consumed:
            total_cost += compute_cost(abs(reading.energy_consumed), {"rate": 0.20})
            total_carbon += emission_from_kwh(abs(reading.energy_consumed))
            usage_sum += abs(reading.energy_consumed)
            active_devices = 1  # At least one active device if we have readings
    
    # Calculate usage percentage for gauge
    max_expected = 5.0  # 5 kW max expected
    usage_percent = min(100, int((usage_sum / max_expected) * 100)) if max_expected > 0 else 0

    return render_template(
        "dashboard.html",
        latest=latest,
        readings=readings[-50:],  # Last 50 readings for chart
        devices=devices,
        total_cost=total_cost,
        total_carbon=total_carbon,
        active_devices=active_devices,
        usage_percent=usage_percent
    )


from collections import deque
from datetime import datetime, timedelta
import random

def load_initial_readings_from_file():
    """Load last 100 readings from household_power_consumption.txt efficiently"""
    readings = []
    file_path = "app/household_power_consumption.txt"

    try:
        with open(file_path, 'r') as file:
            # Keep only the last 101 lines (100 + maybe header)
            last_lines = deque(file, maxlen=101)

        # Convert to list for slicing
        last_lines = list(last_lines)

        # Remove header if exists
        if last_lines and last_lines[0].startswith("Date"):
            last_lines = last_lines[1:]

        for idx, line in enumerate(last_lines[-100:]):  # ✅ now works
            parts = line.strip().split(';')
            if len(parts) >= 3:
                try:
                    global_active_power = float(parts[2].strip()) if parts[2].strip() != '?' else 0.0
                    
                    class MockReading:
                        def __init__(self, energy, timestamp):
                            self.energy_consumed = energy
                            self.timestamp = timestamp
                            self.voltage = float(parts[4].strip()) if len(parts) > 4 and parts[4].strip() != '?' else None
                            self.device = None
                            self.device_id = 1

                    # Spread timestamps across last 24 hours
                    base_time = datetime.now() - timedelta(hours=24)
                    timestamp = base_time + timedelta(minutes=idx * 10)

                    reading = MockReading(global_active_power, timestamp)
                    readings.append(reading)

                except (ValueError, IndexError):
                    continue

        print(f"Loaded {len(readings)} readings from file")

    except FileNotFoundError:
        print(f"File {file_path} not found - using mock data")
        base_time = datetime.now() - timedelta(hours=24)
        for i in range(50):
            mock_reading = type('MockReading', (), {
                'energy_consumed': random.uniform(0.5, 3.0),
                'timestamp': base_time + timedelta(minutes=i * 30),
                'voltage': 240.0,
                'device': None,
                'device_id': 1
            })()
            readings.append(mock_reading)

    return readings



# -----------------
# API: Submit IoT Reading
# -----------------
@bp.route("/api/submit_reading", methods=["POST"])
def submit_reading():
    data = request.json
    user_id = data.get("user_id") or (current_user.id if current_user.is_authenticated else None)

    if not user_id:
        return jsonify({"status": "error", "message": "No user_id provided"}), 400

    reading = EnergyReading(
        user_id=user_id,
        device_id=data.get("device_id"),
        timestamp=datetime.datetime.utcnow(),
        energy_consumed=data["energy_consumed"],
        voltage=data.get("voltage"),
        global_intensity=data.get("global_intensity"),
        sub_metering_1=data.get("sub_metering_1"),
        sub_metering_2=data.get("sub_metering_2"),
        sub_metering_3=data.get("sub_metering_3")
    )
    db.session.add(reading)
    db.session.commit()

    # Generate recommendations (append)
    recs = generate_recommendations(reading)
    for r in recs:
        new_rec = Recommendation(user_id=user_id, recommendation_text=r)
        db.session.add(new_rec)
    db.session.commit()
    return jsonify({"status": "success", "recommendations": recs})


# -----------------
# Simulation control API
# -----------------
@bp.route("/api/start_simulation", methods=["POST"])
 
def api_start_simulation():
    data = request.json or {}
    device_id = data.get("device_id") or request.form.get("device_id")
    interval = int(data.get("interval", 10))
    use_model = data.get("use_model", True)
    if not device_id:
        return jsonify({"status": "error", "message": "device_id required"}), 400
    started = start_simulation(current_user.id, int(device_id), interval_seconds=interval, use_model=use_model)
    return jsonify({"status": "started" if started else "already_running"})


@bp.route("/api/stop_simulation", methods=["POST"])
 
def api_stop_simulation():
    data = request.json or {}
    device_id = data.get("device_id") or request.form.get("device_id")
    if not device_id:
        return jsonify({"status": "error", "message": "device_id required"}), 400
    stop_simulation(current_user.id, int(device_id))
    return jsonify({"status": "stopped"})


@bp.route("/api/list_simulations")
 
def api_list_simulations():
    sims = list_simulations()
    return jsonify({"simulations": sims})


@bp.route("/api/latest_metrics")
@login_required
def api_latest_metrics():
    """
    Return latest reading per device for current user + cost/carbon metrics.
    Uses simulated data (from txt file / appliance ratings), not EnergyReading table.
    """
    import random, datetime
    tariff = {"type": "flat", "rate": 0.20}  # could later be time-of-use
    emission_factor = 0.233  # kg CO2 per kWh (example EU avg)

    devices = Device.query.filter_by(user_id=current_user.id).all()
    out = []

    # --- Example appliance power ratings (kW) ---
    power_library = {
        "Fridge": (0.1, 0.3),
        "TV": (0.05, 0.2),
        "AC": (0.5, 2.0),
        "Heater": (1.0, 3.0),
        "Washing Machine": (0.5, 2.0),
        "Computer": (0.1, 0.5),
        "Lighting": (0.05, 0.2)
    }

    for d in devices:
        # Decide if this device is currently active (random for demo)
        active = random.choice([True, False, False])  # ~33% chance
        if active:
            # Pick a simulated reading based on type, or default small load
            if d.device_name in power_library:
                low, high = power_library[d.device_name]
                reading = round(random.uniform(low, high), 3)
            else:
                reading = round(random.uniform(0.05, 0.5), 3)

            kwh = reading
            cost = kwh * tariff["rate"]
            carbon = kwh * emission_factor
            out.append({
                "device_id": d.id,
                "device_name": d.device_name,
                "device_type": d.device_type,
                "last_reading": reading,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "kwh": kwh,
                "cost": round(cost, 4),
                "carbon_kg": round(carbon, 4)
            })
        else:
            out.append({
                "device_id": d.id,
                "device_name": d.device_name,
                "device_type": d.device_type,
                "last_reading": None,
                "timestamp": None,
                "kwh": 0.0,
                "cost": 0.0,
                "carbon_kg": 0.0
            })

    return jsonify({"devices": out})






# -----------------
# Profile
# -----------------
@bp.route("/profile")
@login_required
def profile():
    user = current_user

    # Get all devices for this user
    devices = Device.query.filter_by(user_id=user.id).all()
    device_count = len(devices)

    # Simulate usage summary
    usage_summary = []
    total_kwh = 0
    total_cost = 0
    emission_factor = 0.233
    tariff_rate = 0.20

    for d in devices:
        kwh = round(random.uniform(0.1, 2.0), 2)
        cost = kwh * tariff_rate
        carbon = kwh * emission_factor
        usage_summary.append({
            "device_name": d.device_name,
            "device_type": d.device_type,
            "kwh": kwh,
            "cost": round(cost, 2),
            "carbon": round(carbon, 2)
        })
        total_kwh += kwh
        total_cost += cost

    # --- NEW: Get dynamic data ---
    # Alerts
    alerts = []
    try:
        alerts = api_alerts().json["alerts"]
    except Exception as e:
        print(f"⚠️ Failed to load alerts: {e}")

    # Recommendations
    recs = []
    try:
        recs = api_recommendations().json["recommendations"]
    except Exception as e:
        print(f"⚠️ Failed to load recommendations: {e}")

    # Goals
    goals = {}
    try:
        goals = api_goals().json
    except Exception as e:
        print(f"⚠️ Failed to load goals: {e}")

    return render_template(
        "profile.html",
        user=user,
        devices=devices,
        device_count=device_count,
        usage_summary=usage_summary,
        total_kwh=round(total_kwh, 2),
        total_cost=round(total_cost, 2),
        alerts=alerts,
        recommendations=recs,
        goals=goals
    )




# -----------------
# Add Device
# -----------------
@bp.route("/add_device", methods=["POST"])
 
def add_device():
    name = request.form.get("device_name")
    dtype = request.form.get("device_type")
    simulate = request.form.get("simulate")

    new_device = Device(
        user_id=current_user.id,
        device_name=name,
        device_type=dtype
    )
    db.session.add(new_device)
    db.session.commit()

    if simulate:
        # start simulation for this device (background thread)
        try:
            start_simulation(current_user.id, new_device.id, interval_seconds=10, use_model=True)
            flash(f"Device '{name}' connected and simulation started!", "success")
        except Exception:
            flash(f"Device '{name}' connected, but simulation failed to start.", "warning")
    else:
        flash(f"Device '{name}' connected!", "success")

    return redirect(url_for("main.dashboard"))





@bp.route("/recommendations")
 
def recommendations():
    return render_template("recommendations.html")


@bp.route("/goals")
 
def goals():
    return render_template("goals.html")


@bp.route("/alerts")
 
def alerts():
    return render_template("alerts.html")

from flask import Response
from app.utils.data_stream import simulate_live_stream

@bp.route("/api/live_data")
def live_data():
    """
    SSE endpoint: stream household readings from the txt file using
    simulate_live_stream, but also embed recommendations & alerts
    each cycle (always guaranteed at least one).
    """
    def event_stream():
        import json
        file_path = "app/household_power_consumption.txt"

        for reading in simulate_live_stream(file_path, rate=2.0):
            payload = {
                "timestamp": (
                    reading["timestamp"].isoformat()
                    if hasattr(reading["timestamp"], "isoformat")
                    else str(reading["timestamp"])
                ),
                "energy_consumed": reading["power"],
                "voltage": reading["voltage"],
                "intensity": reading["intensity"],
                "sub_metering": reading["sub_metering"]
            }

            try:
                recs = api_recommendations().json.get("recommendations", [])
                alerts = api_alerts().json.get("alerts", [])
            except Exception as e:
                print(f"⚠️ Failed to inject recs/alerts: {e}")
                recs, alerts = [], []

            # ✅ Ensure at least one recommendation
            if not recs:
                recs = [{
                    "device_name": None,
                    "suggestion": "💡 Keep up the good work! No changes needed."
                }]

            # ✅ Ensure at least one alert
            if not alerts:
                alerts = [{
                    "device_name": None,
                    "device_type": None,
                    "message": "✅ No abnormal usage detected.",
                    "timestamp":  datetime.utcnow().isoformat()
                }]

            payload["recommendations"] = recs
            payload["alerts"] = alerts

            yield f"data: {json.dumps(payload)}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")







# -----------------
# AI Predictions Page & API
# -----------------
@bp.route("/predictions")
 
def predictions():
    """Render the AI Prediction Hub page."""
    return render_template("predictions.html")





@bp.route("/api/historical_vs_predicted", methods=["GET"])
 
def api_historical_vs_predicted():
    """
    Get historical data vs recent predictions for comparison.
    Returns last 24 hours of actual readings alongside predictions.
    """
    try:
        # Get last 24 hours of actual readings
        cutoff = datetime.utcnow() - datetime.timedelta(hours=24)
        actual_readings = EnergyReading.query.filter(
            EnergyReading.user_id == current_user.id,
            EnergyReading.timestamp >= cutoff
        ).order_by(EnergyReading.timestamp.asc()).all()
        
        # Get predictions for comparison
        predictions = Prediction.query.filter(
            Prediction.user_id == current_user.id,
            Prediction.timestamp >= cutoff
        ).order_by(Prediction.timestamp.asc()).all()
        
        response = {
            'status': 'success',
            'actual': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'consumption': r.energy_consumed
                } for r in actual_readings
            ],
            'predicted': [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'consumption': p.predicted_consumption
                } for p in predictions
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500




def timing_decorator(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            print(f"⏱️ {func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"⏱️ {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper

# ... (keep all your existing imports and routes above)

@bp.route("/api/debug_user_data")
def debug_user_data():
    """Debug endpoint to check user data availability"""
    from app.models import EnergyReading
    
    user_id = current_user.id
    cutoff = datetime.utcnow() - datetime.timedelta(days=30)
    
    readings = EnergyReading.query.filter(
        EnergyReading.user_id == user_id,
        EnergyReading.timestamp >= cutoff
    ).order_by(EnergyReading.timestamp.asc()).all()
    
    debug_info = {
        'user_id': user_id,
        'total_readings': len(readings),
        'readings_sample': [],
        'timestamp_range': None
    }
    
    if readings:
        debug_info['timestamp_range'] = {
            'first': readings[0].timestamp.isoformat(),
            'last': readings[-1].timestamp.isoformat()
        }
        
        # Sample of readings
        for i, reading in enumerate(readings[:5]):
            debug_info['readings_sample'].append({
                'id': reading.id,
                'timestamp': reading.timestamp.isoformat(),
                'energy_consumed': reading.energy_consumed,
                'voltage': reading.voltage,
                'global_intensity': reading.global_intensity
            })
    
    return jsonify(debug_info)

@bp.route("/api/generate_forecast", methods=["GET"])
@login_required  # ADD THIS
def api_generate_forecast():
    print("🎯 ===== /api/generate_forecast CALLED =====")
    print(f"👤 User ID: {current_user.id}")
    print(f"📧 User Email: {current_user.email}")
    print(f"🔍 Is authenticated: {current_user.is_authenticated}")
    
    try:
        # Add more debug prints throughout the function
        print("🔄 Step 1: Getting forecaster instance...")
        forecaster = get_forecaster()
        print(f"🔄 Forecaster: {forecaster}")
        print(f"🔄 Forecaster model: {forecaster.model}")
        start_time = time.time()
        print("🔄 Step 1: Getting forecaster instance...")
        forecaster = get_forecaster()
        
        # Check for scenario parameter
        scenario = request.args.get('scenario', type=float)
        print(f"⚙️ Scenario parameter: {scenario}")
        
        if scenario:
            print("🔄 Step 2: Generating forecast with scenario...")
            result = forecaster.predict_with_scenario(
                user_id=current_user.id, 
                adjustment_factor=scenario
            )
        else:
            print("🔄 Step 2: Generating standard forecast...")
            result = forecaster.predict(user_id=current_user.id)
        
        print(f"✅ Step 3: Forecast generated successfully")
        print(f"   - Predictions count: {len(result['predictions'])}")
        print(f"   - Timestamps count: {len(result['timestamps'])}")
        
        print("🔄 Step 4: Identifying peak periods...")
        peaks = forecaster.identify_peak_periods(
            result['predictions'], 
            result['timestamps']
        )
        print(f"   - Peak periods found: {len(peaks)}")
        
        print("🔄 Step 5: Calculating costs...")
        costs = forecaster.calculate_cost_forecast(result['predictions'])
        print(f"   - Total cost: ${costs['total_cost']}")
        print(f"   - Total kWh: {costs['total_kwh']}")
        
        response = {
            'status': 'success',
            'forecast': {
                'timestamps': [ts.isoformat() for ts in result['timestamps']],
                'predictions': result['predictions'],
                'confidence_lower': result['confidence_intervals']['lower'],
                'confidence_upper': result['confidence_intervals']['upper']
            },
            'peaks': peaks,
            'cost_forecast': costs,
            'model': 'LSTM_user_data'
        }
        
        if scenario:
            response['scenario'] = {
                'applied': True,
                'factor': scenario,
                'description': result.get('scenario_applied', '')
            }
        
        execution_time = time.time() - start_time
        print(f"✅ Step 6: Response prepared in {execution_time:.2f}s")
        print("🎉 ===== /api/generate_forecast COMPLETED =====")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"💥 ===== /api/generate_forecast FAILED =====")
        print(f"❌ Error type: {type(e).__name__}")
        print(f"❌ Error message: {str(e)}")
        import traceback
        print("🔍 Stack trace:")
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate forecast: {str(e)}'
        }), 500

@bp.route("/api/scenario_comparison", methods=["POST"])
 
def api_scenario_comparison():
    """Fast scenario comparison."""
    try:
        data = request.get_json()
        scenarios = data.get('scenarios', [
            {"name": "Current", "factor": 1.0},
            {"name": "10% Reduction", "factor": 0.9},
            {"name": "20% Reduction", "factor": 0.8}
        ])
        
        forecaster = get_forecaster()
        results = []
        
        # Generate baseline once - FIXED: added user_id
        baseline = forecaster.predict(user_id=current_user.id)
        baseline_costs = forecaster.calculate_cost_forecast(baseline['predictions'])
        
        for scenario in scenarios:
            # FIXED: added user_id parameter
            scenario_result = forecaster.predict_with_scenario(
                user_id=current_user.id,
                adjustment_factor=scenario['factor']
            )
            costs = forecaster.calculate_cost_forecast(scenario_result['predictions'])
            
            results.append({
                'name': scenario['name'],
                'factor': scenario['factor'],
                'total_kwh': costs['total_kwh'],
                'total_cost': costs['total_cost'],
                'predictions': scenario_result['predictions'],  # Removed .tolist() since it's already a list
                'timestamps': [ts.isoformat() for ts in scenario_result['timestamps']]
            })
        
        # Calculate savings compared to baseline
        baseline_cost = baseline_costs['total_cost']
        for r in results:
            if r['factor'] != 1.0:  # Not baseline
                r['savings'] = baseline_cost - r['total_cost']
                r['savings_percent'] = ((baseline_cost - r['total_cost']) / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        return jsonify({
            'status': 'success',
            'scenarios': results
        })
    
    except Exception as e:
        print(f"❌ Scenario comparison error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route("/api/peak_analysis", methods=["GET"])
 
def api_peak_analysis():
    """
    Analyze peak usage patterns from predictions.
    """
    try:
        forecaster = get_forecaster()
        # FIXED: added user_id parameter
        result = forecaster.predict(user_id=current_user.id)
        peaks = forecaster.identify_peak_periods(
            result['predictions'],
            result['timestamps'],
            threshold_percentile=70
        )
        
        # Group peaks by time of day
        morning = []  # 6-12
        afternoon = []  # 12-18
        evening = []  # 18-24
        night = []  # 0-6
        
        for peak in peaks:
            hour = peak['timestamp'].hour
            if 6 <= hour < 12:
                morning.append(peak)
            elif 12 <= hour < 18:
                afternoon.append(peak)
            elif 18 <= hour < 24:
                evening.append(peak)
            else:
                night.append(peak)
        
        # Generate recommendations
        recommendations = []
        if len(evening) > len(morning):
            recommendations.append({
                'message': 'Your highest energy usage is in the evening. Consider shifting heavy appliance use to morning hours.',
                'potential_savings': '$2-5 per day'
            })
        
        if len(peaks) > 12:  # More than half the day at peak
            recommendations.append({
                'message': 'You have sustained high usage. Review always-on appliances and phantom loads.',
                'potential_savings': '10-15% reduction possible'
            })
        
        return jsonify({
            'status': 'success',
            'peak_summary': {
                'total_peak_hours': len(peaks),
                'by_period': {
                    'morning': len(morning),
                    'afternoon': len(afternoon),
                    'evening': len(evening),
                    'night': len(night)
                }
            },
            'peaks': peaks,
            'recommendations': recommendations
        })
    
    except Exception as e:
        print(f"❌ Peak analysis error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



# Forecast API
@bp.route("/api/forecast", methods=["GET"])
def api_forecast():
    """Return H-hour forecast as JSON. Optional query params: adjustment (-30,0,30), horizon"""
    adjustment = float(request.args.get("adjustment", 0.0))
    horizon = int(request.args.get("horizon", os.getenv("HORIZON", "24")))

    model, scaler = get_model_and_scaler()
    df_pred = ml_utils.predict_from_db(db.engine, model, scaler, horizon=horizon)

    print("DEBUG forecast df columns:", df_pred.columns.tolist())
    print(df_pred.head())


    factor = 1.0 + (adjustment / 100.0)
    df_pred["predicted"] = df_pred["predicted"] * factor

    kwh_rate = float(os.getenv("KWH_RATE", "0.15"))
    df_pred["estimated_cost"] = df_pred["predicted"] * kwh_rate

    def status_recommend(row):
        if row["predicted"] > df_pred["predicted"].mean() * 1.2:
            return ("High", "Shift usage if possible")
        elif row["predicted"] < df_pred["predicted"].mean() * 0.8:
            return ("Low", "Good time for heavy loads")
        else:
            return ("Normal", "Standard usage")

    statuses = df_pred.apply(status_recommend, axis=1)
    df_pred["status"] = [s[0] for s in statuses]
    df_pred["recommendation"] = [s[1] for s in statuses]


    result = df_pred.reset_index().to_dict(orient="records")
    return jsonify(result)


# -----------------
# Recommendations
# -----------------
# ----------- RECOMMENDATIONS API -----------
@bp.route("/api/recommendations")
def api_recommendations():
    recs = []

    # Example: build recommendations
    for d in Device.query.all():
        if random.random() < 0.2:  # simulate random recommendation chance
            recs.append({
                "device_name": d.device_name,
                "suggestion": f"Consider turning off {d.device_name} when not in use."
            })

    # ✅ Always return at least one suggestion
    if not recs:
        recs.append({
            "device_name": None,
            "suggestion": "💡 Keep up the good work! No changes needed."
        })

    return jsonify({"recommendations": recs})




@bp.route("/api/alerts")
def api_alerts():
    alerts = []

    # Example: build alerts from conditions
    for d in Device.query.all():
        if random.random() < 0.2:  # simulate random alert chance
            alerts.append({
                "device_name": d.device_name,
                "device_type": d.device_type,
                "message": f"High usage detected for {d.device_name}",
                "timestamp":  datetime.utcnow().isoformat()
            })

    # ✅ Always return at least one message
    if not alerts:
        alerts.append({
            "device_name": None,
            "device_type": None,
            "message": "✅ No abnormal usage detected.",
            "timestamp":  datetime.utcnow().isoformat()
        })

    return jsonify({"alerts": alerts})



# -----------------
# Goals (Gamification)
# -----------------
@bp.route("/api/goals")
def api_goals():
    import random

    devices = Device.query.filter_by(user_id=current_user.id).all()

    total_kwh = 0
    for d in devices:
        total_kwh += round(random.uniform(0.1, 2.0), 2)

    daily_goal = 20  # kWh (could be user-configurable later)

    return jsonify({
        "goal": daily_goal,
        "current_usage": round(total_kwh, 2),
        "progress": round((total_kwh / daily_goal) * 100, 1),
        "status": "On Track" if total_kwh <= daily_goal else "Exceeded"
    })




