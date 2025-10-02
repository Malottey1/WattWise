# app/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app import db, bcrypt
from app.models import User, EnergyReading, Prediction, Recommendation, Device, SavingsLog
from app.forms import RegistrationForm, LoginForm
from app.utils import generate_recommendation, emission_from_kwh, compute_cost
from sqlalchemy import func
import datetime
from flask import Response, stream_with_context
from app.utils.data_stream import simulate_live_stream
from flask import Blueprint, render_template, request, jsonify
from app.models import Device, Reading


from app.simulator import start_simulation, stop_simulation, seed_from_ucidata, list_simulations

bp = Blueprint("main", __name__)

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
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        user = User(
            name=form.name.data,
            email=form.email.data,
            password_hash=hashed_pw
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("main.login"))
    return render_template("register.html", form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            return redirect(url_for("main.dashboard"))
        else:
            flash("Login failed. Please check email and password.", "danger")
    return render_template("login.html", form=form)


@bp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("main.login"))


# -----------------
# Dashboard
# -----------------
@bp.route("/dashboard")
 
def dashboard():
    # Latest reading
    latest = EnergyReading.query.filter_by(user_id=current_user.id)\
        .order_by(EnergyReading.timestamp.desc()).first()

    # Recent readings for chart (last 50, oldest first)
    readings = EnergyReading.query.filter_by(user_id=current_user.id)\
        .order_by(EnergyReading.timestamp.desc()).limit(50).all()[::-1]

    # Predictions (last 24, oldest first)
    preds = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.timestamp.desc()).limit(24).all()[::-1]

    # Recommendations (last 5)
    recs = Recommendation.query.filter_by(user_id=current_user.id)\
        .order_by(Recommendation.generated_at.desc()).limit(5).all()

    # Devices
    devices = Device.query.filter_by(user_id=current_user.id).all()

    # Gamification/savings - simple aggregate for demo
    total_saved_kwh = 0.0
    points = 0
    carbon = 0.0
    # example: read from savings log if exists
    sl = SavingsLog.query.filter_by(user_id=current_user.id).order_by(SavingsLog.created_at.desc()).first()
    if sl:
        total_saved_kwh = sl.energy_saved
        carbon = sl.carbon_offset or emission_from_kwh(total_saved_kwh)
        points = sl.gamification_points or 0

    gamification = {
        "savings_kwh": total_saved_kwh,
        "carbon_offset": carbon,
        "points": points,
        "lightbulbs": int(total_saved_kwh * 10)
    }

    return render_template(
        "dashboard.html",
        latest=latest,
        readings=readings,
        predictions=preds,
        recommendations=recs,
        devices=devices,
        gamification=gamification
    )


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
    recs = generate_recommendation(reading)
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
 
def api_latest_metrics():
    """
    Return latest reading per device for current user + cost/carbon metrics
    """
    tariff = {"type": "flat", "rate": 0.20}  # default; could be user configurable
    devices = Device.query.filter_by(user_id=current_user.id).all()
    out = []
    for d in devices:
        last = EnergyReading.query.filter_by(user_id=current_user.id, device_id=d.id).order_by(EnergyReading.timestamp.desc()).first()
        if last:
            # treat energy_consumed field as instantaneous kW for the demo
            kwh = abs(last.energy_consumed)  # our readings store kW-ish for demo; convert appropriately in frontend
            cost = compute_cost(kwh, tariff)
            carbon = emission_from_kwh(kwh)
            out.append({
                "device_id": d.id,
                "device_name": d.device_name,
                "device_type": d.device_type,
                "last_reading": last.energy_consumed,
                "timestamp": last.timestamp.isoformat(),
                "kwh": kwh,
                "cost": round(cost, 4),
                "carbon_kg": carbon
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
# History
# -----------------
@bp.route("/history")
 
def history():
    readings = EnergyReading.query.filter_by(user_id=current_user.id)\
        .order_by(EnergyReading.timestamp.desc()).all()
    return render_template("history.html", readings=readings)


# -----------------
# Profile
# -----------------
@bp.route("/profile")
 
def profile():
    user = current_user

    devices = Device.query.filter_by(user_id=user.id).all()

    total_readings = db.session.query(func.count(EnergyReading.id))\
        .filter_by(user_id=user.id).scalar()
    avg_consumption = db.session.query(func.avg(EnergyReading.energy_consumed))\
        .filter_by(user_id=user.id).scalar()

    stats = {
        "total_readings": total_readings or 0,
        "avg_consumption": avg_consumption or 0.0,
    }

    return render_template("profile.html", user=user, devices=devices, stats=stats)


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


# -----------------
# Placeholder pages (analytics/predictions/etc.)
# -----------------
@bp.route("/analytics")
 
def analytics():
    return render_template("analytics.html")


@bp.route("/predictions")
 
def predictions():
    return render_template("predictions.html")


@bp.route("/recommendations")
 
def recommendations():
    return render_template("recommendations.html")


@bp.route("/goals")
 
def goals():
    return render_template("goals.html")


@bp.route("/alerts")
 
def alerts():
    return render_template("alerts.html")

@bp.route("/api/live_data")
def live_data():
    def event_stream():
        file_path = "household_power_consumption.txt"
        for reading in simulate_live_stream(file_path, rate=1):
            yield f"data: {reading}\n\n"
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


# --- API Stubs ---

@bp.route("/api/latest_metrics")
def latest_metrics():
    """
    Returns JSON of latest readings and stats for each device.
    Currently stubbed with dummy values.
    """
    devices = Device.query.all()
    data = []
    for d in devices:
        data.append({
            "device_id": d.id,
            "last_reading": None,  # replace later with actual Reading
            "cost": 0.0,
            "carbon_kg": 0.0,
            "timestamp": None
        })
    return jsonify({"devices": data})


@bp.route("/api/start_simulation", methods=["POST"])
def start_simulation():
    """
    Stub: Starts simulating data for a device.
    Expects JSON { device_id, interval, use_model }
    """
    payload = request.get_json()
    device_id = payload.get("device_id")
    interval = payload.get("interval", 10)
    use_model = payload.get("use_model", False)

    # TODO: hook into your simulation logic
    return jsonify({"status": "started", "device_id": device_id, "interval": interval, "use_model": use_model})


@bp.route("/api/stop_simulation", methods=["POST"])
def stop_simulation():
    """
    Stub: Stops simulation for a device.
    Expects JSON { device_id }
    """
    payload = request.get_json()
    device_id = payload.get("device_id")

    # TODO: stop simulation logic
    return jsonify({"status": "stopped", "device_id": device_id})



