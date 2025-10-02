from datetime import datetime
from app import db, login_manager
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    devices = db.relationship("Device", backref="owner", lazy=True)
    # REMOVED: readings relationship since EnergyReading no longer has user_id
    predictions = db.relationship("Prediction", backref="user", lazy=True)
    recommendations = db.relationship("Recommendation", backref="user", lazy=True)
    savings = db.relationship("SavingsLog", backref="user", lazy=True)


class Device(db.Model):
    __tablename__ = "devices"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    device_name = db.Column(db.String(100), nullable=False)
    device_type = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # REMOVED: readings relationship since EnergyReading no longer has device_id


class EnergyReading(db.Model):
    __tablename__ = "energy_readings"

    id = db.Column(db.Integer, primary_key=True)
    #device_id = db.Column(db.Integer, db.ForeignKey("devices.id"), nullable=False)  # <-- add this
    timestamp = db.Column(db.DateTime, nullable=False)
    global_active_power = db.Column(db.Float, nullable=False)
    global_reactive_power = db.Column(db.Float, nullable=False)
    voltage = db.Column(db.Float)
    global_intensity = db.Column(db.Float)
    sub_metering_1 = db.Column(db.Float)
    sub_metering_2 = db.Column(db.Float)
    sub_metering_3 = db.Column(db.Float)

    #device = db.relationship("Device", backref="readings")



class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    predicted_consumption = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Recommendation(db.Model):
    __tablename__ = "recommendations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    recommendation_text = db.Column(db.Text, nullable=False)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)


class SavingsLog(db.Model):
    __tablename__ = "savings_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    period_start = db.Column(db.Date, nullable=False)
    period_end = db.Column(db.Date, nullable=False)
    energy_saved = db.Column(db.Float, nullable=False)
    carbon_offset = db.Column(db.Float)
    gamification_points = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)