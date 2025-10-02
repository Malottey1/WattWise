from app import db
from app.models import User, Device, EnergyReading
import datetime
import random

def seed():
    print("🌱 Seeding database with demo data...")

    # Clear existing data
    db.drop_all()
    db.create_all()

    # Add a test user
    user = User(
        username="demo_user",
        email="demo@example.com",
        password_hash="hashedpassword123"  # Normally hash properly!
    )
    db.session.add(user)
    db.session.commit()

    # Add devices
    devices = [
        Device(user_id=user.id, device_name="Smart Fridge", device_type="appliance"),
        Device(user_id=user.id, device_name="Air Conditioner", device_type="appliance"),
        Device(user_id=user.id, device_name="Solar Panel", device_type="renewable"),
    ]
    db.session.add_all(devices)
    db.session.commit()

    # Add energy readings (last 24 hours, hourly)
    base_time = datetime.datetime.now() - datetime.timedelta(hours=24)
    for hour in range(24):
        timestamp = base_time + datetime.timedelta(hours=hour)
        reading = EnergyReading(
            user_id=user.id,
            device_id=devices[0].id,
            timestamp=timestamp,
            energy_consumed=round(random.uniform(0.2, 1.5), 3),
            voltage=round(random.uniform(210, 240), 2),
            global_intensity=round(random.uniform(5, 15), 2)
        )
        db.session.add(reading)

    db.session.commit()
    print("✅ Database seeded successfully!")

if __name__ == "__main__":
    seed()
