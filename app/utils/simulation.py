# app/utils/simulation.py

import random
import datetime

DEVICE_LIBRARY = {
    "Air Conditioner": {"power_kw": 2.5, "probability": { "day": 0.2, "night": 0.05 }},
    "Fridge": {"power_kw": 0.15, "probability": { "day": 1.0, "night": 1.0 }},
    "Washing Machine": {"power_kw": 0.5, "probability": { "day": 0.1, "night": 0.02 }},
    "Dishwasher": {"power_kw": 1.0, "probability": { "day": 0.05, "night": 0.1 }},
    "TV": {"power_kw": 0.2, "probability": { "day": 0.15, "night": 0.3 }},
}

def get_active_devices():
    """Return a simulated list of active devices with power usage."""
    now = datetime.datetime.now().hour
    tod = "day" if 7 <= now <= 22 else "night"

    active = []
    for name, meta in DEVICE_LIBRARY.items():
        if random.random() < meta["probability"][tod]:
            usage = round(meta["power_kw"] * random.uniform(0.8, 1.2), 3)
            active.append({"device": name, "power_kw": usage})
    return active
