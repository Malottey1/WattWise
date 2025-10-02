import random


def random_device_draw(device_type):
    """
    Return a typical power draw (kW) for common device types, with small randomization.
    These are demo ranges.
    """
    base = 0.5
    if device_type.lower().startswith("air"):
        base = random.uniform(0.8, 2.5)
    elif "fridge" in device_type.lower() or "refrigerator" in device_type.lower():
        base = random.uniform(0.1, 0.4)
    elif "solar" in device_type.lower():
        # negative = production
        base = -abs(random.uniform(0.2, 1.5))
    elif "meter" in device_type.lower() or "smart" in device_type.lower():
        base = random.uniform(0.1, 1.5)
    else:
        base = random.uniform(0.05, 1.2)
    # jitter
    return round(base * random.uniform(0.85, 1.15), 3)

# app/utils/devices.py

def kwh_from_active_power(active_power_kw, minutes=1.0):
    """
    Convert active power in kW into energy consumed in kWh 
    over a given time duration (in minutes).
    
    kWh = kW * (minutes / 60)
    """
    if active_power_kw is None:
        return 0.0
    try:
        return float(active_power_kw) * (minutes / 60.0)
    except Exception:
        return 0.0

