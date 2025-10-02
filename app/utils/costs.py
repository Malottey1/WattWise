# app/utils.py
import random
from datetime import timedelta
from math import isfinite

# keep your existing functions (generate_recommendation, calculate_gamification_points, carbon_offset_from_savings)
# I assume they are already present; add the following helpers.

def compute_cost(kwh, tariff):
    """
    Compute cost for given kWh according to tariff dict.
    tariff can be:
      - flat: {"type":"flat","rate": 0.20} -> 0.20 currency units per kWh
      - time_of_use: {"type":"tou","peak_rate":0.30,"offpeak_rate":0.15,"peak_hours":[16,17,18,19]}
      - tiered: {"type":"tiered","tiers":[(100,0.10),(200,0.15),(999999,0.20)]}
    """
    if kwh is None or not isfinite(kwh):
        return 0.0

    ttype = tariff.get("type", "flat")
    if ttype == "flat":
        return kwh * float(tariff.get("rate", 0.2))
    if ttype == "tou":
        # if caller passes timestamp-aware usage already, it should pick appropriate rate.
        # Here kwh is for the current period; tariff must include 'is_peak' boolean optionally.
        is_peak = tariff.get("is_peak", False)
        rate = float(tariff.get("peak_rate", 0.3) if is_peak else tariff.get("offpeak_rate", 0.15))
        return kwh * rate
    if ttype == "tiered":
        remaining = kwh
        cost = 0.0
        for limit, r in tariff.get("tiers", []):
            take = min(remaining, limit)
            cost += take * r
            remaining -= take
            if remaining <= 0:
                break
        # if remaining still positive, apply last tier rate
        if remaining > 0:
            cost += remaining * tariff.get("tiers")[-1][1]
        return cost
    # default flat
    return kwh * float(tariff.get("rate", 0.2))


def kwh_from_active_power(active_power_kw, minutes=60):
    """
    Convert instantaneous active power (kW) to energy in kWh for a given minute window.
    active_power_kw: in kW (e.g., 0.45)
    minutes: number of minutes that the reading covers
    """
    return float(active_power_kw) * (minutes / 60.0)


def emission_from_kwh(kwh, kg_per_kwh=0.42):
    """
    Convert kWh to kg CO2-equivalent. Default ~0.42 kg CO2 per kWh (global average-ish).
    """
    return round(kwh * kg_per_kwh, 3)



