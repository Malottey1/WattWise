# app/utils/recommendations.py

def generate_recommendations(predictions, active_devices, rates=0.15):
    """Generate rule-based recommendations."""
    recs = []

    # Example rule 1: shift peak loads
    peak_hours = [h for h in predictions if h["pred_kwh"] > 3.0]  # arbitrary threshold
    if peak_hours and any(d["device"] == "Air Conditioner" for d in active_devices):
        recs.append("Running AC during peak demand hours. Consider shifting use to off-peak.")

    # Example rule 2: stagger high loads
    high_power = [d for d in active_devices if d["power_kw"] > 1.0]
    if len(high_power) > 1:
        recs.append("Multiple high-power devices running simultaneously. Try staggering to save costs.")

    # Example savings estimate
    potential_savings = sum(d["power_kw"] for d in high_power) * rates * 2
    if recs:
        recs.append(f"Estimated savings if applied: ${potential_savings:.2f}/day")

    return recs
