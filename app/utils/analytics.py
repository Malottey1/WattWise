# app/utils/analytics.py

import numpy as np

def detect_anomalies(data, threshold=2.5):
    """Flag points where usage deviates significantly from mean."""
    mean = np.mean(data)
    std = np.std(data)
    anomalies = [(i, v) for i, v in enumerate(data) if abs(v - mean) > threshold * std]
    return anomalies

def usage_patterns(data, labels):
    """Aggregate usage by time of day or device category."""
    breakdown = {}
    for label, value in zip(labels, data):
        breakdown[label] = breakdown.get(label, 0) + value
    return breakdown
