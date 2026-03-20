"""
simulation/injectors/anomaly_injector.py

Takes a normal stream CSV and injects anomalies.
Anomaly types: spike, drift, freeze, out_of_range, noise_burst.

Output: data/synthetic/anomaly/
"""

import csv
import os
import random
import copy
from datetime import datetime

ANOMALY_TYPES = ["spike", "drift", "freeze", "out_of_range", "noise_burst"]

# Severity multipliers for out_of_range
SEVERITY_LEVELS = {"low": 1.1, "medium": 1.5, "high": 2.5, "critical": 4.0}


def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def inject_spike(rows, target_sensor, magnitude=3.0):
    """Single-sample extreme spike on target sensor."""
    targets = [i for i, r in enumerate(rows) if r["sensor_id"] == target_sensor]
    if not targets:
        return rows
    idx = random.choice(targets)
    row = copy.deepcopy(rows[idx])
    val = float(row["value"])
    row["value"] = round(val * magnitude, 4)
    row["anomaly"] = 1
    result = copy.deepcopy(rows)
    result[idx] = row
    return result


def inject_drift(rows, target_sensor, drift_rate=0.02, start_frac=0.3):
    """Gradual drift from start_frac of the stream onward."""
    targets = [i for i, r in enumerate(rows) if r["sensor_id"] == target_sensor]
    start_idx = int(len(targets) * start_frac)
    result = copy.deepcopy(rows)
    for step, idx in enumerate(targets[start_idx:]):
        val = float(result[idx]["value"])
        result[idx]["value"] = round(val * (1 + drift_rate * step), 4)
        result[idx]["anomaly"] = 1
    return result


def inject_freeze(rows, target_sensor, duration_samples=50):
    """Sensor stuck at a value for duration_samples consecutive readings."""
    targets = [i for i, r in enumerate(rows) if r["sensor_id"] == target_sensor]
    if len(targets) < duration_samples:
        return rows
    start = random.randint(0, len(targets) - duration_samples)
    freeze_val = float(rows[targets[start]]["value"])
    result = copy.deepcopy(rows)
    for idx in targets[start: start + duration_samples]:
        result[idx]["value"] = freeze_val
        result[idx]["anomaly"] = 1
    return result


def inject_out_of_range(rows, target_sensor, severity="high"):
    """Replace all values for sensor with out-of-spec values."""
    mult = SEVERITY_LEVELS.get(severity, 1.5)
    result = copy.deepcopy(rows)
    for i, row in enumerate(result):
        if row["sensor_id"] == target_sensor:
            val = float(row["value"])
            result[i]["value"] = round(val * mult, 4)
            result[i]["anomaly"] = 1
    return result


def inject_noise_burst(rows, target_sensor, noise_scale=5.0, burst_len=30):
    """Localized high-frequency noise burst."""
    targets = [i for i, r in enumerate(rows) if r["sensor_id"] == target_sensor]
    if len(targets) < burst_len:
        return rows
    start = random.randint(0, len(targets) - burst_len)
    result = copy.deepcopy(rows)
    for idx in targets[start: start + burst_len]:
        val = float(result[idx]["value"])
        result[idx]["value"] = round(val + random.gauss(0, abs(val) * noise_scale * 0.1), 4)
        result[idx]["anomaly"] = 1
    return result


INJECTORS = {
    "spike": inject_spike,
    "drift": inject_drift,
    "freeze": inject_freeze,
    "out_of_range": inject_out_of_range,
    "noise_burst": inject_noise_burst,
}


def inject(
    input_path,
    anomaly_type=None,
    target_sensor=None,
    output_dir="data/synthetic/anomaly",
    **kwargs,
):
    rows = load_csv(input_path)
    all_sensors = list(set(r["sensor_id"] for r in rows))

    if anomaly_type is None:
        anomaly_type = random.choice(ANOMALY_TYPES)
    if target_sensor is None:
        target_sensor = random.choice(all_sensors)

    injector = INJECTORS[anomaly_type]
    result = injector(rows, target_sensor, **kwargs)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"anomaly_{anomaly_type}_{target_sensor}_{ts_tag}.csv"
    out_path = os.path.join(output_dir, out_name)
    save_csv(result, out_path)
    n_anomaly = sum(1 for r in result if r.get("anomaly") == 1)
    print(f"Injected [{anomaly_type}] on [{target_sensor}] -> {out_path} ({n_anomaly} anomaly rows)")
    return out_path


if __name__ == "__main__":
    # Example: inject spike on motor_rpm from a normal stream
    import sys
    normal_file = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/normal/normal_highway_baseline.csv"
    inject(normal_file, anomaly_type="spike", target_sensor="motor_rpm")
    inject(normal_file, anomaly_type="drift", target_sensor="battery_temp_c")
    inject(normal_file, anomaly_type="freeze", target_sensor="wheel_speed_rpm")
