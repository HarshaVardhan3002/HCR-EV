"""
simulation/generators/normal_stream.py

Generates synthetic normal-operating EV sensor data.
No MATLAB. No Simulink. Pure Python.

Output: CSV files in data/synthetic/normal/
Schema per row: timestamp, sensor_id, value, domain, regime
"""

import random
import math
import csv
import os
import time
from datetime import datetime

# Drive regimes
REGIMES = ["highway", "urban", "dynamic", "stationary"]

# Sensor domains with normal operating ranges
# Format: (min, max, unit, noise_std)
SENSOR_SPECS = {
    # Motor
    "motor_rpm":        (0, 12000, "rpm",  150.0),
    "motor_torque_nm":  (0, 450,   "Nm",   5.0),
    "motor_temp_c":     (20, 95,   "C",    0.5),
    "motor_current_a":  (0, 500,   "A",    3.0),

    # Wheels (per wheel -- generate 4 channels)
    "wheel_speed_rpm":  (0, 1200,  "rpm",  10.0),
    "wheel_slip_ratio": (0.0, 0.1, "",     0.005),

    # Brakes
    "brake_pressure_bar":  (0, 200, "bar", 1.0),
    "brake_temp_c":        (20, 350, "C",  2.0),

    # Battery
    "battery_soc_pct":   (10, 100, "%",   0.1),
    "battery_voltage_v": (280, 420, "V",  0.5),
    "battery_current_a": (-300, 300, "A", 2.0),
    "battery_temp_c":    (15, 45,  "C",   0.2),
    "cell_voltage_v":    (3.0, 4.2, "V",  0.01),

    # Suspension
    "suspension_travel_mm": (-80, 80, "mm", 2.0),
    "suspension_load_n":    (2000, 8000, "N", 50.0),

    # Thermal
    "inverter_temp_c":  (20, 80,  "C",   0.5),
    "coolant_temp_c":   (20, 90,  "C",   0.3),
    "ambient_temp_c":   (-10, 45, "C",   0.1),
}

# Regime-specific scaling: values are (min_pct, max_pct) of range
REGIME_SCALE = {
    "highway":     {"motor_rpm": (0.6, 0.9), "motor_current_a": (0.4, 0.8)},
    "urban":       {"motor_rpm": (0.1, 0.5), "brake_pressure_bar": (0.1, 0.6)},
    "dynamic":     {"motor_rpm": (0.3, 1.0), "wheel_slip_ratio": (0.02, 0.09)},
    "stationary":  {"motor_rpm": (0.0, 0.02), "battery_current_a": (-0.05, 0.05)},
}


def gaussian_noise(value, std):
    return value + random.gauss(0, std)


def scale_value(sensor_id, regime, spec_min, spec_max):
    scale = REGIME_SCALE.get(regime, {}).get(sensor_id)
    if scale:
        lo = spec_min + scale[0] * (spec_max - spec_min)
        hi = spec_min + scale[1] * (spec_max - spec_min)
    else:
        lo, hi = spec_min, spec_max
    return random.uniform(lo, hi)


def generate_row(ts, sensor_id, regime, spec):
    spec_min, spec_max, unit, noise_std = spec
    base = scale_value(sensor_id, regime, spec_min, spec_max)
    value = gaussian_noise(base, noise_std)
    value = max(spec_min, min(spec_max, value))
    return {
        "timestamp": ts,
        "sensor_id": sensor_id,
        "value": round(value, 4),
        "unit": unit,
        "domain": sensor_id.split("_")[0],
        "regime": regime,
        "anomaly": 0,
    }


def generate_stream(
    n_samples=5000,
    output_dir="data/synthetic/normal",
    filename=None,
    regime=None,
):
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"normal_{ts_tag}.csv"

    rows = []
    t = time.time()
    for i in range(n_samples):
        t += 0.02  # 50Hz sample rate simulation
        current_regime = regime or random.choice(REGIMES)
        for sensor_id, spec in SENSOR_SPECS.items():
            row = generate_row(round(t, 3), sensor_id, current_regime, spec)
            rows.append(row)

    out_path = os.path.join(output_dir, filename)
    fieldnames = ["timestamp", "sensor_id", "value", "unit", "domain", "regime", "anomaly"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    generate_stream(n_samples=5000)
    for regime in REGIMES:
        generate_stream(n_samples=1000, regime=regime, filename=f"normal_{regime}_baseline.csv")
