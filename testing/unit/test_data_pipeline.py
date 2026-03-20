"""
testing/unit/test_data_pipeline.py

Unit tests for synthetic data generator and anomaly injector.
Run: pytest testing/unit/test_data_pipeline.py -v
"""

import os
import sys
import csv
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from simulation.generators.normal_stream import generate_stream, SENSOR_SPECS, REGIMES
from simulation.injectors.anomaly_injector import inject, ANOMALY_TYPES


def test_normal_stream_generates_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_stream(n_samples=100, output_dir=tmpdir)
        assert os.path.exists(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0


def test_normal_stream_row_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_stream(n_samples=10, output_dir=tmpdir)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        # 10 samples * number of sensors
        assert len(rows) == 10 * len(SENSOR_SPECS)


def test_normal_stream_columns():
    expected = {"timestamp", "sensor_id", "value", "unit", "domain", "regime", "anomaly"}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_stream(n_samples=10, output_dir=tmpdir)
        with open(path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == expected


def test_normal_stream_no_anomalies():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_stream(n_samples=50, output_dir=tmpdir)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert all(int(r["anomaly"]) == 0 for r in rows)


def test_normal_stream_values_in_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_stream(n_samples=200, output_dir=tmpdir)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            sid = row["sensor_id"]
            val = float(row["value"])
            spec = SENSOR_SPECS.get(sid)
            if spec:
                spec_min, spec_max, _, _ = spec
                margin = (spec_max - spec_min) * 0.05  # 5% margin for noise
                assert val >= spec_min - margin, f"{sid} val {val} below min {spec_min}"
                assert val <= spec_max + margin, f"{sid} val {val} above max {spec_max}"


def test_normal_stream_all_regimes():
    with tempfile.TemporaryDirectory() as tmpdir:
        for regime in REGIMES:
            path = generate_stream(n_samples=20, output_dir=tmpdir, regime=regime)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert all(r["regime"] == regime for r in rows)


def test_anomaly_spike_sets_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        normal_path = generate_stream(n_samples=100, output_dir=tmpdir)
        anomaly_path = inject(
            normal_path,
            anomaly_type="spike",
            target_sensor="motor_rpm",
            output_dir=tmpdir,
        )
        with open(anomaly_path) as f:
            rows = list(csv.DictReader(f))
        anomaly_rows = [r for r in rows if int(r["anomaly"]) == 1]
        assert len(anomaly_rows) >= 1


def test_anomaly_drift_sets_multiple_flags():
    with tempfile.TemporaryDirectory() as tmpdir:
        normal_path = generate_stream(n_samples=200, output_dir=tmpdir)
        anomaly_path = inject(
            normal_path,
            anomaly_type="drift",
            target_sensor="battery_temp_c",
            output_dir=tmpdir,
        )
        with open(anomaly_path) as f:
            rows = list(csv.DictReader(f))
        anomaly_rows = [r for r in rows if int(r["anomaly"]) == 1]
        assert len(anomaly_rows) > 5


def test_all_anomaly_types_produce_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        normal_path = generate_stream(n_samples=300, output_dir=tmpdir)
        for atype in ANOMALY_TYPES:
            path = inject(
                normal_path,
                anomaly_type=atype,
                target_sensor="motor_rpm",
                output_dir=tmpdir,
            )
            assert os.path.exists(path), f"No output for anomaly type: {atype}"
