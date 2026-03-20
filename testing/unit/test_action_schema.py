"""
testing/unit/test_action_schema.py

Unit tests for ActionSchema and APEXDispatcher.
Run: pytest testing/unit/test_action_schema.py -v
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from routing.l3_helios.inference.action_schema import (
    ActionSchema, MotorAction, BrakeAction, BatteryAction,
    SuspensionAction, ACTION_SCHEMA_DIM, CONFIDENCE_SAFE_THRESHOLD
)
from routing.l4_apex.apex import APEXDispatcher, WATCHDOG_TIMEOUT_MS


def make_valid_schema(regime="highway", latency_ms=140.0):
    return ActionSchema(
        regime=regime,
        motor=MotorAction(target_rpm=6000.0, torque_limit_nm=300.0, confidence=0.92),
        brake=BrakeAction(pressure_bar=5.0, confidence=0.88),
        battery=BatteryAction(charge_rate_pct=20.0, confidence=0.85),
        suspension=SuspensionAction(damping_pct=60.0, confidence=0.78),
        overall_confidence=0.88,
        latency_ms=latency_ms,
    )


def test_action_schema_dim():
    assert ACTION_SCHEMA_DIM == 14


def test_schema_serializes_to_json():
    schema = make_valid_schema()
    j = schema.to_json()
    assert "motor" in j
    assert "brake" in j
    assert "highway" in j


def test_safe_schema_dispatches():
    schema = make_valid_schema()
    assert schema.is_safe_to_dispatch() is True


def test_low_confidence_blocks_dispatch():
    schema = make_valid_schema()
    schema.overall_confidence = 0.3
    assert schema.is_safe_to_dispatch() is False


def test_low_brake_confidence_blocks_dispatch():
    schema = make_valid_schema()
    schema.brake.confidence = 0.4
    schema.brake.override = False
    assert schema.is_safe_to_dispatch() is False


def test_brake_override_bypasses_confidence():
    schema = make_valid_schema()
    schema.brake.confidence = 0.1
    schema.brake.override = True
    assert schema.is_safe_to_dispatch() is True


def test_fallback_schema_is_flagged():
    ema = {"motor": {"mean": 3000.0}}
    schema = ActionSchema.fallback_schema(ema, regime="urban")
    assert schema.is_fallback is True
    assert schema.regime == "urban"


def test_apex_accepts_valid_schema():
    dispatcher = APEXDispatcher(dry_run=True)
    schema = make_valid_schema()
    result = dispatcher.dispatch(schema)
    assert result["dispatched"] is True


def test_apex_rejects_low_confidence():
    dispatcher = APEXDispatcher(dry_run=True)
    schema = make_valid_schema()
    schema.overall_confidence = 0.2
    result = dispatcher.dispatch(schema)
    assert result["dispatched"] is False


def test_apex_rejects_exceeded_latency():
    dispatcher = APEXDispatcher(dry_run=True)
    schema = make_valid_schema(latency_ms=WATCHDOG_TIMEOUT_MS + 10)
    result = dispatcher.dispatch(schema)
    assert result["dispatched"] is False


def test_apex_brake_safety_clamps_pressure():
    dispatcher = APEXDispatcher(dry_run=True)
    schema = make_valid_schema()
    schema.brake.pressure_bar = 500.0  # way over hardlimit
    schema = dispatcher.apply_brake_safety_layer(schema)
    assert schema.brake.pressure_bar <= 220.0
    assert schema.brake.override is True


def test_apex_dry_run_does_not_raise():
    dispatcher = APEXDispatcher(dry_run=True)
    schema = make_valid_schema()
    try:
        dispatcher.dispatch(schema)
    except Exception as e:
        assert False, f"Dry run raised: {e}"
