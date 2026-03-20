"""
routing/l4_apex/apex.py

L4: Action Processing and Execution.
Receives ActionSchema from HELIOS (L3), applies safety gates, dispatches.

Key rules:
- Brake safety layer: ML confidence check + deterministic override + hardware watchdog
- L4 -> L1 feedback: offline only. No runtime loop.
- L2/L3 dependency: resolved at L2 via EMA (not here).
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from routing.l3_helios.inference.action_schema import ActionSchema, CONFIDENCE_SAFE_THRESHOLD

# Hardware watchdog timeout (ms). If L3 exceeds this, force fallback.
WATCHDOG_TIMEOUT_MS = 180.0

# Deterministic brake limits -- these override ML output if exceeded
BRAKE_PRESSURE_HARDLIMIT_BAR = 220.0   # physical max
BRAKE_PRESSURE_EMERGENCY_BAR = 200.0   # triggers override flag


class APEXDispatcher:
    """
    L4 action dispatch with deterministic safety overlay.

    Phase 1 scope:
    - Schema validation
    - Brake safety layer (deterministic rules)
    - Hardware watchdog (simulated)
    - Dispatch logging

    Phase 3 additions:
    - Actual CAN bus or hardware interface
    - Real watchdog timer integration
    - ASIL-D compliant brake override circuit
    """

    def __init__(self, dry_run=True):
        self.dry_run = dry_run   # True = log only, no hardware calls
        self.dispatch_log = []
        self.last_dispatch_ts = None
        self.watchdog_active = False

    def validate_schema(self, schema: ActionSchema):
        """Returns (valid: bool, reason: str)."""
        if schema is None:
            return False, "null schema"
        if schema.schema_version != "0.1.0":
            return False, f"schema version mismatch: {schema.schema_version}"
        if not schema.is_safe_to_dispatch():
            return False, f"confidence too low: {schema.overall_confidence:.2f}"
        if schema.latency_ms > WATCHDOG_TIMEOUT_MS:
            return False, f"latency exceeded watchdog: {schema.latency_ms:.1f}ms"
        return True, "ok"

    def apply_brake_safety_layer(self, schema: ActionSchema):
        """
        Deterministic brake safety. Overrides ML output if out of spec.
        This layer is ASIL-D compliant by design intent -- not ML dependent.
        """
        if schema.brake is None:
            return schema
        if schema.brake.pressure_bar > BRAKE_PRESSURE_HARDLIMIT_BAR:
            schema.brake.pressure_bar = BRAKE_PRESSURE_HARDLIMIT_BAR
            schema.brake.override = True
        if schema.brake.pressure_bar > BRAKE_PRESSURE_EMERGENCY_BAR:
            schema.brake.override = True
        if schema.brake.confidence < CONFIDENCE_SAFE_THRESHOLD and not schema.brake.override:
            # Conservative: zero pressure if low confidence and no override
            schema.brake.pressure_bar = 0.0
            schema.brake.override = True
        return schema

    def dispatch(self, schema: ActionSchema):
        """
        Main dispatch method. Returns dispatch result dict.
        """
        t0 = time.monotonic()
        schema = self.apply_brake_safety_layer(schema)
        valid, reason = self.validate_schema(schema)

        result = {
            "timestamp": t0,
            "valid": valid,
            "reason": reason,
            "regime": schema.regime if schema else "unknown",
            "is_fallback": schema.is_fallback if schema else True,
            "dispatched": False,
        }

        if not valid:
            result["action"] = "rejected"
            self._log(result)
            return result

        if self.dry_run:
            # Simulate dispatch
            result["action"] = "dry_run_ok"
            result["dispatched"] = True
            result["motor_rpm"] = schema.motor.target_rpm if schema.motor else None
            result["brake_bar"] = schema.brake.pressure_bar if schema.brake else None
            result["battery_rate"] = schema.battery.charge_rate_pct if schema.battery else None
        else:
            # Phase 3: real hardware dispatch goes here
            result["action"] = "hardware_dispatch"
            result["dispatched"] = True

        dispatch_ms = (time.monotonic() - t0) * 1000
        result["apex_latency_ms"] = round(dispatch_ms, 3)
        self._log(result)
        return result

    def _log(self, result):
        self.dispatch_log.append(result)
        status = "OK" if result["dispatched"] else "REJECTED"
        print(f"[APEX] {status} | regime={result['regime']} | {result.get('reason','')}")

    def flush_log(self, path="testing/benchmarks/apex_dispatch_log.jsonl"):
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            for entry in self.dispatch_log:
                f.write(json.dumps(entry) + "\n")
        self.dispatch_log.clear()


if __name__ == "__main__":
    from routing.l3_helios.inference.action_schema import ActionSchema, MotorAction, BrakeAction, BatteryAction, SuspensionAction
    dispatcher = APEXDispatcher(dry_run=True)
    schema = ActionSchema(
        regime="highway",
        motor=MotorAction(target_rpm=7000.0, torque_limit_nm=350.0, confidence=0.91),
        brake=BrakeAction(pressure_bar=10.0, confidence=0.87),
        battery=BatteryAction(charge_rate_pct=15.0, confidence=0.82),
        suspension=SuspensionAction(damping_pct=65.0, confidence=0.79),
        overall_confidence=0.87,
        latency_ms=145.0,
    )
    result = dispatcher.dispatch(schema)
    import json
    print(json.dumps(result, indent=2))
