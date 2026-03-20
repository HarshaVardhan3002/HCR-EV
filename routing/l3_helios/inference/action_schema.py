"""
routing/l3_helios/inference/action_schema.py

Defines the action schema output format for HELIOS (L3).
This is the structured JSON that Gemma 3 1B headless outputs
via the custom action_schema_head.

All downstream systems (L4 APEX) consume this schema.
Schema version is included for compatibility checking.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time

SCHEMA_VERSION = "0.1.0"

# Action confidence thresholds
CONFIDENCE_SAFE_THRESHOLD = 0.75
CONFIDENCE_DEFER_THRESHOLD = 0.5  # Below this -> use L2 EMA fallback


@dataclass
class MotorAction:
    target_rpm: float           # 0..12000
    torque_limit_nm: float      # 0..450
    regen_enable: bool = True
    confidence: float = 0.0


@dataclass
class BrakeAction:
    pressure_bar: float         # 0..200
    abs_enable: bool = True
    override: bool = False      # True = deterministic rule layer active
    confidence: float = 0.0


@dataclass
class BatteryAction:
    charge_rate_pct: float      # -100..100 (negative = regen)
    temp_limit_c: float = 45.0
    confidence: float = 0.0


@dataclass
class SuspensionAction:
    damping_pct: float          # 0..100
    confidence: float = 0.0


@dataclass
class ActionSchema:
    """
    Full output schema for one HELIOS inference cycle.
    Gemma 3 1B headless outputs a vector that maps to this schema
    via action_schema_head: Linear(hidden_dim -> ACTION_SCHEMA_DIM).
    L4 APEX deserializes and dispatches.
    """
    schema_version: str = SCHEMA_VERSION
    timestamp: float = field(default_factory=time.monotonic)
    regime: str = "unknown"
    motor: Optional[MotorAction] = None
    brake: Optional[BrakeAction] = None
    battery: Optional[BatteryAction] = None
    suspension: Optional[SuspensionAction] = None
    overall_confidence: float = 0.0
    is_fallback: bool = False   # True if L2 EMA was used, not fresh HELIOS output
    latency_ms: float = 0.0

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    def is_safe_to_dispatch(self):
        """
        Basic safety gate before L4 dispatch.
        Brake actions require high confidence OR override flag.
        """
        if self.brake and not self.brake.override:
            if self.brake.confidence < CONFIDENCE_SAFE_THRESHOLD:
                return False
        return self.overall_confidence >= CONFIDENCE_DEFER_THRESHOLD

    @classmethod
    def fallback_schema(cls, l2_ema: dict, regime: str = "unknown"):
        """Construct a conservative fallback schema from L2 EMA."""
        schema = cls(regime=regime, is_fallback=True)
        schema.motor = MotorAction(
            target_rpm=l2_ema.get("motor", {}).get("mean", 0),
            torque_limit_nm=200.0,
            confidence=0.4,
        )
        schema.brake = BrakeAction(pressure_bar=0.0, confidence=0.9)
        schema.overall_confidence = 0.4
        return schema


# ACTION_SCHEMA_DIM: number of floats the action_schema_head outputs
# motor: 3 (rpm, torque, regen) + brake: 3 (pressure, abs, override)
# battery: 2 (charge_rate, temp_limit) + suspension: 1 (damping)
# confidences: 5 (one per subsystem + overall)
ACTION_SCHEMA_DIM = 3 + 3 + 2 + 1 + 5  # = 14


if __name__ == "__main__":
    schema = ActionSchema(
        regime="highway",
        motor=MotorAction(target_rpm=6000.0, torque_limit_nm=300.0, confidence=0.92),
        brake=BrakeAction(pressure_bar=5.0, confidence=0.88),
        battery=BatteryAction(charge_rate_pct=20.0, confidence=0.85),
        suspension=SuspensionAction(damping_pct=60.0, confidence=0.78),
        overall_confidence=0.88,
        latency_ms=142.0,
    )
    print(schema.to_json())
    print(f"Safe to dispatch: {schema.is_safe_to_dispatch()}")
    print(f"ACTION_SCHEMA_DIM: {ACTION_SCHEMA_DIM}")
