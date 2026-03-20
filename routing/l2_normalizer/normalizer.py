"""
routing/l2_normalizer/normalizer.py

L2 Complexity Normalizer -- Phase 1 stub.
Handles: stream fusion, static priority, EMA buffer for L3 fallback.
No ML in Phase 1. Pure rule-based + statistical.
"""

from collections import deque
import time

# ASIL safety priority -- higher = more critical
# Source: architecture decision, session 2
SAFETY_PRIORITY = {
    "brakes":           4,  # ASIL-D
    "wheels":           4,  # ASIL-D
    "motor":            3,  # ASIL-C
    "battery":          3,  # ASIL-C
    "suspension":       2,  # ASIL-B
    "thermal":          2,  # ASIL-B
}

DRIVE_REGIMES = ["highway", "urban", "dynamic", "stationary"]

# EMA window for HELIOS prior fallback (last 5 outputs)
HELIOS_EMA_WINDOW = 5
EMA_ALPHA = 0.3  # decay factor


class ComplexityNormalizer:
    """
    L2: fuses L1 outputs into a normalized context vector for L3.

    Phase 1 scope:
    - Static priority weighting
    - Temporal alignment via fixed-window buffer
    - EMA of last N HELIOS priors (resolves L2/L3 circular dependency)
    - Adaptive regime detection (rule-based, not learned)

    Phase 2 additions (not here yet):
    - Learned projection encoders per stream
    - Cross-stream correlation filtering
    - Learned regime classifier
    """

    def __init__(self):
        self.stream_buffers = {domain: deque(maxlen=50) for domain in SAFETY_PRIORITY}
        self.helios_prior_buffer = deque(maxlen=HELIOS_EMA_WINDOW)
        self.helios_ema = None
        self.current_regime = "stationary"
        self.last_update = time.monotonic()

    def ingest(self, domain, value, timestamp=None):
        """Push a new L1 output into the appropriate stream buffer."""
        if domain not in self.stream_buffers:
            raise ValueError(f"Unknown domain: {domain}")
        self.stream_buffers[domain].append({
            "value": value,
            "ts": timestamp or time.monotonic(),
            "priority": SAFETY_PRIORITY[domain],
        })

    def detect_regime(self):
        """
        Rule-based regime detection from current buffer state.
        Motor RPM and brake pressure are primary signals.
        Returns one of: highway, urban, dynamic, stationary.
        """
        motor_buf = self.stream_buffers.get("motor", deque())
        brake_buf = self.stream_buffers.get("brakes", deque())

        if not motor_buf:
            return "stationary"

        motor_vals = [e["value"] for e in motor_buf if isinstance(e["value"], (int, float))]
        if not motor_vals:
            return "stationary"

        avg_rpm = sum(motor_vals) / len(motor_vals)

        if avg_rpm < 100:
            return "stationary"
        elif avg_rpm > 6000:
            return "highway"

        brake_vals = [e["value"] for e in brake_buf if isinstance(e["value"], (int, float))]
        avg_brake = sum(brake_vals) / len(brake_vals) if brake_vals else 0

        if avg_brake > 50:
            return "dynamic"
        return "urban"

    def fuse(self):
        """
        Produce a priority-weighted summary of all stream buffers.
        Returns dict: domain -> (weighted_mean, priority, n_samples)
        """
        self.current_regime = self.detect_regime()
        summary = {}
        for domain, buf in self.stream_buffers.items():
            if not buf:
                summary[domain] = None
                continue
            vals = [e["value"] for e in buf if isinstance(e["value"], (int, float))]
            if not vals:
                summary[domain] = None
                continue
            mean = sum(vals) / len(vals)
            priority = SAFETY_PRIORITY[domain]
            summary[domain] = {
                "mean": round(mean, 4),
                "priority": priority,
                "n": len(vals),
                "regime": self.current_regime,
            }
        return summary

    def update_helios_prior(self, helios_output):
        """
        Call after each HELIOS inference. Maintains EMA for fallback.
        helios_output: dict (action schema from L3)
        """
        self.helios_prior_buffer.append(helios_output)
        if self.helios_ema is None:
            self.helios_ema = helios_output
        else:
            # EMA update -- only for scalar fields
            updated = {}
            for k, v in helios_output.items():
                if isinstance(v, (int, float)) and k in self.helios_ema:
                    prev = self.helios_ema[k]
                    updated[k] = EMA_ALPHA * v + (1 - EMA_ALPHA) * prev
                else:
                    updated[k] = v
            self.helios_ema = updated

    def get_helios_fallback(self):
        """
        Returns EMA of last HELIOS_EMA_WINDOW outputs.
        Used when L3 is unavailable or exceeds latency budget.
        """
        return self.helios_ema

    def context_packet(self):
        """
        Assemble the full context packet to send to L3.
        Returns: {regime, fused_streams, helios_fallback_available}
        """
        fused = self.fuse()
        return {
            "regime": self.current_regime,
            "streams": fused,
            "helios_fallback": self.helios_ema is not None,
            "timestamp": time.monotonic(),
        }


if __name__ == "__main__":
    norm = ComplexityNormalizer()
    # Simulate a few L1 outputs
    norm.ingest("motor", 4500.0)
    norm.ingest("motor", 4600.0)
    norm.ingest("brakes", 20.0)
    norm.ingest("battery", 380.0)
    norm.ingest("wheels", 900.0)
    packet = norm.context_packet()
    import json
    print(json.dumps(packet, indent=2))
