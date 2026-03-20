"""
testing/unit/test_normalizer.py

Unit tests for L2 ComplexityNormalizer.
Run: pytest testing/unit/test_normalizer.py -v
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from routing.l2_normalizer.normalizer import ComplexityNormalizer, SAFETY_PRIORITY


def test_normalizer_instantiates():
    norm = ComplexityNormalizer()
    assert norm is not None


def test_ingest_known_domain():
    norm = ComplexityNormalizer()
    norm.ingest("motor", 4500.0)
    assert len(norm.stream_buffers["motor"]) == 1


def test_ingest_unknown_domain_raises():
    norm = ComplexityNormalizer()
    try:
        norm.ingest("radar", 1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_fuse_returns_all_domains():
    norm = ComplexityNormalizer()
    for domain in SAFETY_PRIORITY:
        norm.ingest(domain, 100.0)
    result = norm.fuse()
    for domain in SAFETY_PRIORITY:
        assert domain in result


def test_fuse_empty_domain_returns_none():
    norm = ComplexityNormalizer()
    result = norm.fuse()
    for domain in SAFETY_PRIORITY:
        assert result[domain] is None


def test_regime_detection_stationary():
    norm = ComplexityNormalizer()
    norm.ingest("motor", 50.0)
    norm.ingest("motor", 30.0)
    assert norm.detect_regime() == "stationary"


def test_regime_detection_highway():
    norm = ComplexityNormalizer()
    for _ in range(10):
        norm.ingest("motor", 7000.0)
    assert norm.detect_regime() == "highway"


def test_helios_ema_initializes_on_first_update():
    norm = ComplexityNormalizer()
    assert norm.helios_ema is None
    norm.update_helios_prior({"motor_rpm": 5000.0, "confidence": 0.9})
    assert norm.helios_ema is not None


def test_helios_ema_updates():
    norm = ComplexityNormalizer()
    norm.update_helios_prior({"motor_rpm": 5000.0})
    norm.update_helios_prior({"motor_rpm": 6000.0})
    # EMA should be between initial and new value
    ema_rpm = norm.helios_ema["motor_rpm"]
    assert 5000.0 < ema_rpm < 6000.0


def test_fallback_none_before_helios_call():
    norm = ComplexityNormalizer()
    assert norm.get_helios_fallback() is None


def test_context_packet_structure():
    norm = ComplexityNormalizer()
    norm.ingest("motor", 4000.0)
    packet = norm.context_packet()
    assert "regime" in packet
    assert "streams" in packet
    assert "helios_fallback" in packet
    assert "timestamp" in packet


def test_priority_ordering():
    assert SAFETY_PRIORITY["brakes"] == SAFETY_PRIORITY["wheels"]
    assert SAFETY_PRIORITY["brakes"] > SAFETY_PRIORITY["motor"]
    assert SAFETY_PRIORITY["motor"] == SAFETY_PRIORITY["battery"]
    assert SAFETY_PRIORITY["motor"] > SAFETY_PRIORITY["suspension"]
