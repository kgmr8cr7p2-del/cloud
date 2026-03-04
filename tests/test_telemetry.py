"""
Tests for telemetry module.
"""

import time
import pytest

from telemetry.telemetry import Telemetry


class TestTelemetry:
    def test_record_and_get(self):
        t = Telemetry(window_sec=5.0)
        t.record("inference", 10.0)
        t.record("inference", 20.0)
        t.record("inference", 30.0)
        m = t.get("inference")
        assert m.current_ms == 30.0
        assert abs(m.avg_ms - 20.0) < 0.1

    def test_p95(self):
        t = Telemetry(window_sec=10.0)
        for i in range(100):
            t.record("inference", float(i))
        m = t.get("inference")
        assert m.p95_ms >= 94  # p95 of 0..99

    def test_fps_counting(self):
        t = Telemetry(window_sec=1.0)
        for _ in range(10):
            t.record("capture", 5.0)
        fps = t.get_fps()
        assert fps["capture"] >= 5  # at least some recorded

    def test_empty_stage(self):
        t = Telemetry()
        m = t.get("inference")
        assert m.current_ms == 0.0
        assert m.avg_ms == 0.0
        assert m.p95_ms == 0.0
