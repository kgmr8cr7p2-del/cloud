"""
Tests for One Euro Filter and EMA filter.
"""

import math
import pytest

from filters.smoothing import OneEuroFilter, EMAFilter, FilterPair


class TestOneEuroFilter:
    def test_first_value_passthrough(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        result = f(5.0, t=0.0)
        assert result == 5.0

    def test_smooth_constant_signal(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        for i in range(100):
            result = f(10.0, t=i * 0.016)
        assert abs(result - 10.0) < 0.01

    def test_filters_noise(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.001, d_cutoff=1.0)
        import random
        random.seed(42)
        values = [50.0 + random.gauss(0, 5) for _ in range(200)]
        filtered = []
        for i, v in enumerate(values):
            filtered.append(f(v, t=i * 0.016))

        # Filtered signal variance should be less than raw
        raw_var = sum((v - 50) ** 2 for v in values) / len(values)
        filt_var = sum((v - 50) ** 2 for v in filtered[50:]) / len(filtered[50:])
        assert filt_var < raw_var

    def test_low_lag_fast_movement(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
        # Jump from 0 to 100
        f(0.0, t=0.0)
        result = f(100.0, t=0.016)
        # With high beta, filter should track fast movements closely
        assert result > 50.0  # should track more than half the jump

    def test_reset(self):
        f = OneEuroFilter()
        f(10.0, t=0.0)
        f(20.0, t=0.016)
        f.reset()
        result = f(50.0, t=1.0)
        assert result == 50.0  # after reset, first value passes through

    def test_alpha_bounds(self):
        # alpha should always be in (0, 1]
        alpha = OneEuroFilter._alpha(0.001, 0.001)
        assert 0 < alpha <= 1
        alpha = OneEuroFilter._alpha(1000.0, 0.001)
        assert 0 < alpha <= 1


class TestEMAFilter:
    def test_first_value(self):
        f = EMAFilter(alpha=0.3)
        assert f(10.0) == 10.0

    def test_convergence(self):
        f = EMAFilter(alpha=0.3)
        for _ in range(200):
            result = f(100.0)
        assert abs(result - 100.0) < 0.01

    def test_smoothing(self):
        f = EMAFilter(alpha=0.3)
        f(0.0)
        result = f(100.0)
        assert result == 30.0  # 0.3 * 100 + 0.7 * 0

    def test_reset(self):
        f = EMAFilter(alpha=0.5)
        f(10.0)
        f(20.0)
        f.reset()
        assert f(50.0) == 50.0


class TestFilterPair:
    def test_off_mode_passthrough(self):
        fp = FilterPair()
        fp.configure({"type": "off"})
        x, y = fp.apply(10.0, 20.0)
        assert x == 10.0
        assert y == 20.0

    def test_ema_mode(self):
        fp = FilterPair()
        fp.configure({"type": "ema", "ema_alpha": 0.5, "enable_x": True, "enable_y": True})
        x, y = fp.apply(10.0, 20.0, t=0.0)
        assert x == 10.0
        assert y == 20.0
        x, y = fp.apply(20.0, 40.0, t=0.016)
        assert x == 15.0  # 0.5 * 20 + 0.5 * 10
        assert y == 30.0

    def test_one_euro_mode(self):
        fp = FilterPair()
        fp.configure({
            "type": "one_euro",
            "one_euro_min_cutoff": 1.0,
            "one_euro_beta": 0.007,
            "one_euro_d_cutoff": 1.0,
            "enable_x": True,
            "enable_y": True,
        })
        x, y = fp.apply(10.0, 20.0, t=0.0)
        assert x == 10.0
        assert y == 20.0

    def test_selective_axis(self):
        fp = FilterPair()
        fp.configure({"type": "ema", "ema_alpha": 0.5, "enable_x": False, "enable_y": True})
        fp.apply(10.0, 20.0, t=0.0)
        x, y = fp.apply(20.0, 40.0, t=0.016)
        assert x == 20.0  # X not filtered
        assert y == 30.0  # Y filtered

    def test_reset_clears_state(self):
        fp = FilterPair()
        fp.configure({"type": "ema", "ema_alpha": 0.5, "enable_x": True, "enable_y": True})
        fp.apply(10.0, 20.0, t=0.0)
        fp.apply(20.0, 40.0, t=0.016)
        fp.reset()
        x, y = fp.apply(100.0, 200.0, t=1.0)
        assert x == 100.0
        assert y == 200.0
