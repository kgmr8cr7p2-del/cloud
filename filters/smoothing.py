"""
Smoothing filters: One Euro Filter and Exponential Moving Average.
"""

from __future__ import annotations

import math
import time


class OneEuroFilter:
    """
    One Euro Filter — adaptive low-pass filter for noisy signals.
    Jitter is removed at low speeds; lag is minimized at high speeds.

    Parameters
    ----------
    min_cutoff : low-speed cutoff frequency (Hz). Lower = more smoothing.
    beta       : speed coefficient. Higher = less lag but more jitter.
    d_cutoff   : derivative cutoff frequency (Hz).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: float | None = None) -> float:
        if t is None:
            t = time.perf_counter()

        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-6
        self._t_prev = t

        # Derivative
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self._x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filtered value
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class EMAFilter:
    """Exponential moving average filter."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self._value: float | None = None

    def reset(self) -> None:
        self._value = None

    def __call__(self, x: float, t: float | None = None) -> float:
        if self._value is None:
            self._value = x
            return x
        self._value = self.alpha * x + (1 - self.alpha) * self._value
        return self._value


class FilterPair:
    """Manages X/Y filter pair with type switching."""

    def __init__(self) -> None:
        self._fx: OneEuroFilter | EMAFilter | None = None
        self._fy: OneEuroFilter | EMAFilter | None = None
        self._type = "off"
        self._enable_x = True
        self._enable_y = True

    def configure(self, cfg: dict) -> None:
        ftype = cfg.get("type", "off")
        self._enable_x = cfg.get("enable_x", True)
        self._enable_y = cfg.get("enable_y", True)

        if ftype != self._type:
            self._type = ftype
            self._fx = None
            self._fy = None

        if ftype == "one_euro":
            params = dict(
                min_cutoff=cfg.get("one_euro_min_cutoff", 1.0),
                beta=cfg.get("one_euro_beta", 0.007),
                d_cutoff=cfg.get("one_euro_d_cutoff", 1.0),
            )
            if self._fx is None:
                self._fx = OneEuroFilter(**params)
                self._fy = OneEuroFilter(**params)
            else:
                for attr, val in params.items():
                    setattr(self._fx, attr, val)
                    setattr(self._fy, attr, val)
        elif ftype == "ema":
            alpha = cfg.get("ema_alpha", 0.3)
            if self._fx is None:
                self._fx = EMAFilter(alpha)
                self._fy = EMAFilter(alpha)
            else:
                self._fx.alpha = alpha
                self._fy.alpha = alpha
        else:
            self._fx = None
            self._fy = None

    def apply(self, x: float, y: float, t: float | None = None) -> tuple[float, float]:
        if self._type == "off" or (self._fx is None):
            return x, y
        rx = self._fx(x, t) if self._enable_x else x
        ry = self._fy(y, t) if self._enable_y else y
        return rx, ry

    def reset(self) -> None:
        if self._fx:
            self._fx.reset()
        if self._fy:
            self._fy.reset()
