"""
Mouse aim controller: computes error, applies speed profile,
generates smooth mouse movements via WinAPI (Windows) or pynput (cross-platform).
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any

from filters.smoothing import FilterPair
from app_logging.logger import get_logger

log = get_logger()


# ── Mouse movement backend ──────────────────────────────────────────
_mouse_backend: str | None = None
_pynput_controller = None


def _init_mouse_backend() -> None:
    global _mouse_backend, _pynput_controller
    if _mouse_backend is not None:
        return

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SendInput
            _mouse_backend = "win32"
            return
        except Exception:
            pass

    try:
        from pynput.mouse import Controller
        _pynput_controller = Controller()
        _mouse_backend = "pynput"
        return
    except ImportError:
        pass

    _mouse_backend = "none"
    log.warn("No mouse movement backend available (install pynput for cross-platform support)")


def _send_mouse_move(dx: int, dy: int) -> None:
    """Move mouse by (dx, dy) pixels relative."""
    _init_mouse_backend()

    if _mouse_backend == "win32":
        _send_mouse_move_win32(dx, dy)
    elif _mouse_backend == "pynput":
        try:
            _pynput_controller.move(dx, dy)
        except Exception:
            pass


def _send_mouse_move_win32(dx: int, dy: int) -> None:
    """Move mouse via Win32 SendInput."""
    import ctypes

    INPUT_MOUSE = 0
    MOUSEEVENTF_MOVE = 0x0001

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        class _INPUT_UNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("union", _INPUT_UNION),
        ]

    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi.dx = dx
    inp.union.mi.dy = dy
    inp.union.mi.dwFlags = MOUSEEVENTF_MOVE
    inp.union.mi.mouseData = 0
    inp.union.mi.time = 0
    inp.union.mi.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    try:
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    except Exception:
        pass


class AimController:
    """
    Computes aim error, applies speed profile and filters,
    sends smooth mouse movements.
    """

    def __init__(self, config_getter) -> None:
        self._cfg = config_getter
        self._filter = FilterPair()
        self._prev_vx: float = 0.0
        self._prev_vy: float = 0.0
        self._prev_time: float = 0.0
        self._active = False
        self._key_held = False

    @property
    def is_active(self) -> bool:
        cfg = self._cfg()
        if not cfg.get("enabled", False):
            return False
        mode = cfg.get("activation_mode", "hold")
        if mode == "always":
            return True
        return self._key_held

    def set_key_held(self, held: bool) -> None:
        self._key_held = held

    def configure_filter(self, filter_cfg: dict) -> None:
        self._filter.configure(filter_cfg)

    def reset_filter(self) -> None:
        self._filter.reset()

    def compute_aim_point(self, det_cx: float, det_cy: float,
                          det_w: float, det_h: float) -> tuple[float, float]:
        """Compute aim point on detection bbox."""
        cfg = self._cfg()
        aim_point = cfg.get("aim_point", "center")
        ox = cfg.get("aim_offset_x", 0.0)
        oy = cfg.get("aim_offset_y", 0.0)

        if aim_point == "upper_third":
            return det_cx + ox * det_w, det_cy - det_h / 3 + oy * det_h
        elif aim_point == "custom":
            return det_cx + ox * det_w, det_cy + oy * det_h
        else:  # center
            return det_cx + ox * det_w, det_cy + oy * det_h

    def compute_and_move(self, target_x: float, target_y: float,
                         screen_cx: float, screen_cy: float,
                         filter_cfg: dict | None = None) -> tuple[float, float]:
        """
        Compute error, filter, apply speed profile, move mouse.
        Returns (dx, dy) error after filtering.
        """
        cfg = self._cfg()
        if filter_cfg:
            self._filter.configure(filter_cfg)

        # Raw error
        raw_dx = target_x - screen_cx
        raw_dy = target_y - screen_cy

        # Apply filter to error or target based on config
        apply_to = filter_cfg.get("apply_to", "error") if filter_cfg else "error"
        t = time.perf_counter()
        if apply_to == "error":
            dx, dy = self._filter.apply(raw_dx, raw_dy, t)
        else:
            fx, fy = self._filter.apply(target_x, target_y, t)
            dx = fx - screen_cx
            dy = fy - screen_cy

        if not self.is_active:
            return dx, dy

        # Dead zone
        deadzone = cfg.get("deadzone", 3)
        dist = math.hypot(dx, dy)
        if dist < deadzone:
            self._prev_vx = 0
            self._prev_vy = 0
            return dx, dy

        # Speed profile
        sensitivity = cfg.get("sensitivity", 1.0)
        max_speed = cfg.get("max_speed", 800)
        max_accel = cfg.get("max_acceleration", 5000)
        profile = cfg.get("speed_profile", "power")

        speed_mult = self._speed_multiplier(dist, cfg, profile)
        move_x = dx * sensitivity * speed_mult
        move_y = dy * sensitivity * speed_mult

        # Clamp speed
        now = time.perf_counter()
        dt = now - self._prev_time if self._prev_time > 0 else 1 / 240
        if dt <= 0:
            dt = 1 / 240
        self._prev_time = now

        move_mag = math.hypot(move_x, move_y)
        max_move = max_speed * dt
        if move_mag > max_move:
            scale = max_move / move_mag
            move_x *= scale
            move_y *= scale

        # Clamp acceleration
        ax = (move_x / dt - self._prev_vx) if dt > 0 else 0
        ay = (move_y / dt - self._prev_vy) if dt > 0 else 0
        accel = math.hypot(ax, ay)
        if accel > max_accel:
            scale = max_accel / accel
            vx = self._prev_vx + ax * scale * dt
            vy = self._prev_vy + ay * scale * dt
            move_x = vx * dt
            move_y = vy * dt

        self._prev_vx = move_x / dt if dt > 0 else 0
        self._prev_vy = move_y / dt if dt > 0 else 0

        # Send mouse movement
        ix = int(round(move_x))
        iy = int(round(move_y))
        if ix != 0 or iy != 0:
            _send_mouse_move(ix, iy)

        return dx, dy

    def _speed_multiplier(self, dist: float, cfg: dict, profile: str) -> float:
        if profile == "power":
            r0 = cfg.get("power_r0", 200.0)
            gamma = cfg.get("power_gamma", 0.7)
            if r0 <= 0:
                r0 = 200
            return (dist / r0) ** gamma
        elif profile == "piecewise":
            r1 = cfg.get("piecewise_r1", 30)
            r2 = cfg.get("piecewise_r2", 200)
            slow = cfg.get("piecewise_slow_factor", 0.3)
            fast = cfg.get("piecewise_fast_factor", 1.5)
            if dist < r1:
                return slow
            elif dist < r2:
                t = (dist - r1) / (r2 - r1)
                return slow + t * (1.0 - slow)
            else:
                return fast
        return 1.0
