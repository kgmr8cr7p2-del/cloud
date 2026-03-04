"""
Screen capture module: ROI around screen center.
Primary: dxcam (DXGI-based, Windows).  Fallback: mss.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from app_logging.logger import get_logger

log = get_logger()


def _get_monitors() -> list[dict]:
    """Return list of monitors with index / geometry."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        monitors: list[dict] = []

        def _cb(hmon, hdc, lprect, lparam):
            import ctypes
            r = lprect.contents
            monitors.append({
                "left": r.left, "top": r.top,
                "width": r.right - r.left,
                "height": r.bottom - r.top,
            })
            return 1

        MONITORENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.POINTER(ctypes.wintypes.RECT), ctypes.c_double,
        )
        import ctypes.wintypes
        cb = MONITORENUMPROC(_cb)
        user32.EnumDisplayMonitors(None, None, cb, 0)
        if monitors:
            return monitors
    except Exception:
        pass

    # fallback
    try:
        import mss
        with mss.mss() as s:
            return [
                {"left": m["left"], "top": m["top"],
                 "width": m["width"], "height": m["height"]}
                for m in s.monitors[1:]  # skip virtual "all" monitor
            ]
    except Exception:
        return [{"left": 0, "top": 0, "width": 1920, "height": 1080}]


def get_monitor_list() -> list[dict]:
    return _get_monitors()


class ScreenCapture:
    """
    Captures ROI around the center of a chosen monitor.
    Runs in its own thread, pushes frames into a shared slot (latest frame wins).
    """

    def __init__(self, config_getter) -> None:
        self._cfg = config_getter  # callable returning capture section dict
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_time: float = 0.0
        self._dxcam_camera = None
        self._dxcam_available: bool | None = None
        self._use_dxcam = False
        self._dxcam_output_idx: int | None = None
        self._dxcam_region: tuple[int, int, int, int] | None = None
        self._dxcam_target_fps: int | None = None
        self._dxcam_started = False
        self._mss = None
        self._monitors = _get_monitors()

    @property
    def monitors(self) -> list[dict]:
        return self._monitors

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="capture")
        self._thread.start()
        log.info("Capture thread started")

    def stop(self) -> None:
        self._running = False
        self._stop_dxcam_stream()
        if self._thread:
            self._thread.join(timeout=2)
        self._cleanup_dxcam()
        self._cleanup_mss()
        log.info("Capture thread stopped")

    def grab(self) -> tuple[np.ndarray | None, float]:
        """Return (frame_rgb, timestamp). Non-blocking."""
        with self._frame_lock:
            return self._latest_frame, self._frame_time

    # ── internal ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            cfg = self._cfg()
            if cfg.get("paused", False):
                self._stop_dxcam_stream()
                time.sleep(0.05)
                continue

            fps_limit = cfg.get("fps_limit", 0)
            frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0

            t0 = time.perf_counter()
            region = self._compute_region(cfg)
            method = cfg.get("method", "dxcam")
            frame = self._capture_frame(cfg, region)

            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_time = time.perf_counter()

            elapsed = time.perf_counter() - t0
            if method != "dxcam" and frame_interval > 0:
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _compute_region(self, cfg: dict) -> tuple[int, int, int, int]:
        idx = cfg.get("monitor_index", 0)
        if idx >= len(self._monitors):
            idx = 0
        mon = self._monitors[idx]

        cx = mon["left"] + mon["width"] // 2
        cy = mon["top"] + mon["height"] // 2
        rw = cfg.get("roi_width", 640) // 2
        rh = cfg.get("roi_height", 640) // 2

        left = max(mon["left"], cx - rw)
        top = max(mon["top"], cy - rh)
        right = min(mon["left"] + mon["width"], cx + rw)
        bottom = min(mon["top"] + mon["height"], cy + rh)
        return (left, top, right, bottom)

    def _capture_frame(self, cfg: dict, region: tuple[int, int, int, int]) -> np.ndarray | None:
        method = cfg.get("method", "dxcam")
        if method == "dxcam":
            self._cleanup_mss()
            return self._capture_dxcam(cfg, region)
        self._cleanup_dxcam()
        return self._capture_mss(region)

    def _capture_dxcam(self, cfg: dict, region: tuple[int, int, int, int]) -> np.ndarray | None:
        try:
            if self._dxcam_available is False:
                return self._capture_mss(region)

            try:
                import dxcam
            except ImportError:
                if self._dxcam_available is not False:
                    log.warn("dxcam not installed, using mss fallback")
                self._dxcam_available = False
                return self._capture_mss(region)

            self._dxcam_available = True
            idx = cfg.get("monitor_index", 0)
            target_fps = int(cfg.get("fps_limit", 0) or 0)

            if self._dxcam_camera is None or self._dxcam_output_idx != idx:
                self._cleanup_dxcam()
                self._dxcam_camera = dxcam.create(device_idx=0, output_idx=idx, output_color="RGB")
                self._dxcam_output_idx = idx
                log.info(f"dxcam initialized, monitor {idx}")

            if (not self._dxcam_started
                    or self._dxcam_region != region
                    or self._dxcam_target_fps != target_fps):
                self._stop_dxcam_stream()
                self._dxcam_camera.start(region=region, target_fps=target_fps, video_mode=True)
                self._dxcam_started = True
                self._dxcam_region = region
                self._dxcam_target_fps = target_fps

            frame = self._dxcam_camera.get_latest_frame()
            if frame is not None:
                self._use_dxcam = True
                return frame
            return None
        except Exception as e:
            if self._use_dxcam or self._dxcam_camera is not None:
                log.warn(f"dxcam failed ({e}), falling back to mss")
            self._use_dxcam = False
            self._cleanup_dxcam()
            return self._capture_mss(region)

    def _capture_mss(self, region: tuple[int, int, int, int]) -> np.ndarray | None:
        try:
            if self._mss is None:
                import mss
                self._mss = mss.mss()
                log.info("mss initialized")

            area = {
                "left": region[0], "top": region[1],
                "width": region[2] - region[0],
                "height": region[3] - region[1],
            }
            shot = self._mss.grab(area)
            frame = np.asarray(shot, dtype=np.uint8)[:, :, :3]
            return frame[:, :, ::-1].copy()
        except Exception as e:
            log.error(f"mss capture error: {e}")
            self._cleanup_mss()
            return None

    def _stop_dxcam_stream(self) -> None:
        if self._dxcam_camera is not None and self._dxcam_started:
            try:
                self._dxcam_camera.stop()
            except Exception:
                pass
        self._dxcam_started = False
        self._dxcam_region = None
        self._dxcam_target_fps = None

    def _cleanup_dxcam(self) -> None:
        self._stop_dxcam_stream()
        if self._dxcam_camera is not None:
            try:
                del self._dxcam_camera
            except Exception:
                pass
            self._dxcam_camera = None
        self._dxcam_output_idx = None
        self._use_dxcam = False

    def _cleanup_mss(self) -> None:
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None
