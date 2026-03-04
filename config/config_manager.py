"""
Thread-safe configuration manager with JSON persistence,
schema versioning, debounced auto-save, and instant apply.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable


CONFIG_VERSION = 2
DEFAULT_CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "_version": CONFIG_VERSION,

    # ── Capture ──────────────────────────────────────────────────────
    "capture": {
        "monitor_index": 0,
        "roi_width": 640,
        "roi_height": 640,
        "fps_limit": 0,          # 0 = unlimited
        "method": "dxcam",       # "dxcam" | "mss"
        "paused": False,
        "show_roi_border": True,
    },

    # ── Inference ────────────────────────────────────────────────────
    "inference": {
        "model_path": "yolov8n.pt",
        "device": "auto",        # "auto" | "cpu" | "cuda"
        "backend": "torch",      # "torch" | "tensorrt"
        "fp16": True,
        "warmup_runs": 3,
        "input_size": 640,
        "engine_path": "",
    },

    # ── Detection ────────────────────────────────────────────────────
    "detection": {
        "confidence": 0.45,
        "nms_iou": 0.45,
        "min_bbox_width": 10,
        "min_bbox_height": 10,
        "max_detections": 20,
        "downscale": 1.0,
    },

    # ── Tracking ─────────────────────────────────────────────────────
    "tracking": {
        "lost_frames_threshold": 15,
        "assoc_iou_threshold": 0.25,
        "assoc_distance_threshold": 150,
        "assoc_mode": "iou",     # "iou" | "distance"
        "switch_penalty": 50.0,
    },

    # ── Aim ──────────────────────────────────────────────────────────
    "aim": {
        "enabled": False,
        "activation_mode": "hold",       # "always" | "hold"
        "activation_key": "x2",          # mouse extra button by default
        "sensitivity": 1.0,
        "max_speed": 800,
        "max_acceleration": 5000,
        "deadzone": 3,
        "aim_point": "center",           # "center" | "upper_third" | "custom"
        "aim_offset_x": 0.0,
        "aim_offset_y": -0.33,
        "speed_profile": "power",        # "power" | "piecewise"
        # power profile params
        "power_r0": 200.0,
        "power_gamma": 0.7,
        # piecewise profile params
        "piecewise_r1": 30,
        "piecewise_r2": 200,
        "piecewise_slow_factor": 0.3,
        "piecewise_fast_factor": 1.5,
        "tick_rate": 240,
    },

    # ── Filters ──────────────────────────────────────────────────────
    "filters": {
        "type": "one_euro",      # "off" | "ema" | "one_euro"
        "apply_to": "error",     # "error" | "target"
        "ema_alpha": 0.3,
        "one_euro_min_cutoff": 1.0,
        "one_euro_beta": 0.007,
        "one_euro_d_cutoff": 1.0,
        "enable_x": True,
        "enable_y": True,
    },

    # ── Overlay ──────────────────────────────────────────────────────
    "overlay": {
        "enabled": True,
        "draw_bboxes": True,
        "draw_locked_target": True,
        "draw_crosshair": True,
        "draw_aim_line": True,
        "show_metrics": True,
    },

    # ── Preview ──────────────────────────────────────────────────────
    "preview": {
        "enabled": True,
        "draw_bboxes": True,
        "draw_text": True,
        "max_fps": 20,
    },

    # ── Export ────────────────────────────────────────────────────────
    "export": {
        "onnx_opset": 17,
        "onnx_dynamic": False,
        "trt_fp16": True,
        "trt_workspace_mb": 4096,
        "trt_engine_path": "",
    },

    # ── GUI ───────────────────────────────────────────────────────────
    "gui": {
        "hotkey_toggle_mouse": "f6",
        "hotkey_toggle_overlay": "f7",
        "hotkey_lock_target": "",
        "hotkey_pause_capture": "f8",
    },
}


def _migrate(data: dict) -> dict:
    """Migrate older config versions forward."""
    v = data.get("_version", 1)
    if v < 2:
        data.setdefault("filters", DEFAULT_CONFIG["filters"])
        data.setdefault("export", DEFAULT_CONFIG["export"])
        data.setdefault("gui", DEFAULT_CONFIG["gui"])
        data["_version"] = 2
    return data


class ConfigManager:
    """Thread-safe live configuration with debounced auto-save."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else DEFAULT_CONFIG_PATH
        self._lock = threading.RLock()
        self._data: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self._listeners: list[Callable[[str, Any], None]] = []
        self._save_timer: threading.Timer | None = None
        self._debounce_sec = 0.5
        self._load()

    # ── public api ───────────────────────────────────────────────────

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Read a value via dotpath, e.g. 'aim.sensitivity'."""
        with self._lock:
            parts = dotpath.split(".")
            node = self._data
            for p in parts:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    return default
            return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def set(self, dotpath: str, value: Any) -> None:
        """Write a value and schedule auto-save."""
        with self._lock:
            parts = dotpath.split(".")
            node = self._data
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
            self._schedule_save()
        for cb in self._listeners:
            try:
                cb(dotpath, value)
            except Exception:
                pass

    def section(self, name: str) -> dict[str, Any]:
        """Return a deep copy of a top-level section."""
        with self._lock:
            return copy.deepcopy(self._data.get(name, {}))

    def add_listener(self, callback: Callable[[str, Any], None]) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, Any], None]) -> None:
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def save_now(self) -> None:
        with self._lock:
            self._cancel_timer()
            self._write()

    def load_profile(self, path: str | Path) -> None:
        self._path = Path(path)
        self._load()

    def reset_defaults(self) -> None:
        with self._lock:
            self._data = copy.deepcopy(DEFAULT_CONFIG)
            self._write()
        for cb in self._listeners:
            try:
                cb("*", None)
            except Exception:
                pass

    def all_data(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    # ── internals ────────────────────────────────────────────────────

    def _load(self) -> None:
        with self._lock:
            if self._path.exists():
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    loaded = _migrate(loaded)
                    self._deep_merge(self._data, loaded)
                except Exception:
                    pass  # keep defaults on corrupt file

    def _write(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            tmp.replace(self._path)
        except Exception:
            if tmp.exists():
                os.remove(tmp)

    def _schedule_save(self) -> None:
        self._cancel_timer()
        self._save_timer = threading.Timer(self._debounce_sec, self._debounced_save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _debounced_save(self) -> None:
        with self._lock:
            self._write()

    def _cancel_timer(self) -> None:
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                ConfigManager._deep_merge(base[k], v)
            else:
                base[k] = v
