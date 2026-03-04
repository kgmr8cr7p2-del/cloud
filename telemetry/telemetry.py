"""
FPS / latency telemetry: current, average, p95 over a sliding window.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StageMetrics:
    current_ms: float = 0.0
    avg_ms: float = 0.0
    p95_ms: float = 0.0


class Telemetry:
    """Collects per-stage timing and exposes current / avg / p95."""

    STAGES = (
        "capture", "preprocess", "inference", "postprocess",
        "tracking", "aim", "overlay", "total",
    )

    def __init__(self, window_sec: float = 5.0) -> None:
        self._window = window_sec
        self._lock = threading.Lock()
        self._history: dict[str, deque[tuple[float, float]]] = {
            s: deque() for s in self.STAGES
        }
        self._current: dict[str, float] = {s: 0.0 for s in self.STAGES}
        self._frame_times: deque[float] = deque()
        self._fps_capture: float = 0.0
        self._fps_inference: float = 0.0
        self._fps_total: float = 0.0
        self._capture_times: deque[float] = deque()
        self._inference_times: deque[float] = deque()
        self._total_times: deque[float] = deque()

    def record(self, stage: str, duration_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._current[stage] = duration_ms
            dq = self._history.get(stage)
            if dq is not None:
                dq.append((now, duration_ms))
                self._prune(dq, now)

            if stage == "capture":
                self._capture_times.append(now)
                self._prune_ts(self._capture_times, now)
            elif stage == "inference":
                self._inference_times.append(now)
                self._prune_ts(self._inference_times, now)
            elif stage == "total":
                self._total_times.append(now)
                self._prune_ts(self._total_times, now)

    def get(self, stage: str) -> StageMetrics:
        now = time.monotonic()
        with self._lock:
            dq = self._history.get(stage)
            if not dq:
                return StageMetrics()
            self._prune(dq, now)
            vals = [v for _, v in dq]
            if not vals:
                return StageMetrics()
            vals_sorted = sorted(vals)
            idx95 = max(0, int(len(vals_sorted) * 0.95) - 1)
            return StageMetrics(
                current_ms=self._current.get(stage, 0.0),
                avg_ms=sum(vals) / len(vals),
                p95_ms=vals_sorted[idx95],
            )

    def get_fps(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            self._prune_ts(self._capture_times, now)
            self._prune_ts(self._inference_times, now)
            self._prune_ts(self._total_times, now)
            return {
                "capture": len(self._capture_times) / self._window if self._capture_times else 0,
                "inference": len(self._inference_times) / self._window if self._inference_times else 0,
                "total": len(self._total_times) / self._window if self._total_times else 0,
            }

    def _prune(self, dq: deque, now: float) -> None:
        cutoff = now - self._window
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _prune_ts(self, dq: deque, now: float) -> None:
        cutoff = now - self._window
        while dq and dq[0] < cutoff:
            dq.popleft()
