"""
Target tracking: lock onto first person, hold until lost, with hysteresis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from detection.postprocess import Detection, iou, center_distance
from app_logging.logger import get_logger

log = get_logger()


@dataclass
class TrackedTarget:
    detection: Detection
    lost_count: int = 0
    age: int = 0  # frames since first locked


class TargetTracker:
    """
    Selects closest-to-center target, locks it, holds until lost.
    After loss, selects new closest target with switch penalty.
    """

    def __init__(self, config_getter) -> None:
        self._cfg = config_getter
        self._locked: TrackedTarget | None = None
        self._prev_detections: list[Detection] = []
        self._prev_locked_cx: float | None = None
        self._prev_locked_cy: float | None = None

    @property
    def locked_target(self) -> TrackedTarget | None:
        return self._locked

    def reset(self) -> None:
        if self._locked:
            log.info("Target manually unlocked")
        self._locked = None
        self._prev_locked_cx = None
        self._prev_locked_cy = None

    def update(self, detections: list[Detection],
               roi_cx: float, roi_cy: float) -> TrackedTarget | None:
        """
        Update tracker with new detections.
        Returns locked target (or None if no valid target).
        """
        cfg = self._cfg()
        lost_thresh = cfg.get("lost_frames_threshold", 15)
        assoc_mode = cfg.get("assoc_mode", "iou")
        iou_thresh = cfg.get("assoc_iou_threshold", 0.25)
        dist_thresh = cfg.get("assoc_distance_threshold", 150)
        switch_penalty = cfg.get("switch_penalty", 50.0)

        if self._locked is not None:
            # Try to associate locked target with a detection
            best_match = self._associate(
                self._locked.detection, detections,
                assoc_mode, iou_thresh, dist_thresh,
            )
            if best_match is not None:
                self._prev_locked_cx = best_match.cx
                self._prev_locked_cy = best_match.cy
                self._locked.detection = best_match
                self._locked.lost_count = 0
                self._locked.age += 1
            else:
                # Fallback: try distance-based association when IoU fails
                # (handles intermittent detections where bbox changes shape)
                if assoc_mode == "iou" and detections:
                    fallback = self._associate(
                        self._locked.detection, detections,
                        "distance", 0, dist_thresh,
                    )
                    if fallback is not None:
                        self._prev_locked_cx = fallback.cx
                        self._prev_locked_cy = fallback.cy
                        self._locked.detection = fallback
                        self._locked.lost_count = 0
                        self._locked.age += 1
                    else:
                        self._locked.lost_count += 1
                else:
                    self._locked.lost_count += 1

                if self._locked is not None and self._locked.lost_count >= lost_thresh:
                    log.info(f"Target lost after {self._locked.age} frames "
                             f"(not seen for {lost_thresh} frames)")
                    self._prev_locked_cx = None
                    self._prev_locked_cy = None
                    self._locked = None

        if self._locked is None and detections:
            self._locked = self._select_new(
                detections, roi_cx, roi_cy, switch_penalty,
            )
            if self._locked:
                self._prev_locked_cx = self._locked.detection.cx
                self._prev_locked_cy = self._locked.detection.cy
                log.info(f"New target locked: conf={self._locked.detection.confidence:.2f} "
                         f"at ({self._locked.detection.cx:.0f}, {self._locked.detection.cy:.0f})")

        self._prev_detections = detections
        return self._locked

    def _associate(self, target: Detection, detections: list[Detection],
                   mode: str, iou_thresh: float, dist_thresh: float,
                   ) -> Detection | None:
        best: Detection | None = None
        best_score = float("inf") if mode == "distance" else -1.0

        for d in detections:
            if mode == "iou":
                score = iou(target, d)
                if score >= iou_thresh and score > best_score:
                    best_score = score
                    best = d
            else:
                dist = center_distance(target, d)
                if dist <= dist_thresh and dist < best_score:
                    best_score = dist
                    best = d
        return best

    def _select_new(self, detections: list[Detection],
                    roi_cx: float, roi_cy: float,
                    switch_penalty: float) -> TrackedTarget | None:
        if not detections:
            return None

        def score(d: Detection) -> float:
            dist = math.hypot(d.cx - roi_cx, d.cy - roi_cy)
            # Apply switch penalty: if we had a previous target,
            # penalize detections far from it to avoid jumping
            if self._prev_locked_cx is not None:
                offset = math.hypot(
                    d.cx - self._prev_locked_cx,
                    d.cy - self._prev_locked_cy,
                )
                dist += offset * (switch_penalty / 100.0)
            return dist

        scored = sorted(detections, key=score)
        return TrackedTarget(detection=scored[0])
