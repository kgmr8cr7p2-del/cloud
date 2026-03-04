"""
Tests for target tracking logic with synthetic bounding boxes.
"""

import pytest

from detection.postprocess import Detection, iou, center_distance, nms
from tracking.tracker import TargetTracker, TrackedTarget


def make_det(cx: float, cy: float, w: float = 50, h: float = 100,
             conf: float = 0.8) -> Detection:
    return Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf)


class TestIoU:
    def test_identical_boxes(self):
        d = make_det(100, 100, 50, 50)
        assert abs(iou(d, d) - 1.0) < 1e-6

    def test_no_overlap(self):
        a = make_det(0, 0, 10, 10)
        b = make_det(100, 100, 10, 10)
        assert iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = Detection(0, 0, 10, 10, 0.9)
        b = Detection(5, 5, 15, 15, 0.9)
        # Overlap area = 5*5=25, union = 100+100-25=175
        expected = 25.0 / 175.0
        assert abs(iou(a, b) - expected) < 1e-6


class TestCenterDistance:
    def test_same_center(self):
        a = make_det(50, 50)
        b = make_det(50, 50)
        assert center_distance(a, b) == 0.0

    def test_known_distance(self):
        a = make_det(0, 0)
        b = make_det(3, 4)
        assert abs(center_distance(a, b) - 5.0) < 1e-6


class TestNMS:
    def test_no_suppression(self):
        dets = [make_det(0, 0, conf=0.9), make_det(200, 200, conf=0.8)]
        result = nms(dets, iou_threshold=0.5)
        assert len(result) == 2

    def test_suppresses_overlap(self):
        dets = [
            make_det(100, 100, 50, 50, conf=0.9),
            make_det(105, 105, 50, 50, conf=0.7),
        ]
        result = nms(dets, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_empty_input(self):
        assert nms([], 0.5) == []


class TestTargetTracker:
    def _cfg(self):
        return {
            "lost_frames_threshold": 3,
            "assoc_mode": "iou",
            "assoc_iou_threshold": 0.2,
            "assoc_distance_threshold": 100,
            "switch_penalty": 50.0,
        }

    def test_selects_closest_to_center(self):
        tracker = TargetTracker(self._cfg)
        dets = [
            make_det(300, 300),  # far from center
            make_det(160, 160),  # closer to center (160,160)
            make_det(320, 320),  # farthest
        ]
        roi_cx, roi_cy = 160, 160
        locked = tracker.update(dets, roi_cx, roi_cy)
        assert locked is not None
        # Should select the detection at (160, 160)
        assert abs(locked.detection.cx - 160) < 1

    def test_holds_target(self):
        tracker = TargetTracker(self._cfg)
        roi_cx, roi_cy = 160, 160

        # Frame 1: lock onto target at (160, 160)
        d1 = make_det(160, 160)
        locked = tracker.update([d1], roi_cx, roi_cy)
        assert locked is not None
        assert abs(locked.detection.cx - 160) < 1

        # Frame 2: target moved slightly, still associated
        d2 = make_det(165, 162)
        locked = tracker.update([d2], roi_cx, roi_cy)
        assert locked is not None
        assert abs(locked.detection.cx - 165) < 1
        assert locked.lost_count == 0

    def test_loses_target_after_threshold(self):
        tracker = TargetTracker(self._cfg)
        roi_cx, roi_cy = 160, 160

        # Frame 1: lock
        locked = tracker.update([make_det(160, 160)], roi_cx, roi_cy)
        assert locked is not None

        # Frames 2,3,4: no detections at all → lost_count increases
        for i in range(3):
            locked = tracker.update([], roi_cx, roi_cy)

        # After 3 frames (= lost_frames_threshold), target should be lost
        assert locked is None

    def test_reacquires_after_loss(self):
        tracker = TargetTracker(self._cfg)
        roi_cx, roi_cy = 160, 160

        # Lock, then lose
        tracker.update([make_det(160, 160)], roi_cx, roi_cy)
        for _ in range(3):
            tracker.update([], roi_cx, roi_cy)

        # New detection appears
        new_det = make_det(200, 200)
        locked = tracker.update([new_det], roi_cx, roi_cy)
        assert locked is not None
        assert abs(locked.detection.cx - 200) < 1

    def test_does_not_switch_while_locked(self):
        tracker = TargetTracker(self._cfg)
        roi_cx, roi_cy = 160, 160

        # Lock onto target A at (160, 160)
        d_a = make_det(160, 160, 50, 100)
        locked = tracker.update([d_a], roi_cx, roi_cy)
        assert locked is not None

        # Frame 2: A moved slightly, B appears far from A but closer to center
        d_a2 = make_det(165, 162, 50, 100)  # target A moved slightly
        d_b = make_det(60, 60, 50, 100)     # target B, far from A (no IoU overlap)
        locked = tracker.update([d_a2, d_b], roi_cx, roi_cy)
        assert locked is not None
        # Should still track A (via IoU association), not switch to B
        assert abs(locked.detection.cx - 165) < 1

    def test_manual_reset(self):
        tracker = TargetTracker(self._cfg)
        roi_cx, roi_cy = 160, 160
        tracker.update([make_det(160, 160)], roi_cx, roi_cy)
        tracker.reset()
        assert tracker.locked_target is None

    def test_distance_mode(self):
        cfg = self._cfg()
        cfg["assoc_mode"] = "distance"
        cfg["assoc_distance_threshold"] = 50
        tracker = TargetTracker(lambda: cfg)
        roi_cx, roi_cy = 160, 160

        tracker.update([make_det(160, 160)], roi_cx, roi_cy)
        # Move within distance threshold
        locked = tracker.update([make_det(180, 170)], roi_cx, roi_cy)
        assert locked is not None
        assert locked.lost_count == 0

        # Move beyond threshold
        locked = tracker.update([make_det(500, 500)], roi_cx, roi_cy)
        assert locked is not None
        assert locked.lost_count == 1
