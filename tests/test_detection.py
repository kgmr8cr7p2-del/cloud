"""
Tests for detection postprocessing.
"""

import numpy as np
import pytest

from detection.postprocess import Detection, postprocess, nms, iou


class TestDetection:
    def test_properties(self):
        d = Detection(10, 20, 60, 120, 0.9)
        assert d.cx == 35.0
        assert d.cy == 70.0
        assert d.w == 50.0
        assert d.h == 100.0
        assert d.area == 5000.0

    def test_iou_self(self):
        d = Detection(0, 0, 100, 100, 0.9)
        assert abs(iou(d, d) - 1.0) < 1e-6


class TestPostprocess:
    def test_numpy_input(self):
        # [x1, y1, x2, y2, conf, class_id]
        raw = np.array([
            [10, 20, 60, 120, 0.9, 0],  # person, high conf
            [10, 20, 60, 120, 0.1, 0],  # person, low conf (filtered)
            [10, 20, 60, 120, 0.9, 1],  # not person (filtered)
        ], dtype=np.float32)
        cfg = {
            "confidence": 0.5,
            "nms_iou": 0.5,
            "min_bbox_width": 5,
            "min_bbox_height": 5,
            "max_detections": 10,
        }
        dets = postprocess(raw, cfg)
        assert len(dets) == 1
        assert dets[0].confidence == pytest.approx(0.9, abs=1e-6)

    def test_empty_input(self):
        raw = np.empty((0, 6), dtype=np.float32)
        cfg = {"confidence": 0.5, "nms_iou": 0.5,
               "min_bbox_width": 5, "min_bbox_height": 5,
               "max_detections": 10}
        assert postprocess(raw, cfg) == []

    def test_min_bbox_filter(self):
        raw = np.array([
            [10, 20, 15, 25, 0.9, 0],  # 5x5 — too small
        ], dtype=np.float32)
        cfg = {"confidence": 0.3, "nms_iou": 0.5,
               "min_bbox_width": 10, "min_bbox_height": 10,
               "max_detections": 10}
        assert postprocess(raw, cfg) == []

    def test_max_detections(self):
        rows = [[i * 100, 0, i * 100 + 50, 100, 0.9, 0] for i in range(20)]
        raw = np.array(rows, dtype=np.float32)
        cfg = {"confidence": 0.3, "nms_iou": 0.99,
               "min_bbox_width": 1, "min_bbox_height": 1,
               "max_detections": 5}
        dets = postprocess(raw, cfg)
        assert len(dets) == 5
