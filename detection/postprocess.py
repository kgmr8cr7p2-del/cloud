"""
Detection post-processing: filter by confidence, NMS, bbox size, person class only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PERSON_CLASS_ID = 0  # COCO class index for 'person'


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = PERSON_CLASS_ID

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0, self.w) * max(0, self.h)


def iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Detection, b: Detection) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: list[Detection] = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best, d) < iou_threshold]
    return keep


def postprocess(raw_results: Any, cfg: dict) -> list[Detection]:
    """
    Convert raw inference output to filtered Detection list.
    Expects ultralytics Results object or raw tensor.
    """
    conf_thresh = cfg.get("confidence", 0.45)
    iou_thresh = cfg.get("nms_iou", 0.45)
    min_w = cfg.get("min_bbox_width", 10)
    min_h = cfg.get("min_bbox_height", 10)
    max_dets = cfg.get("max_detections", 20)

    detections: list[Detection] = []

    try:
        # Handle ultralytics Results
        if hasattr(raw_results, 'boxes'):
            boxes = raw_results.boxes
            if boxes is None or len(boxes) == 0:
                return []
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], 'item') else int(boxes.cls[i])
                if cls_id != PERSON_CLASS_ID:
                    continue
                conf = float(boxes.conf[i].item()) if hasattr(boxes.conf[i], 'item') else float(boxes.conf[i])
                if conf < conf_thresh:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                d = Detection(x1, y1, x2, y2, conf, cls_id)
                if d.w >= min_w and d.h >= min_h:
                    detections.append(d)
        # Handle raw numpy array [N, 6] => x1,y1,x2,y2,conf,cls
        elif isinstance(raw_results, np.ndarray):
            for row in raw_results:
                cls_id = int(row[5])
                if cls_id != PERSON_CLASS_ID:
                    continue
                conf = float(row[4])
                if conf < conf_thresh:
                    continue
                d = Detection(float(row[0]), float(row[1]),
                              float(row[2]), float(row[3]), conf, cls_id)
                if d.w >= min_w and d.h >= min_h:
                    detections.append(d)
    except Exception:
        return []

    # Apply NMS
    detections = nms(detections, iou_thresh)

    # Limit count
    return detections[:max_dets]
