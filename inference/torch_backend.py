"""
PyTorch inference backend using Ultralytics YOLO.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from app_logging.logger import get_logger

log = get_logger()


class TorchBackend:
    """Load and run YOLOv8 .pt model via ultralytics."""

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cpu"
        self._model_path: str = ""
        self._input_size: int = 640
        self._fp16: bool = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def input_size(self) -> int:
        return self._input_size

    def load(self, model_path: str, device: str = "auto",
             fp16: bool = True, input_size: int = 640,
             warmup_runs: int = 3) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            log.error("ultralytics not installed")
            raise

        self._model_path = model_path
        self._input_size = input_size
        self._fp16 = fp16

        # Resolve device
        if device == "auto":
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

        log.info(f"Loading model: {model_path} on {self._device}")
        self._model = YOLO(model_path)

        if self._device == "cuda":
            self._model.to("cuda")
            log.info("Model moved to CUDA")

        # Warmup
        if warmup_runs > 0:
            log.info(f"Warming up ({warmup_runs} runs)...")
            dummy = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
            for _ in range(warmup_runs):
                self._model.predict(
                    dummy, imgsz=input_size, device=self._device,
                    half=fp16 and self._device == "cuda",
                    verbose=False,
                )
            log.info("Warmup complete")

    def predict(self, frame: np.ndarray, conf: float = 0.45,
                iou: float = 0.45, classes: list[int] | None = None,
                ) -> Any:
        """Run inference. Returns ultralytics Results[0]."""
        if self._model is None:
            return None

        results = self._model.predict(
            frame,
            imgsz=self._input_size,
            device=self._device,
            half=self._fp16 and self._device == "cuda",
            conf=conf,
            iou=iou,
            classes=classes or [0],
            verbose=False,
        )
        return results[0] if results else None

    def unload(self) -> None:
        self._model = None
        log.info("Torch model unloaded")
