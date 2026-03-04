"""
TensorRT inference backend.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from app_logging.logger import get_logger

log = get_logger()


def _is_trt_available() -> bool:
    try:
        import tensorrt
        return True
    except ImportError:
        return False


class TensorRTBackend:
    """Run inference on a pre-built TensorRT engine."""

    def __init__(self) -> None:
        self._engine = None
        self._context = None
        self._bindings: list = []
        self._input_shape: tuple = ()
        self._output_shapes: list[tuple] = []
        self._stream = None
        self._d_inputs: list = []
        self._d_outputs: list = []
        self._h_outputs: list[np.ndarray] = []
        self._loaded = False
        self._engine_path = ""

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def engine_path(self) -> str:
        return self._engine_path

    def load(self, engine_path: str, warmup_runs: int = 3) -> None:
        if not _is_trt_available():
            raise RuntimeError("TensorRT not available")

        import tensorrt as trt

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise RuntimeError("pycuda required for TensorRT inference. "
                               "pip install pycuda")

        self._engine_path = engine_path
        logger = trt.Logger(trt.Logger.WARNING)

        log.info(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        # Allocate buffers
        self._d_inputs = []
        self._d_outputs = []
        self._h_outputs = []
        self._bindings = []

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(size)
            self._bindings.append(int(device_mem))

            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._d_inputs.append(device_mem)
                self._input_shape = tuple(shape)
            else:
                self._d_outputs.append(device_mem)
                host_mem = np.empty(shape, dtype=dtype)
                self._h_outputs.append(host_mem)
                self._output_shapes.append(tuple(shape))

        self._loaded = True
        log.info(f"TensorRT engine loaded. Input shape: {self._input_shape}")

        # Warmup
        if warmup_runs > 0:
            dummy = np.random.rand(*self._input_shape).astype(np.float32)
            for _ in range(warmup_runs):
                self._infer(dummy)
            log.info("TensorRT warmup complete")

    def predict(self, frame: np.ndarray, conf: float = 0.45,
                input_size: int = 640) -> np.ndarray | None:
        """
        Preprocess frame, run TRT inference, return raw output.
        Returns np.ndarray of detections [N, 6] (x1,y1,x2,y2,conf,cls).
        """
        if not self._loaded:
            return None

        # Preprocess: resize, normalize, NCHW, batch
        import cv2
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (input_size, input_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)         # add batch
        img = np.ascontiguousarray(img)

        raw = self._infer(img)
        if raw is None:
            return None

        # Post-process raw TRT output to [N, 6]
        return self._decode_output(raw, conf, h, w, input_size)

    def _infer(self, input_data: np.ndarray) -> np.ndarray | None:
        try:
            import pycuda.driver as cuda
        except ImportError:
            return None

        # Copy input to device
        cuda.memcpy_htod_async(self._d_inputs[0], input_data, self._stream)

        # Set tensor addresses
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            self._context.set_tensor_address(name, self._bindings[i])

        # Run inference
        self._context.execute_async_v3(stream_handle=self._stream.handle)

        # Copy outputs back
        for i, d_out in enumerate(self._d_outputs):
            cuda.memcpy_dtoh_async(self._h_outputs[i], d_out, self._stream)
        self._stream.synchronize()

        return self._h_outputs[0].copy() if self._h_outputs else None

    def _decode_output(self, raw: np.ndarray, conf_thresh: float,
                       orig_h: int, orig_w: int,
                       input_size: int) -> np.ndarray:
        """Decode YOLOv8 TRT output to [N, 6]."""
        # YOLOv8 output: [1, 84, 8400] -> transpose to [8400, 84]
        if raw.ndim == 3:
            raw = raw[0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T  # now [8400, 84]

        # Columns: cx, cy, w, h, class_scores[80]
        cx = raw[:, 0]
        cy = raw[:, 1]
        w = raw[:, 2]
        h = raw[:, 3]
        scores = raw[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]

        mask = confs >= conf_thresh
        if not np.any(mask):
            return np.empty((0, 6), dtype=np.float32)

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        # Scale to original image size
        sx = orig_w / input_size
        sy = orig_h / input_size

        x1 = (cx - w / 2) * sx
        y1 = (cy - h / 2) * sy
        x2 = (cx + w / 2) * sx
        y2 = (cy + h / 2) * sy

        return np.stack([x1, y1, x2, y2, confs, class_ids.astype(np.float32)], axis=1)

    def unload(self) -> None:
        self._engine = None
        self._context = None
        self._loaded = False
        self._d_inputs.clear()
        self._d_outputs.clear()
        self._h_outputs.clear()
        self._bindings.clear()
        log.info("TensorRT engine unloaded")
