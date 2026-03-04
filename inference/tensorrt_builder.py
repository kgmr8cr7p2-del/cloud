"""
TensorRT engine builder: ONNX → TensorRT engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app_logging.logger import get_logger

log = get_logger()


def is_tensorrt_available() -> bool:
    try:
        import tensorrt
        return True
    except ImportError:
        return False


def build_engine(
    onnx_path: str,
    engine_path: str | None = None,
    fp16: bool = True,
    int8: bool = False,
    workspace_mb: int = 4096,
    verbose: bool = False,
) -> str:
    """
    Build a TensorRT engine from an ONNX model.
    Returns path to the .engine file.
    """
    if not is_tensorrt_available():
        raise RuntimeError("TensorRT is not installed. "
                           "Install tensorrt package for GPU acceleration.")

    import tensorrt as trt

    if engine_path is None:
        engine_path = str(Path(onnx_path).with_suffix(".engine"))

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    log.info(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parsing failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("FP16 enabled")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        log.info("INT8 enabled")

    log.info("Building TensorRT engine (this may take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    log.info(f"TensorRT engine saved: {engine_path}")
    return engine_path
