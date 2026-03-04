"""
Export .pt model to ONNX format.
"""

from __future__ import annotations

from pathlib import Path

from app_logging.logger import get_logger

log = get_logger()


def export_to_onnx(
    pt_path: str,
    output_path: str | None = None,
    opset: int = 17,
    input_size: int = 640,
    dynamic: bool = False,
    simplify: bool = True,
) -> str:
    """
    Export a YOLOv8 .pt to ONNX.
    Returns the path to the exported .onnx file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed")

    model = YOLO(pt_path)
    log.info(f"Exporting {pt_path} to ONNX (opset={opset}, "
             f"size={input_size}, dynamic={dynamic})")

    result_path = model.export(
        format="onnx",
        opset=opset,
        imgsz=input_size,
        dynamic=dynamic,
        simplify=simplify,
    )

    # ultralytics returns the export path
    result_str = str(result_path)
    if output_path and result_str != output_path:
        import shutil
        shutil.move(result_str, output_path)
        result_str = output_path

    log.info(f"ONNX exported: {result_str}")
    return result_str


def validate_onnx(onnx_path: str) -> bool:
    """Quick check that the ONNX file is valid."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        log.info(f"ONNX model valid: {onnx_path}")
        return True
    except Exception as e:
        log.error(f"ONNX validation failed: {e}")
        return False
