"""
Main entry point: orchestrates capture → inference → tracking → aim pipeline.

Thread A: capture loop (ScreenCapture)
Thread B: inference + tracking + aim (Pipeline)
Thread C: GUI (DearPyGui, main thread)
Thread D: overlay (Win32 window)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import cv2

from config.config_manager import ConfigManager
from app_logging.logger import get_logger
from capture.capture import ScreenCapture, get_monitor_list
from inference.torch_backend import TorchBackend
from inference.tensorrt_backend import TensorRTBackend
from detection.postprocess import postprocess, Detection
from tracking.tracker import TargetTracker
from aim.mouse_control import AimController
from filters.smoothing import FilterPair
from telemetry.telemetry import Telemetry
from overlay.overlay import OverlayWindow
from gui.app_gui import AppGUI

log = get_logger()


def _check_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_trt() -> bool:
    try:
        import tensorrt
        return True
    except ImportError:
        return False


def _get_monitor_rect(monitors: list[dict], monitor_index: int) -> tuple[int, int, int, int]:
    if 0 <= monitor_index < len(monitors):
        mon = monitors[monitor_index]
        return (mon["left"], mon["top"], mon["width"], mon["height"])
    return (0, 0, 1920, 1080)


class Pipeline:
    """Inference + tracking + aim pipeline running in a background thread."""

    def __init__(self, cfg: ConfigManager, capture: ScreenCapture,
                 telemetry: Telemetry, overlay: OverlayWindow,
                 gui: AppGUI) -> None:
        self._cfg = cfg
        self._capture = capture
        self._telemetry = telemetry
        self._overlay = overlay
        self._gui = gui

        self._torch_backend = TorchBackend()
        self._trt_backend = TensorRTBackend()
        self._tracker = TargetTracker(lambda: cfg.section("tracking"))
        self._aim = AimController(lambda: cfg.section("aim"))

        self._running = False
        self._thread: threading.Thread | None = None
        self._last_detections: list[Detection] = []
        self._last_locked = None
        self._last_error = (0.0, 0.0)
        self._model_loaded = False
        self._model_error: str | None = None
        self._model_load_lock = threading.Lock()
        self._last_frame_time = 0.0

    @property
    def detections(self) -> list[Detection]:
        return self._last_detections

    @property
    def locked_target(self):
        return self._last_locked

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_error(self) -> str | None:
        return self._model_error

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._model_loaded = False
        self._model_error = None
        self._last_detections = []
        self._last_locked = None
        self._last_frame_time = 0.0
        threading.Thread(target=self._load_model, daemon=True, name="model-loader").start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pipeline")
        self._thread.start()
        log.info("Pipeline thread started")

    def stop(self) -> None:
        if not self._running and self._thread is None:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        self._last_detections = []
        self._last_locked = None
        self._last_error = (0.0, 0.0)
        self._model_loaded = False
        self._model_error = None
        self._last_frame_time = 0.0
        with self._model_load_lock:
            self._torch_backend.unload()
            self._trt_backend.unload()
        self._tracker.reset()
        self._aim.reset_filter()
        log.info("Pipeline thread stopped")

    def reload_model(self) -> None:
        """Reload model from config (called from GUI thread)."""
        self._model_loaded = False
        self._model_error = None
        threading.Thread(target=self._load_model, daemon=True).start()

    def reset_target(self) -> None:
        self._tracker.reset()

    def export_onnx(self) -> None:
        from inference.onnx_export import export_to_onnx, validate_onnx
        try:
            pt_path = self._cfg.get("inference.model_path", "yolov8n.pt")
            opset = self._cfg.get("export.onnx_opset", 17)
            dynamic = self._cfg.get("export.onnx_dynamic", False)
            input_size = self._cfg.get("inference.input_size", 640)

            log.info(f"Exporting {pt_path} to ONNX...")
            onnx_path = export_to_onnx(
                pt_path, opset=opset, input_size=input_size, dynamic=dynamic,
            )
            validate_onnx(onnx_path)

            # Sanity check
            self._sanity_check_onnx(onnx_path, input_size)
            log.info("ONNX export complete!")
        except Exception as e:
            log.error(f"ONNX export failed: {e}")

    def build_trt(self) -> None:
        from inference.tensorrt_builder import build_engine
        try:
            pt_path = self._cfg.get("inference.model_path", "yolov8n.pt")
            onnx_path = str(Path(pt_path).with_suffix(".onnx"))
            if not Path(onnx_path).exists():
                log.info("ONNX not found, exporting first...")
                self.export_onnx()

            engine_path = self._cfg.get("export.trt_engine_path", "")
            if not engine_path:
                engine_path = str(Path(onnx_path).with_suffix(".engine"))

            fp16 = self._cfg.get("export.trt_fp16", True)
            workspace = self._cfg.get("export.trt_workspace_mb", 4096)

            log.info(f"Building TensorRT engine from {onnx_path}...")
            result = build_engine(
                onnx_path, engine_path, fp16=fp16, workspace_mb=workspace,
            )
            self._cfg.set("inference.engine_path", result)
            self._cfg.set("export.trt_engine_path", result)
            log.info("TensorRT build complete!")
        except Exception as e:
            log.error(f"TensorRT build failed: {e}")

    def _load_model(self) -> None:
        with self._model_load_lock:
            backend = self._cfg.get("inference.backend", "torch")
            self._model_loaded = False
            self._model_error = None
            self._torch_backend.unload()
            self._trt_backend.unload()

            try:
                if backend == "tensorrt":
                    engine_path = self._cfg.get("inference.engine_path", "")
                    if not engine_path:
                        engine_path = self._cfg.get("export.trt_engine_path", "")
                    if not engine_path or not Path(engine_path).exists():
                        raise FileNotFoundError(
                            "TensorRT engine не найден. Выберите .engine файл или соберите его."
                        )
                    self._trt_backend.load(
                        engine_path,
                        warmup_runs=self._cfg.get("inference.warmup_runs", 3),
                    )
                    self._model_loaded = True
                    log.info("TensorRT backend ready")
                else:
                    model_path = self._cfg.get("inference.model_path", "yolov8n.pt")
                    device = self._cfg.get("inference.device", "auto")
                    fp16 = self._cfg.get("inference.fp16", True)
                    input_size = self._cfg.get("inference.input_size", 640)
                    warmup = self._cfg.get("inference.warmup_runs", 3)

                    self._torch_backend.load(
                        model_path, device=device, fp16=fp16,
                        input_size=input_size, warmup_runs=warmup,
                    )
                    self._model_loaded = True
                    log.info("Torch backend ready")
            except Exception as e:
                self._model_error = str(e)
                log.error(f"Model load failed: {e}")
                self._model_loaded = False

    def _loop(self) -> None:
        while self._running:
            t_total_start = time.perf_counter()

            if not self._model_loaded:
                time.sleep(0.1)
                continue

            # ── Capture ──────────────────────────────────────────────
            t0 = time.perf_counter()
            frame, frame_time = self._capture.grab()
            self._telemetry.record("capture", (time.perf_counter() - t0) * 1000)

            if frame is None:
                time.sleep(0.005)
                continue
            if frame_time <= self._last_frame_time:
                time.sleep(0.001)
                continue
            self._last_frame_time = frame_time

            # ── Preprocess ───────────────────────────────────────────
            t0 = time.perf_counter()
            cap_cfg = self._cfg.section("capture")
            roi_w = cap_cfg.get("roi_width", 640)
            roi_h = cap_cfg.get("roi_height", 640)
            roi_cx = roi_w / 2.0
            roi_cy = roi_h / 2.0

            downscale = self._cfg.get("detection.downscale", 1.0)
            input_frame = frame
            if 0 < downscale < 1.0:
                new_w = max(1, int(frame.shape[1] * downscale))
                new_h = max(1, int(frame.shape[0] * downscale))
                input_frame = cv2.resize(
                    frame,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA,
                )
            self._telemetry.record("preprocess", (time.perf_counter() - t0) * 1000)

            # ── Inference ────────────────────────────────────────────
            t0 = time.perf_counter()
            backend = self._cfg.get("inference.backend", "torch")
            det_cfg = self._cfg.section("detection")
            conf = det_cfg.get("confidence", 0.45)
            iou_t = det_cfg.get("nms_iou", 0.45)

            raw_results = None
            if backend == "tensorrt" and self._trt_backend.loaded:
                raw_results = self._trt_backend.predict(
                    input_frame, conf=conf,
                    input_size=self._cfg.get("inference.input_size", 640),
                )
            elif self._torch_backend.loaded:
                raw_results = self._torch_backend.predict(
                    input_frame, conf=conf, iou=iou_t, classes=[0],
                )
            self._telemetry.record("inference", (time.perf_counter() - t0) * 1000)

            # ── Postprocess ──────────────────────────────────────────
            t0 = time.perf_counter()
            if raw_results is not None:
                detections = postprocess(raw_results, det_cfg)
                # Scale back if downscaled
                if 0 < downscale < 1.0 and detections:
                    scale = 1.0 / downscale
                    for d in detections:
                        d.x1 *= scale
                        d.y1 *= scale
                        d.x2 *= scale
                        d.y2 *= scale
            else:
                detections = []
            self._last_detections = detections
            self._telemetry.record("postprocess", (time.perf_counter() - t0) * 1000)

            # ── Tracking ─────────────────────────────────────────────
            t0 = time.perf_counter()
            locked = self._tracker.update(detections, roi_cx, roi_cy)
            self._last_locked = locked
            self._telemetry.record("tracking", (time.perf_counter() - t0) * 1000)

            # ── Aim ──────────────────────────────────────────────────
            t0 = time.perf_counter()
            dx, dy = 0.0, 0.0
            if locked is not None:
                d = locked.detection
                aim_x, aim_y = self._aim.compute_aim_point(
                    d.cx, d.cy, d.w, d.h,
                )
                filter_cfg = self._cfg.section("filters")
                self._aim.configure_filter(filter_cfg)
                dx, dy = self._aim.compute_and_move(
                    aim_x, aim_y, roi_cx, roi_cy, filter_cfg,
                )
            else:
                self._aim.reset_filter()
            self._last_error = (dx, dy)
            self._telemetry.record("aim", (time.perf_counter() - t0) * 1000)

            # ── Overlay data ─────────────────────────────────────────
            t0 = time.perf_counter()
            # Get monitor center for overlay coordinates
            monitors = self._capture.monitors
            mon_idx = cap_cfg.get("monitor_index", 0)
            if mon_idx < len(monitors):
                mon = monitors[mon_idx]
                scr_cx = mon["left"] + mon["width"] // 2
                scr_cy = mon["top"] + mon["height"] // 2
                ov_off_x = scr_cx - roi_w // 2
                ov_off_y = scr_cy - roi_h // 2
            else:
                ov_off_x, ov_off_y = 0, 0
                scr_cx, scr_cy = roi_cx, roi_cy

            # Shift detection coords from ROI-local to screen
            ov_dets = []
            for det in detections:
                from detection.postprocess import Detection as Det
                ov_dets.append(Det(
                    det.x1 + ov_off_x, det.y1 + ov_off_y,
                    det.x2 + ov_off_x, det.y2 + ov_off_y,
                    det.confidence, det.class_id,
                ))

            ov_locked = None
            if locked:
                from detection.postprocess import Detection as Det
                from tracking.tracker import TrackedTarget
                d = locked.detection
                ov_locked = TrackedTarget(
                    detection=Det(
                        d.x1 + ov_off_x, d.y1 + ov_off_y,
                        d.x2 + ov_off_x, d.y2 + ov_off_y,
                        d.confidence, d.class_id,
                    )
                )

            # Build metrics text
            fps = self._telemetry.get_fps()
            inf_m = self._telemetry.get("inference")
            tot_m = self._telemetry.get("total")
            if backend == "tensorrt" and self._trt_backend.loaded:
                device_name = "TensorRT"
            elif self._torch_backend.loaded:
                device_name = str(self._torch_backend.device)
            else:
                device_name = "н/д"
            metrics_text = (
                f"FPS: захват={fps['capture']:.0f} инференс={fps['inference']:.0f} "
                f"всего={fps['total']:.0f}\n"
                f"Инференс: {inf_m.current_ms:.1f}мс ср={inf_m.avg_ms:.1f} "
                f"p95={inf_m.p95_ms:.1f}\n"
                f"Всего: {tot_m.current_ms:.1f}мс  Устройство: {device_name}  "
                f"Цель: {'ЗАХВАЧЕНА' if locked else 'НЕТ'}"
            )

            overlay_cfg = self._cfg.section("overlay")
            if overlay_cfg.get("enabled", True):
                self._overlay.set_config(overlay_cfg)
                self._overlay.update_data({
                    "detections": ov_dets,
                    "locked_target": ov_locked,
                    "roi_cx": scr_cx,
                    "roi_cy": scr_cy,
                    "error": self._last_error,
                    "metrics_text": metrics_text,
                })

            self._telemetry.record("overlay", (time.perf_counter() - t0) * 1000)

            # ── Total ────────────────────────────────────────────────
            total_ms = (time.perf_counter() - t_total_start) * 1000
            self._telemetry.record("total", total_ms)

    def _sanity_check_onnx(self, onnx_path: str, input_size: int) -> None:
        """Run a quick inference to verify ONNX export."""
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path)
            inp_name = sess.get_inputs()[0].name
            dummy = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
            outputs = sess.run(None, {inp_name: dummy})
            if outputs and outputs[0] is not None and outputs[0].size > 0:
                log.info(f"ONNX sanity check passed. Output shape: {outputs[0].shape}")
            else:
                log.warn("ONNX sanity check: empty output")
        except Exception as e:
            log.warn(f"ONNX sanity check failed: {e}")


class HotkeyManager:
    """Listens for hotkeys in a background thread."""

    def __init__(self, cfg: ConfigManager, pipeline: Pipeline,
                 overlay: OverlayWindow) -> None:
        self._cfg = cfg
        self._pipeline = pipeline
        self._overlay = overlay
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True, name="hotkeys")
        self._thread.start()
        log.info("Hotkey listener started")

    def stop(self) -> None:
        if not self._running and self._thread is None:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        log.info("Hotkey listener stopped")

    def _listen(self) -> None:
        try:
            from pynput import keyboard, mouse
        except ImportError:
            log.warn("pynput not installed, hotkeys disabled")
            self._running = False
            return

        aim_cfg = self._cfg.section("aim")
        activation_key = aim_cfg.get("activation_key", "x2")

        pressed_keys: set[str] = set()

        def on_key_press(key):
            if not self._running:
                return False
            try:
                k = key.char if hasattr(key, 'char') and key.char else key.name
            except AttributeError:
                k = str(key)

            pressed_keys.add(k.lower())
            self._handle_hotkey(k.lower(), pressed=True)

        def on_key_release(key):
            if not self._running:
                return False
            try:
                k = key.char if hasattr(key, 'char') and key.char else key.name
            except AttributeError:
                k = str(key)
            pressed_keys.discard(k.lower())
            self._handle_hotkey(k.lower(), pressed=False)

        def on_mouse_click(x, y, button, pressed):
            if not self._running:
                return False
            btn_name = button.name if hasattr(button, 'name') else str(button)
            self._handle_hotkey(btn_name.lower(), pressed=pressed)

        kl = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        ml = mouse.Listener(on_click=on_mouse_click)
        kl.start()
        ml.start()

        while self._running:
            time.sleep(0.1)

        kl.stop()
        ml.stop()

    def _handle_hotkey(self, key: str, pressed: bool) -> None:
        gui_cfg = self._cfg.section("gui")

        # Hold-to-activate for aim
        act_key = self._cfg.get("aim.activation_key", "x2").lower()
        if key == act_key:
            self._pipeline._aim.set_key_held(pressed)

        if not pressed:
            return

        # Toggle mouse control
        hk = gui_cfg.get("hotkey_toggle_mouse", "f6").lower()
        if key == hk:
            cur = self._cfg.get("aim.enabled", False)
            self._cfg.set("aim.enabled", not cur)
            log.info(f"Mouse control: {'ON' if not cur else 'OFF'}")

        # Toggle overlay
        hk = gui_cfg.get("hotkey_toggle_overlay", "f7").lower()
        if key == hk:
            cur = self._cfg.get("overlay.enabled", True)
            self._cfg.set("overlay.enabled", not cur)
            self._overlay.set_visible(not cur)
            log.info(f"Overlay: {'ON' if not cur else 'OFF'}")

        # Lock/Unlock target
        hk = gui_cfg.get("hotkey_lock_target", "").lower()
        if hk and key == hk:
            self._pipeline.reset_target()
            log.info("Target reset via hotkey")

        # Pause capture
        hk = gui_cfg.get("hotkey_pause_capture", "f8").lower()
        if key == hk:
            cur = self._cfg.get("capture.paused", False)
            self._cfg.set("capture.paused", not cur)
            log.info(f"Capture: {'PAUSED' if not cur else 'RESUMED'}")


class RuntimeController:
    """Starts and stops capture, inference and overlay from the GUI."""

    def __init__(self, cfg: ConfigManager, capture: ScreenCapture,
                 pipeline: Pipeline, hotkey_mgr: HotkeyManager,
                 overlay: OverlayWindow, gui: AppGUI) -> None:
        self._cfg = cfg
        self._capture = capture
        self._pipeline = pipeline
        self._hotkey_mgr = hotkey_mgr
        self._overlay = overlay
        self._gui = gui
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._gui.set_runtime_state(True)
        try:
            self._capture.start()
            self._pipeline.start()
            self._hotkey_mgr.start()
            mon_rect = _get_monitor_rect(
                self._capture.monitors,
                self._cfg.get("capture.monitor_index", 0),
            )
            self._overlay.start(mon_rect)
            self._overlay.set_visible(self._cfg.get("overlay.enabled", True))
            log.info("Runtime started by user")
        except Exception as e:
            log.error(f"Failed to start runtime: {e}")
            self.stop()

    def stop(self) -> None:
        if not self._running:
            self._gui.set_runtime_state(False)
            return

        self._hotkey_mgr.stop()
        self._pipeline.stop()
        self._capture.stop()
        self._overlay.stop()
        self._running = False
        self._gui.set_runtime_state(False)
        log.info("Runtime stopped by user")


def main() -> None:
    log.info("Application starting...")

    # ── Config ───────────────────────────────────────────────────────
    cfg = ConfigManager()
    log.info("Config loaded")

    # ── Check capabilities & auto-configure device ──────────────────
    cuda_available = _check_cuda()
    trt_available = _check_trt()
    log.info(f"CUDA: {cuda_available}, TensorRT: {trt_available}")

    # Auto-detect: try CUDA first, fall back to CPU
    device_cfg = cfg.get("inference.device", "auto")
    if device_cfg == "auto":
        if cuda_available:
            cfg.set("inference.device", "cuda")
            cfg.set("inference.fp16", True)
            log.info("Авто-выбор устройства: CUDA (GPU)")
        else:
            cfg.set("inference.device", "cpu")
            cfg.set("inference.fp16", False)
            log.info("Авто-выбор устройства: CPU (GPU недоступна)")
    elif device_cfg == "cuda" and not cuda_available:
        log.warn("CUDA не найдена, переключение на CPU")
        cfg.set("inference.device", "cpu")
        cfg.set("inference.fp16", False)

    # Auto-detect backend: if TRT selected but not available, fall back to torch
    backend_cfg = cfg.get("inference.backend", "torch")
    if backend_cfg == "tensorrt" and not trt_available:
        log.warn("TensorRT не найден, переключение на PyTorch")
        cfg.set("inference.backend", "torch")

    # ── Monitors ─────────────────────────────────────────────────────
    monitors = get_monitor_list()
    log.info(f"Monitors: {len(monitors)}")

    # ── Components ───────────────────────────────────────────────────
    telemetry = Telemetry(window_sec=5.0)
    capture = ScreenCapture(lambda: cfg.section("capture"))
    overlay = OverlayWindow()

    # ── GUI (must be set up before pipeline reference) ───────────────
    gui = AppGUI(
        cfg=cfg,
        capture_monitors=monitors,
        cuda_available=cuda_available,
        trt_available=trt_available,
    )

    # ── Pipeline ─────────────────────────────────────────────────────
    pipeline = Pipeline(cfg, capture, telemetry, overlay, gui)

    # Wire up GUI callbacks
    gui._on_model_reload = pipeline.reload_model
    gui._on_export_onnx = pipeline.export_onnx
    gui._on_build_trt = pipeline.build_trt

    # ── Hotkeys ──────────────────────────────────────────────────────
    hotkey_mgr = HotkeyManager(cfg, pipeline, overlay)
    runtime = RuntimeController(cfg, capture, pipeline, hotkey_mgr, overlay, gui)

    # ── Start everything ─────────────────────────────────────────────
    gui._on_start_requested = runtime.start
    gui._on_stop_requested = runtime.stop
    gui.setup()
    log.info("Application ready. Waiting for user action.")

    # ── Main loop (GUI thread) ───────────────────────────────────────
    gui_update_interval = 0.1
    last_gui_update = 0.0
    last_preview_update = 0.0
    last_preview_frame_time = 0.0
    stage_labels = [
        ("capture", "Захват"),
        ("preprocess", "Подготовка"),
        ("inference", "Инференс"),
        ("postprocess", "Постобработка"),
        ("tracking", "Трекинг"),
        ("aim", "Наведение"),
        ("overlay", "Оверлей"),
        ("total", "Всего"),
    ]

    try:
        while gui.render_frame():
            gui.frame()

            now = time.time()
            if now - last_gui_update >= gui_update_interval:
                last_gui_update = now

                # Update telemetry display
                if not runtime.is_running:
                    gui.update_telemetry("Остановлено.\nНажмите «Запустить», чтобы начать захват.")
                    gui.update_status("Остановлено. Нажмите «Запустить».")
                    gui.clear_preview()
                    gui.update_logs()
                    continue

                fps = telemetry.get_fps()
                lines = [
                    f"FPS: захват {fps['capture']:.0f}  "
                    f"инференс {fps['inference']:.0f}  "
                    f"всего {fps['total']:.0f}",
                ]
                for stage_key, stage_name in stage_labels:
                    metric = telemetry.get(stage_key)
                    lines.append(
                        f"  {stage_name:12s}: {metric.current_ms:6.1f} мс  "
                        f"ср={metric.avg_ms:6.1f}  p95={metric.p95_ms:6.1f}",
                    )
                gui.update_telemetry("\n".join(lines))

                locked = pipeline.locked_target
                n_det = len(pipeline.detections)
                backend = cfg.get("inference.backend", "torch")
                device = "GPU" if cfg.get("inference.device") != "cpu" else "CPU"
                if pipeline.model_error:
                    runtime_status = f"Ошибка модели: {pipeline.model_error}"
                elif not pipeline.model_loaded:
                    runtime_status = "Загрузка модели"
                else:
                    runtime_status = "Работает"
                status = (
                    f"{backend.upper()} ({device}) | "
                    f"Детекций: {n_det} | "
                    f"Цель: {'захвачена' if locked else 'нет'} | "
                    f"{runtime_status}"
                )
                gui.update_status(status)

                mon_rect = _get_monitor_rect(
                    capture.monitors,
                    cfg.get("capture.monitor_index", 0),
                )
                overlay.set_monitor_rect(mon_rect)
                overlay.set_visible(cfg.get("overlay.enabled", True))
                gui.update_logs()

            preview_max_fps = int(cfg.get("preview.max_fps", 20) or 0)
            preview_update_interval = 1.0 / preview_max_fps if preview_max_fps > 0 else 0.0
            if preview_update_interval == 0.0 or now - last_preview_update >= preview_update_interval:
                last_preview_update = now

                if not runtime.is_running:
                    last_preview_frame_time = 0.0
                    gui.clear_preview()
                    continue

                if not cfg.get("preview.enabled", True):
                    gui.clear_preview()
                    continue

                frame, frame_time = capture.grab()
                if frame is not None and frame_time > last_preview_frame_time:
                    last_preview_frame_time = frame_time
                    gui.update_preview(frame, pipeline.detections, pipeline.locked_target)


    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down...")
        runtime.stop()
        cfg.save_now()
        gui.cleanup()
        log.info("Application stopped")


if __name__ == "__main__":
    main()
