"""
Main DearPyGui application GUI.
Tabs: Capture, Detection, Tracking, Aim, Filters, Overlay, Export, Hotkeys, Logs.
All changes apply instantly via shared ConfigManager.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

import dearpygui.dearpygui as dpg
import numpy as np
import cv2

from config.config_manager import ConfigManager
from gui.tooltips import get_tooltip
from app_logging.logger import get_logger

log = get_logger()


def _tip(parent: int | str, key: str) -> None:
    """Attach a Russian tooltip to a DearPyGui widget."""
    text = get_tooltip(key)
    if text:
        with dpg.tooltip(parent):
            dpg.add_text(text, wrap=350)


class AppGUI:
    """DearPyGui-based control panel."""

    def __init__(self, cfg: ConfigManager,
                 on_start_requested: Callable | None = None,
                 on_stop_requested: Callable | None = None,
                 on_model_reload: Callable | None = None,
                 on_export_onnx: Callable | None = None,
                 on_build_trt: Callable | None = None,
                 capture_monitors: list[dict] | None = None,
                 cuda_available: bool = False,
                 trt_available: bool = False) -> None:
        self._cfg = cfg
        self._on_start_requested = on_start_requested
        self._on_stop_requested = on_stop_requested
        self._on_model_reload = on_model_reload
        self._on_export_onnx = on_export_onnx
        self._on_build_trt = on_build_trt
        self._monitors = capture_monitors or []
        self._cuda = cuda_available
        self._trt = trt_available

        self._preview_texture_id: int | str | None = None
        self._preview_rgba: np.ndarray | None = None
        self._telemetry_text_id: int | str | None = None
        self._status_text_id: int | str | None = None
        self._log_text_id: int | str | None = None
        self._font_id: int | str | None = None
        self._start_button_id: int | str | None = None
        self._stop_button_id: int | str | None = None
        self._model_path_input_id: int | str | None = None
        self._engine_path_input_id: int | str | None = None
        self._runtime_running = False
        self._preview_width = 320
        self._preview_height = 320

    def setup(self) -> None:
        dpg.create_context()
        self._bind_cyrillic_font()
        dpg.create_viewport(
            title="Помощник наведения",
            width=820, height=720,
            resizable=True,
        )

        # Create preview texture
        self._preview_width = min(self._cfg.get("capture.roi_width", 640), 640)
        self._preview_height = min(self._cfg.get("capture.roi_height", 640), 640)
        tex_data = [0.0] * (self._preview_width * self._preview_height * 4)
        with dpg.texture_registry():
            self._preview_texture_id = dpg.add_dynamic_texture(
                width=self._preview_width,
                height=self._preview_height,
                default_value=tex_data,
            )
        self._preview_rgba = np.zeros((self._preview_height, self._preview_width, 4), dtype=np.float32)
        self._preview_rgba[:, :, 3] = 1.0
        self._build_file_dialogs()

        with dpg.window(label="Помощник наведения", tag="main_window"):
            self._build_status_bar()
            with dpg.tab_bar():
                self._build_capture_tab()
                self._build_detection_tab()
                self._build_tracking_tab()
                self._build_aim_tab()
                self._build_filters_tab()
                self._build_overlay_tab()
                self._build_export_tab()
                self._build_hotkeys_tab()
                self._build_logs_tab()
            self._build_telemetry_panel()
            self._build_preview_panel()
            self._build_config_buttons()

        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()
        self.set_runtime_state(False)
        dpg.show_viewport()

    def render_frame(self) -> bool:
        """Call from main loop. Returns False when viewport closed."""
        return dpg.is_dearpygui_running()

    def frame(self) -> None:
        dpg.render_dearpygui_frame()

    def cleanup(self) -> None:
        dpg.destroy_context()

    def clear_preview(self) -> None:
        if self._preview_texture_id is None or self._preview_rgba is None:
            return
        try:
            self._preview_rgba[:, :, :3] = 0.0
            dpg.set_value(self._preview_texture_id, self._preview_rgba.ravel())
        except Exception:
            pass

    def set_runtime_state(self, running: bool) -> None:
        self._runtime_running = running
        if self._start_button_id:
            if running:
                dpg.disable_item(self._start_button_id)
            else:
                dpg.enable_item(self._start_button_id)
        if self._stop_button_id:
            if running:
                dpg.enable_item(self._stop_button_id)
            else:
                dpg.disable_item(self._stop_button_id)

        self._set_status_color((100, 255, 100) if running else (255, 100, 100))
        if running:
            self.update_status("Запуск...")
        else:
            self.update_status("Остановлено. Нажмите «Запустить».")
            self.clear_preview()

    # ── Preview ──────────────────────────────────────────────────────

    def update_preview(self, frame: np.ndarray | None,
                       detections: list | None = None,
                       locked: Any = None) -> None:
        if frame is None or self._preview_texture_id is None or self._preview_rgba is None:
            return
        if not self._cfg.get("preview.enabled", True):
            self.clear_preview()
            return

        try:
            h, w = frame.shape[:2]
            preview_frame = frame
            if w != self._preview_width or h != self._preview_height:
                interpolation = cv2.INTER_AREA if w > self._preview_width or h > self._preview_height else cv2.INTER_LINEAR
                preview_frame = cv2.resize(
                    frame,
                    (self._preview_width, self._preview_height),
                    interpolation=interpolation,
                )
                h, w = preview_frame.shape[:2]

            # Draw bboxes on preview if configured
            if self._cfg.get("preview.draw_bboxes", True) and detections:
                if preview_frame is frame:
                    preview_frame = preview_frame.copy()
                sx = self._preview_width / self._cfg.get("capture.roi_width", 640)
                sy = self._preview_height / self._cfg.get("capture.roi_height", 640)
                for det in detections:
                    x1 = int(det.x1 * sx)
                    y1 = int(det.y1 * sy)
                    x2 = int(det.x2 * sx)
                    y2 = int(det.y2 * sy)
                    color = (0, 0, 255) if (locked and det is locked.detection) else (0, 255, 0)
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w - 1, x2), min(h - 1, y2)
                    if x2c > x1c and y2c > y1c:
                        cv2.rectangle(preview_frame, (x1c, y1c), (x2c, y2c), color, thickness=2)

            np.multiply(preview_frame, 1.0 / 255.0, out=self._preview_rgba[:, :, :3], casting="unsafe")
            dpg.set_value(self._preview_texture_id, self._preview_rgba.ravel())
        except Exception:
            pass

    # ── Telemetry ────────────────────────────────────────────────────

    def update_telemetry(self, text: str) -> None:
        if self._telemetry_text_id:
            try:
                dpg.set_value(self._telemetry_text_id, text)
            except Exception:
                pass

    def update_status(self, text: str) -> None:
        if self._status_text_id:
            try:
                dpg.set_value(self._status_text_id, text)
            except Exception:
                pass

    def _set_status_color(self, color: tuple[int, int, int]) -> None:
        if self._status_text_id:
            try:
                dpg.configure_item(self._status_text_id, color=color)
            except Exception:
                pass

    def update_logs(self) -> None:
        if self._log_text_id:
            entries = get_logger().get_entries(100)
            text = "\n".join(f"[{e.timestamp}] [{e.level}] {e.message}" for e in entries)
            try:
                dpg.set_value(self._log_text_id, text)
            except Exception:
                pass

    def _bind_cyrillic_font(self) -> None:
        font_path = self._find_font_path()
        if font_path is None:
            log.warn("GUI font with Cyrillic support not found; DearPyGui will use its default font")
            return

        try:
            with dpg.font_registry():
                with dpg.font(str(font_path), 18) as font_id:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
            self._font_id = font_id
            dpg.bind_font(font_id)
            log.info(f"Loaded GUI font: {font_path.name}")
        except Exception as e:
            log.warn(f"Failed to load GUI font {font_path}: {e}")
            self._font_id = None

    @staticmethod
    def _find_font_path() -> Path | None:
        import sys
        font_dirs: list[Path] = []

        if sys.platform == "win32":
            font_dirs.append(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts")
        elif sys.platform == "darwin":
            font_dirs.extend([
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                Path.home() / "Library/Fonts",
            ])
        else:
            font_dirs.extend([
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path.home() / ".local/share/fonts",
                Path.home() / ".fonts",
            ])

        font_names = ("segoeui.ttf", "tahoma.ttf", "arial.ttf", "calibri.ttf",
                       "verdana.ttf", "Arial.ttf", "DejaVuSans.ttf",
                       "NotoSans-Regular.ttf", "LiberationSans-Regular.ttf",
                       "Ubuntu-R.ttf", "FreeSans.ttf")

        for font_dir in font_dirs:
            if not font_dir.exists():
                continue
            for font_name in font_names:
                font_path = font_dir / font_name
                if font_path.exists():
                    return font_path
            # Search subdirectories (Linux fonts are often in subdirs)
            try:
                for font_name in font_names:
                    matches = list(font_dir.rglob(font_name))
                    if matches:
                        return matches[0]
            except PermissionError:
                continue

        return None

    def _build_file_dialogs(self) -> None:
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            width=700,
            height=420,
            tag="model_file_dialog",
            default_path=str(Path.cwd()),
            callback=self._on_model_file_selected,
        ):
            dpg.add_file_extension(".pt", color=(120, 255, 120, 255))
            dpg.add_file_extension(".onnx", color=(120, 180, 255, 255))
            dpg.add_file_extension(".*")

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            width=700,
            height=420,
            tag="engine_file_dialog",
            default_path=str(Path.cwd()),
            callback=self._on_engine_file_selected,
        ):
            dpg.add_file_extension(".engine", color=(120, 255, 120, 255))
            dpg.add_file_extension(".*")

    def _open_model_file_dialog(self) -> None:
        dpg.show_item("model_file_dialog")

    def _open_engine_file_dialog(self) -> None:
        dpg.show_item("engine_file_dialog")

    def _set_model_path(self, path: str) -> None:
        self._cfg.set("inference.model_path", path)
        if self._model_path_input_id and dpg.get_value(self._model_path_input_id) != path:
            dpg.set_value(self._model_path_input_id, path)

    def _set_engine_path(self, path: str) -> None:
        self._cfg.set("export.trt_engine_path", path)
        self._cfg.set("inference.engine_path", path)
        if self._engine_path_input_id and dpg.get_value(self._engine_path_input_id) != path:
            dpg.set_value(self._engine_path_input_id, path)

    def _on_model_file_selected(self, sender: int | str, app_data: dict, user_data: Any) -> None:
        path = app_data.get("file_path_name", "")
        if not path:
            return
        self._set_model_path(path)

    def _on_engine_file_selected(self, sender: int | str, app_data: dict, user_data: Any) -> None:
        path = app_data.get("file_path_name", "")
        if not path:
            return
        self._set_engine_path(path)

    # ── Tab builders ─────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text("Статус:")
            self._status_text_id = dpg.add_text("Остановлено. Нажмите «Запустить».", color=(255, 100, 100))
            dpg.add_spacer(width=20)
            self._start_button_id = dpg.add_button(
                label="Запустить",
                callback=lambda: self._on_start_requested() if self._on_start_requested else None,
            )
            self._stop_button_id = dpg.add_button(
                label="Остановить",
                callback=lambda: self._on_stop_requested() if self._on_stop_requested else None,
            )
            dpg.add_spacer(width=20)
            gpu_text = f"GPU: {'Доступна' if self._cuda else 'Нет'}"
            dpg.add_text(gpu_text, color=(100, 255, 100) if self._cuda else (255, 100, 100))
            dpg.add_spacer(width=10)
            trt_text = f"TRT: {'Доступен' if self._trt else 'Нет'}"
            dpg.add_text(trt_text, color=(100, 255, 100) if self._trt else (255, 200, 100))
        dpg.add_separator()

    def _build_capture_tab(self) -> None:
        with dpg.tab(label="Захват"):
            dpg.add_text("Параметры захвата экрана", color=(200, 200, 255))
            dpg.add_separator()

            # Monitor
            mon_labels = [f"Монитор {i}: {m['width']}x{m['height']}"
                          for i, m in enumerate(self._monitors)] or ["Монитор 0"]
            w = dpg.add_combo(
                label="Монитор", items=mon_labels,
                default_value=mon_labels[min(self._cfg.get("capture.monitor_index", 0),
                                             len(mon_labels) - 1)],
                callback=lambda s, v: self._cfg.set(
                    "capture.monitor_index", mon_labels.index(v)),
            )
            _tip(w, "capture.monitor_index")

            w = dpg.add_input_int(
                label="Ширина ROI", default_value=self._cfg.get("capture.roi_width", 640),
                min_value=64, max_value=2560, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("capture.roi_width", v),
            )
            _tip(w, "capture.roi_width")

            w = dpg.add_input_int(
                label="Высота ROI", default_value=self._cfg.get("capture.roi_height", 640),
                min_value=64, max_value=2560, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("capture.roi_height", v),
            )
            _tip(w, "capture.roi_height")

            w = dpg.add_input_int(
                label="FPS лимит (0=без)", default_value=self._cfg.get("capture.fps_limit", 0),
                min_value=0, max_value=500, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("capture.fps_limit", v),
            )
            _tip(w, "capture.fps_limit")

            w = dpg.add_combo(
                label="Метод захвата", items=["dxcam", "mss"],
                default_value=self._cfg.get("capture.method", "dxcam"),
                callback=lambda s, v: self._cfg.set("capture.method", v),
            )
            _tip(w, "capture.method")

            w = dpg.add_checkbox(
                label="Пауза захвата",
                default_value=self._cfg.get("capture.paused", False),
                callback=lambda s, v: self._cfg.set("capture.paused", v),
            )
            _tip(w, "capture.paused")

            w = dpg.add_checkbox(
                label="Показать рамку ROI",
                default_value=self._cfg.get("capture.show_roi_border", True),
                callback=lambda s, v: self._cfg.set("capture.show_roi_border", v),
            )
            _tip(w, "capture.show_roi_border")

    def _on_device_mode_changed(self, sender, value) -> None:
        """Handle device mode combo change."""
        if value == "GPU (CUDA)":
            self._cfg.set("inference.device", "cuda")
            self._cfg.set("inference.fp16", True)
        elif value == "CPU":
            self._cfg.set("inference.device", "cpu")
            self._cfg.set("inference.fp16", False)
        else:
            self._cfg.set("inference.device", "auto")
            self._cfg.set("inference.fp16", self._cuda)

    def _get_device_mode_label(self) -> str:
        dev = self._cfg.get("inference.device", "auto")
        if dev == "cuda":
            return "GPU (CUDA)"
        elif dev == "cpu":
            return "CPU"
        return "Авто"

    def _build_detection_tab(self) -> None:
        with dpg.tab(label="Детекция"):
            # ── Model section ────────────────────────────────────────
            dpg.add_text("Модель", color=(200, 200, 255))
            dpg.add_separator()

            with dpg.group(horizontal=True):
                self._model_path_input_id = dpg.add_input_text(
                    label="",
                    default_value=self._cfg.get("inference.model_path", "yolo26n.pt"),
                    callback=lambda s, v: self._set_model_path(v),
                    on_enter=True,
                    width=420,
                )
                dpg.add_button(label="Обзор...", callback=lambda: self._open_model_file_dialog())
                if self._on_model_reload:
                    dpg.add_button(label="Перезагрузить",
                                   callback=lambda: self._on_model_reload())
            _tip(self._model_path_input_id, "inference.model_path")

            # Device mode — simplified: Auto / GPU / CPU
            device_items = ["Авто"]
            if self._cuda:
                device_items.append("GPU (CUDA)")
            device_items.append("CPU")

            cur_label = self._get_device_mode_label()
            if cur_label not in device_items:
                cur_label = "Авто"

            with dpg.group(horizontal=True):
                w = dpg.add_combo(
                    label="Устройство",
                    items=device_items,
                    default_value=cur_label,
                    callback=self._on_device_mode_changed,
                    width=160,
                )
                _tip(w, "inference.device")

                # Status indicator
                if self._cuda:
                    dpg.add_text("  GPU доступна", color=(100, 255, 100))
                else:
                    dpg.add_text("  GPU недоступна, используется CPU", color=(255, 200, 100))

            w = dpg.add_input_int(
                label="Разрешение модели",
                default_value=self._cfg.get("inference.input_size", 640),
                min_value=128, max_value=2048, step=32,
                min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("inference.input_size", v),
            )
            _tip(w, "inference.input_size")

            # ── Detection parameters ─────────────────────────────────
            dpg.add_spacer(height=5)
            dpg.add_text("Параметры детекции", color=(200, 200, 255))
            dpg.add_separator()

            w = dpg.add_slider_float(
                label="Порог уверенности",
                default_value=self._cfg.get("detection.confidence", 0.45),
                min_value=0.05, max_value=1.0, format="%.2f",
                callback=lambda s, v: self._cfg.set("detection.confidence", v),
            )
            _tip(w, "detection.confidence")

            w = dpg.add_slider_float(
                label="Порог NMS IoU",
                default_value=self._cfg.get("detection.nms_iou", 0.45),
                min_value=0.05, max_value=1.0, format="%.2f",
                callback=lambda s, v: self._cfg.set("detection.nms_iou", v),
            )
            _tip(w, "detection.nms_iou")

            w = dpg.add_input_int(
                label="Мин. ширина bbox",
                default_value=self._cfg.get("detection.min_bbox_width", 10),
                min_value=1, max_value=500, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("detection.min_bbox_width", v),
            )
            _tip(w, "detection.min_bbox_width")

            w = dpg.add_input_int(
                label="Мин. высота bbox",
                default_value=self._cfg.get("detection.min_bbox_height", 10),
                min_value=1, max_value=500, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("detection.min_bbox_height", v),
            )
            _tip(w, "detection.min_bbox_height")

            w = dpg.add_input_int(
                label="Макс. детекций",
                default_value=self._cfg.get("detection.max_detections", 20),
                min_value=1, max_value=200, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("detection.max_detections", v),
            )
            _tip(w, "detection.max_detections")

    def _build_tracking_tab(self) -> None:
        with dpg.tab(label="Трекинг"):
            dpg.add_text("Удержание цели", color=(200, 200, 255))
            dpg.add_separator()

            w = dpg.add_input_int(
                label="Порог потери (кадры)",
                default_value=self._cfg.get("tracking.lost_frames_threshold", 15),
                min_value=1, max_value=120, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("tracking.lost_frames_threshold", v),
            )
            _tip(w, "tracking.lost_frames_threshold")

            w = dpg.add_combo(
                label="Метод ассоциации", items=["iou", "distance"],
                default_value=self._cfg.get("tracking.assoc_mode", "iou"),
                callback=lambda s, v: self._cfg.set("tracking.assoc_mode", v),
            )
            _tip(w, "tracking.assoc_mode")

            w = dpg.add_slider_float(
                label="Порог IoU ассоциации",
                default_value=self._cfg.get("tracking.assoc_iou_threshold", 0.25),
                min_value=0.05, max_value=0.9, format="%.2f",
                callback=lambda s, v: self._cfg.set("tracking.assoc_iou_threshold", v),
            )
            _tip(w, "tracking.assoc_iou_threshold")

            w = dpg.add_input_int(
                label="Порог расстояния (px)",
                default_value=self._cfg.get("tracking.assoc_distance_threshold", 150),
                min_value=10, max_value=1000, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("tracking.assoc_distance_threshold", v),
            )
            _tip(w, "tracking.assoc_distance_threshold")

            w = dpg.add_input_float(
                label="Штраф за переключение",
                default_value=self._cfg.get("tracking.switch_penalty", 50.0),
                min_value=0, max_value=500, format="%.1f",
                callback=lambda s, v: self._cfg.set("tracking.switch_penalty", v),
            )
            _tip(w, "tracking.switch_penalty")

    def _build_aim_tab(self) -> None:
        with dpg.tab(label="Наведение"):
            dpg.add_text("Управление мышью", color=(200, 200, 255))
            dpg.add_separator()

            w = dpg.add_checkbox(
                label="Включить управление мышью",
                default_value=self._cfg.get("aim.enabled", False),
                callback=lambda s, v: self._cfg.set("aim.enabled", v),
            )
            _tip(w, "aim.enabled")

            w = dpg.add_combo(
                label="Режим активации", items=["always", "hold"],
                default_value=self._cfg.get("aim.activation_mode", "hold"),
                callback=lambda s, v: self._cfg.set("aim.activation_mode", v),
            )
            _tip(w, "aim.activation_mode")

            w = dpg.add_input_text(
                label="Клавиша активации",
                default_value=self._cfg.get("aim.activation_key", "x2"),
                callback=lambda s, v: self._cfg.set("aim.activation_key", v),
                on_enter=True,
            )
            _tip(w, "aim.activation_key")

            dpg.add_separator()
            dpg.add_text("Параметры движения", color=(200, 255, 200))

            w = dpg.add_slider_float(
                label="Чувствительность",
                default_value=self._cfg.get("aim.sensitivity", 1.0),
                min_value=0.1, max_value=5.0, format="%.2f",
                callback=lambda s, v: self._cfg.set("aim.sensitivity", v),
            )
            _tip(w, "aim.sensitivity")

            w = dpg.add_input_int(
                label="Макс. скорость (px/s)",
                default_value=self._cfg.get("aim.max_speed", 800),
                min_value=50, max_value=5000, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("aim.max_speed", v),
            )
            _tip(w, "aim.max_speed")

            w = dpg.add_input_int(
                label="Макс. ускорение (px/s²)",
                default_value=self._cfg.get("aim.max_acceleration", 5000),
                min_value=100, max_value=50000, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("aim.max_acceleration", v),
            )
            _tip(w, "aim.max_acceleration")

            w = dpg.add_input_int(
                label="Мёртвая зона (px)",
                default_value=self._cfg.get("aim.deadzone", 3),
                min_value=0, max_value=50, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("aim.deadzone", v),
            )
            _tip(w, "aim.deadzone")

            dpg.add_separator()
            dpg.add_text("Точка прицеливания", color=(200, 255, 200))

            w = dpg.add_combo(
                label="Точка на боксе",
                items=["center", "upper_third", "custom"],
                default_value=self._cfg.get("aim.aim_point", "center"),
                callback=lambda s, v: self._cfg.set("aim.aim_point", v),
            )
            _tip(w, "aim.aim_point")

            w = dpg.add_slider_float(
                label="Смещение X",
                default_value=self._cfg.get("aim.aim_offset_x", 0.0),
                min_value=-1.0, max_value=1.0, format="%.2f",
                callback=lambda s, v: self._cfg.set("aim.aim_offset_x", v),
            )
            _tip(w, "aim.aim_offset_x")

            w = dpg.add_slider_float(
                label="Смещение Y",
                default_value=self._cfg.get("aim.aim_offset_y", -0.33),
                min_value=-1.0, max_value=1.0, format="%.2f",
                callback=lambda s, v: self._cfg.set("aim.aim_offset_y", v),
            )
            _tip(w, "aim.aim_offset_y")

            dpg.add_separator()
            dpg.add_text("Профиль скорости", color=(200, 255, 200))

            w = dpg.add_combo(
                label="Профиль", items=["power", "piecewise"],
                default_value=self._cfg.get("aim.speed_profile", "power"),
                callback=lambda s, v: self._cfg.set("aim.speed_profile", v),
            )
            _tip(w, "aim.speed_profile")

            w = dpg.add_input_float(
                label="r0 (степенной)",
                default_value=self._cfg.get("aim.power_r0", 200.0),
                min_value=10, max_value=1000, format="%.0f",
                callback=lambda s, v: self._cfg.set("aim.power_r0", v),
            )
            _tip(w, "aim.power_r0")

            w = dpg.add_slider_float(
                label="gamma (степенной)",
                default_value=self._cfg.get("aim.power_gamma", 0.7),
                min_value=0.1, max_value=2.0, format="%.2f",
                callback=lambda s, v: self._cfg.set("aim.power_gamma", v),
            )
            _tip(w, "aim.power_gamma")

            w = dpg.add_input_int(
                label="r1 (кусочный)",
                default_value=self._cfg.get("aim.piecewise_r1", 30),
                min_value=1, max_value=200, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("aim.piecewise_r1", v),
            )
            _tip(w, "aim.piecewise_r1")

            w = dpg.add_input_int(
                label="r2 (кусочный)",
                default_value=self._cfg.get("aim.piecewise_r2", 200),
                min_value=50, max_value=1000, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("aim.piecewise_r2", v),
            )
            _tip(w, "aim.piecewise_r2")

            w = dpg.add_input_int(
                label="Частота обновления (Гц)",
                default_value=self._cfg.get("aim.tick_rate", 240),
                min_value=30, max_value=1000, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("aim.tick_rate", v),
            )
            _tip(w, "aim.tick_rate")

    def _build_filters_tab(self) -> None:
        with dpg.tab(label="Фильтры"):
            dpg.add_text("Сглаживание", color=(200, 200, 255))
            dpg.add_separator()

            w = dpg.add_combo(
                label="Тип фильтра", items=["off", "ema", "one_euro"],
                default_value=self._cfg.get("filters.type", "one_euro"),
                callback=lambda s, v: self._cfg.set("filters.type", v),
            )
            _tip(w, "filters.type")

            w = dpg.add_combo(
                label="Применять к", items=["error", "target"],
                default_value=self._cfg.get("filters.apply_to", "error"),
                callback=lambda s, v: self._cfg.set("filters.apply_to", v),
            )
            _tip(w, "filters.apply_to")

            dpg.add_separator()
            dpg.add_text("EMA", color=(200, 255, 200))

            w = dpg.add_slider_float(
                label="Альфа EMA",
                default_value=self._cfg.get("filters.ema_alpha", 0.3),
                min_value=0.01, max_value=1.0, format="%.3f",
                callback=lambda s, v: self._cfg.set("filters.ema_alpha", v),
            )
            _tip(w, "filters.ema_alpha")

            dpg.add_separator()
            dpg.add_text("One Euro Filter", color=(200, 255, 200))

            w = dpg.add_slider_float(
                label="Мин. cutoff",
                default_value=self._cfg.get("filters.one_euro_min_cutoff", 1.0),
                min_value=0.01, max_value=10.0, format="%.3f",
                callback=lambda s, v: self._cfg.set("filters.one_euro_min_cutoff", v),
            )
            _tip(w, "filters.one_euro_min_cutoff")

            w = dpg.add_slider_float(
                label="Бета",
                default_value=self._cfg.get("filters.one_euro_beta", 0.007),
                min_value=0.0, max_value=0.5, format="%.4f",
                callback=lambda s, v: self._cfg.set("filters.one_euro_beta", v),
            )
            _tip(w, "filters.one_euro_beta")

            w = dpg.add_slider_float(
                label="D cutoff",
                default_value=self._cfg.get("filters.one_euro_d_cutoff", 1.0),
                min_value=0.1, max_value=5.0, format="%.2f",
                callback=lambda s, v: self._cfg.set("filters.one_euro_d_cutoff", v),
            )
            _tip(w, "filters.one_euro_d_cutoff")

            dpg.add_separator()

            w = dpg.add_checkbox(
                label="Фильтр по X",
                default_value=self._cfg.get("filters.enable_x", True),
                callback=lambda s, v: self._cfg.set("filters.enable_x", v),
            )
            _tip(w, "filters.enable_x")

            w = dpg.add_checkbox(
                label="Фильтр по Y",
                default_value=self._cfg.get("filters.enable_y", True),
                callback=lambda s, v: self._cfg.set("filters.enable_y", v),
            )
            _tip(w, "filters.enable_y")

    def _build_overlay_tab(self) -> None:
        with dpg.tab(label="Оверлей"):
            dpg.add_text("Экранный оверлей и превью", color=(200, 200, 255))
            dpg.add_separator()

            dpg.add_text("Оверлей", color=(200, 255, 200))

            w = dpg.add_checkbox(
                label="Оверлей вкл/выкл",
                default_value=self._cfg.get("overlay.enabled", True),
                callback=lambda s, v: self._cfg.set("overlay.enabled", v),
            )
            _tip(w, "overlay.enabled")

            w = dpg.add_checkbox(
                label="Рамки детекций",
                default_value=self._cfg.get("overlay.draw_bboxes", True),
                callback=lambda s, v: self._cfg.set("overlay.draw_bboxes", v),
            )
            _tip(w, "overlay.draw_bboxes")

            w = dpg.add_checkbox(
                label="Выделение цели",
                default_value=self._cfg.get("overlay.draw_locked_target", True),
                callback=lambda s, v: self._cfg.set("overlay.draw_locked_target", v),
            )
            _tip(w, "overlay.draw_locked_target")

            w = dpg.add_checkbox(
                label="Прицел",
                default_value=self._cfg.get("overlay.draw_crosshair", True),
                callback=lambda s, v: self._cfg.set("overlay.draw_crosshair", v),
            )
            _tip(w, "overlay.draw_crosshair")

            w = dpg.add_checkbox(
                label="Линия наведения",
                default_value=self._cfg.get("overlay.draw_aim_line", True),
                callback=lambda s, v: self._cfg.set("overlay.draw_aim_line", v),
            )
            _tip(w, "overlay.draw_aim_line")

            w = dpg.add_checkbox(
                label="Метрики в оверлее",
                default_value=self._cfg.get("overlay.show_metrics", True),
                callback=lambda s, v: self._cfg.set("overlay.show_metrics", v),
            )
            _tip(w, "overlay.show_metrics")

            dpg.add_separator()
            dpg.add_text("Превью", color=(200, 255, 200))

            w = dpg.add_checkbox(
                label="Превью вкл/выкл",
                default_value=self._cfg.get("preview.enabled", True),
                callback=lambda s, v: self._cfg.set("preview.enabled", v),
            )
            _tip(w, "preview.enabled")

            w = dpg.add_checkbox(
                label="Bbox в превью",
                default_value=self._cfg.get("preview.draw_bboxes", True),
                callback=lambda s, v: self._cfg.set("preview.draw_bboxes", v),
            )
            _tip(w, "preview.draw_bboxes")

            w = dpg.add_input_int(
                label="FPS превью (0=без)",
                default_value=self._cfg.get("preview.max_fps", 20),
                min_value=0, max_value=240, min_clamped=True, max_clamped=True,
                callback=lambda s, v: self._cfg.set("preview.max_fps", v),
            )
            _tip(w, "preview.max_fps")

            w = dpg.add_checkbox(
                label="Текст в превью",
                default_value=self._cfg.get("preview.draw_text", True),
                callback=lambda s, v: self._cfg.set("preview.draw_text", v),
            )
            _tip(w, "preview.draw_text")

    def _build_export_tab(self) -> None:
        with dpg.tab(label="Конвертация"):
            dpg.add_text("Конвертация модели для ускорения", color=(200, 200, 255))
            dpg.add_separator()

            dpg.add_text(
                "Конвертация ускоряет работу модели.\n"
                "Порядок: .pt (обычная) -> ONNX (быстрее) -> TensorRT (максимум скорости, только GPU).",
                color=(180, 180, 180), wrap=700,
            )
            dpg.add_spacer(height=8)

            # ── Step 1: ONNX ─────────────────────────────────────────
            dpg.add_text("Шаг 1: Экспорт в ONNX", color=(100, 200, 255))
            dpg.add_text(
                "Конвертирует .pt модель в формат ONNX. Работает на CPU и GPU.",
                color=(160, 160, 160), wrap=700,
            )

            with dpg.group(horizontal=True):
                w = dpg.add_input_int(
                    label="Opset", default_value=self._cfg.get("export.onnx_opset", 17),
                    min_value=9, max_value=20, min_clamped=True, max_clamped=True,
                    callback=lambda s, v: self._cfg.set("export.onnx_opset", v),
                    width=100,
                )
                _tip(w, "export.onnx_opset")

                w = dpg.add_checkbox(
                    label="Динамич. размеры",
                    default_value=self._cfg.get("export.onnx_dynamic", False),
                    callback=lambda s, v: self._cfg.set("export.onnx_dynamic", v),
                )
                _tip(w, "export.onnx_dynamic")

            if self._on_export_onnx:
                dpg.add_button(
                    label="Экспортировать .pt -> ONNX",
                    callback=lambda: threading.Thread(
                        target=self._on_export_onnx, daemon=True).start(),
                )

            dpg.add_spacer(height=10)
            dpg.add_separator()

            # ── Step 2: TensorRT ─────────────────────────────────────
            trt_color = (100, 200, 255) if self._trt else (120, 120, 120)
            dpg.add_text("Шаг 2: Сборка TensorRT (опционально)", color=trt_color)

            if self._trt:
                dpg.add_text(
                    "Собирает оптимизированный движок TensorRT из ONNX. Максимальная скорость на GPU.",
                    color=(160, 160, 160), wrap=700,
                )
            else:
                dpg.add_text(
                    "TensorRT не установлен. Для использования установите пакеты tensorrt и pycuda.\n"
                    "Без TensorRT модель будет работать через PyTorch (Шаг 1 достаточен).",
                    color=(255, 200, 100), wrap=700,
                )

            with dpg.group(horizontal=True):
                w = dpg.add_checkbox(
                    label="FP16",
                    default_value=self._cfg.get("export.trt_fp16", True),
                    callback=lambda s, v: self._cfg.set("export.trt_fp16", v),
                    enabled=self._trt,
                )
                _tip(w, "export.trt_fp16")

                w = dpg.add_input_int(
                    label="Память (МБ)", default_value=self._cfg.get("export.trt_workspace_mb", 4096),
                    min_value=256, max_value=32768, min_clamped=True, max_clamped=True,
                    callback=lambda s, v: self._cfg.set("export.trt_workspace_mb", v),
                    width=120, enabled=self._trt,
                )
                _tip(w, "export.trt_workspace_mb")

            with dpg.group(horizontal=True):
                self._engine_path_input_id = dpg.add_input_text(
                    label="",
                    default_value=self._cfg.get("export.trt_engine_path", ""),
                    callback=lambda s, v: self._set_engine_path(v),
                    on_enter=True, width=420, enabled=self._trt,
                    hint="Путь к .engine файлу (необязательно)",
                )
                dpg.add_button(label="Обзор...",
                               callback=lambda: self._open_engine_file_dialog(),
                               enabled=self._trt)

            if self._on_build_trt:
                dpg.add_button(
                    label="Собрать ONNX -> TensorRT Engine",
                    callback=lambda: threading.Thread(
                        target=self._on_build_trt, daemon=True).start(),
                    enabled=self._trt,
                )

    def _build_hotkeys_tab(self) -> None:
        with dpg.tab(label="Горячие клавиши"):
            dpg.add_text("Настройка горячих клавиш", color=(200, 200, 255))
            dpg.add_separator()

            w = dpg.add_input_text(
                label="Переключить мышь",
                default_value=self._cfg.get("gui.hotkey_toggle_mouse", "f6"),
                callback=lambda s, v: self._cfg.set("gui.hotkey_toggle_mouse", v),
                on_enter=True,
            )
            _tip(w, "gui.hotkey_toggle_mouse")

            w = dpg.add_input_text(
                label="Переключить оверлей",
                default_value=self._cfg.get("gui.hotkey_toggle_overlay", "f7"),
                callback=lambda s, v: self._cfg.set("gui.hotkey_toggle_overlay", v),
                on_enter=True,
            )
            _tip(w, "gui.hotkey_toggle_overlay")

            w = dpg.add_input_text(
                label="Сброс цели",
                default_value=self._cfg.get("gui.hotkey_lock_target", ""),
                callback=lambda s, v: self._cfg.set("gui.hotkey_lock_target", v),
                on_enter=True,
            )
            _tip(w, "gui.hotkey_lock_target")

            w = dpg.add_input_text(
                label="Пауза захвата",
                default_value=self._cfg.get("gui.hotkey_pause_capture", "f8"),
                callback=lambda s, v: self._cfg.set("gui.hotkey_pause_capture", v),
                on_enter=True,
            )
            _tip(w, "gui.hotkey_pause_capture")

    def _build_logs_tab(self) -> None:
        with dpg.tab(label="Логи"):
            dpg.add_text("Журнал событий", color=(200, 200, 255))
            dpg.add_separator()
            self._log_text_id = dpg.add_input_text(
                multiline=True, readonly=True,
                width=-1, height=400,
                default_value="Лог пуст...",
            )

    def _build_telemetry_panel(self) -> None:
        dpg.add_separator()
        dpg.add_text("Телеметрия", color=(255, 255, 200))
        self._telemetry_text_id = dpg.add_text(
            "Остановлено.\nНажмите «Запустить», чтобы начать захват.",
            color=(180, 255, 180),
        )

    def _build_preview_panel(self) -> None:
        dpg.add_separator()
        dpg.add_text("Превью ROI", color=(255, 255, 200))
        if self._preview_texture_id is not None:
            dpg.add_image(self._preview_texture_id,
                          width=self._preview_width,
                          height=self._preview_height)

    def _build_config_buttons(self) -> None:
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Сохранить сейчас",
                           callback=lambda: self._cfg.save_now())
            dpg.add_button(label="Сброс настроек",
                           callback=lambda: self._cfg.reset_defaults())
