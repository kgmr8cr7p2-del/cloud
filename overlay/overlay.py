"""
Transparent always-on-top click-through overlay window using Win32 API.
Draws bboxes, locked target, crosshair, aim vector, metrics.
On non-Windows platforms the overlay is a no-op stub.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from typing import Any

from app_logging.logger import get_logger

log = get_logger()

_IS_WINDOWS = sys.platform == "win32"

# ── Win32 constants (only used on Windows) ───────────────────────────
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOPMOST = 0x8
WS_EX_TOOLWINDOW = 0x80
WS_POPUP = 0x80000000
GWL_EXSTYLE = -20
LWA_COLORKEY = 0x1
LWA_ALPHA = 0x2
SW_SHOW = 5
SW_HIDE = 0
COLOR_KEY = 0x00000001  # near-black as transparent key

WM_DESTROY = 0x0002
WM_PAINT = 0x000F
WM_TIMER = 0x0113

PS_SOLID = 0
DT_LEFT = 0
DT_SINGLELINE = 0x20
DT_NOCLIP = 0x100

TRANSPARENT_BG = 0
OPAQUE_BG = 1

if _IS_WINDOWS:
    import ctypes.wintypes

    HWND = getattr(ctypes.wintypes, "HWND", ctypes.c_void_p)
    UINT = getattr(ctypes.wintypes, "UINT", ctypes.c_uint)
    WPARAM = getattr(ctypes.wintypes, "WPARAM", ctypes.c_size_t)
    LPARAM = getattr(ctypes.wintypes, "LPARAM", ctypes.c_ssize_t)
    HINSTANCE = getattr(ctypes.wintypes, "HINSTANCE", ctypes.c_void_p)
    HICON = getattr(ctypes.wintypes, "HICON", ctypes.c_void_p)
    HCURSOR = getattr(ctypes.wintypes, "HCURSOR", ctypes.c_void_p)
    HBRUSH = getattr(ctypes.wintypes, "HBRUSH", ctypes.c_void_p)
    LPCWSTR = getattr(ctypes.wintypes, "LPCWSTR", ctypes.c_wchar_p)

    LRESULT = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
    WNDPROC = ctypes.WINFUNCTYPE(
        LRESULT,
        HWND,
        UINT,
        WPARAM,
        LPARAM,
    )


if _IS_WINDOWS:
    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", ctypes.wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", HINSTANCE),
            ("hIcon", HICON),
            ("hCursor", HCURSOR),
            ("hbrBackground", HBRUSH),
            ("lpszMenuName", LPCWSTR),
            ("lpszClassName", LPCWSTR),
        ]


class OverlayWindow:
    """Win32 transparent overlay with GDI drawing."""

    def __init__(self) -> None:
        self._hwnd = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._draw_data: dict[str, Any] = {}
        self._draw_lock = threading.Lock()
        self._config: dict = {}
        self._monitor_rect: tuple[int, int, int, int] = (0, 0, 1920, 1080)

    def _screen_to_client(self, x: int | float, y: int | float) -> tuple[int, int]:
        left, top, _, _ = self._monitor_rect
        return int(x - left), int(y - top)

    def start(self, monitor_rect: tuple[int, int, int, int] | None = None) -> None:
        if self._running:
            return
        if not _IS_WINDOWS:
            log.info("Overlay not available on this platform (Windows only)")
            return
        if monitor_rect:
            self._monitor_rect = monitor_rect
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="overlay")
        self._thread.start()
        log.info("Overlay started")

    def stop(self) -> None:
        self._running = False
        if self._hwnd:
            try:
                ctypes.windll.user32.PostMessageW(self._hwnd, WM_DESTROY, 0, 0)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)
        log.info("Overlay stopped")

    def update_data(self, data: dict[str, Any]) -> None:
        with self._draw_lock:
            self._draw_data = data.copy()

    def set_monitor_rect(self, monitor_rect: tuple[int, int, int, int]) -> None:
        self._monitor_rect = monitor_rect
        if self._hwnd:
            try:
                x, y, w, h = monitor_rect
                ctypes.windll.user32.MoveWindow(self._hwnd, x, y, w, h, True)
            except Exception:
                pass

    def set_config(self, cfg: dict) -> None:
        self._config = cfg.copy()

    def set_visible(self, visible: bool) -> None:
        if self._hwnd:
            try:
                ctypes.windll.user32.ShowWindow(self._hwnd, SW_SHOW if visible else SW_HIDE)
            except Exception:
                pass

    def _run(self) -> None:
        """Create overlay window and run message loop."""
        try:
            self._create_window()
        except Exception as e:
            log.error(f"Overlay window creation failed: {e}")
            self._running = False

    def _create_window(self) -> None:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        kernel32 = ctypes.windll.kernel32
        user32.DefWindowProcW.argtypes = [HWND, UINT, WPARAM, LPARAM]
        user32.DefWindowProcW.restype = LRESULT
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        user32.RegisterClassW.restype = ctypes.wintypes.ATOM
        user32.CreateWindowExW.argtypes = [
            ctypes.wintypes.DWORD,
            LPCWSTR,
            LPCWSTR,
            ctypes.wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            HWND,
            ctypes.c_void_p,
            HINSTANCE,
            ctypes.c_void_p,
        ]
        user32.CreateWindowExW.restype = HWND

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            if msg == WM_PAINT:
                self._on_paint(hwnd)
                return 0
            if msg == WM_TIMER:
                user32.InvalidateRect(hwnd, None, True)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc = WNDPROC(wnd_proc)
        class_name = "AimOverlay"
        hinstance = kernel32.GetModuleHandleW(None)

        wc = WNDCLASSW()
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = hinstance
        wc.lpszClassName = class_name
        wc.hbrBackground = gdi32.CreateSolidBrush(COLOR_KEY)
        wc.style = 0

        atom = user32.RegisterClassW(ctypes.byref(wc))
        if atom == 0:
            err = kernel32.GetLastError()
            # ERROR_CLASS_ALREADY_EXISTS
            if err != 1410:
                raise ctypes.WinError(err)

        x, y, w, h = self._monitor_rect
        ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW

        self._hwnd = user32.CreateWindowExW(
            ex_style, class_name, "Overlay",
            WS_POPUP,
            x, y, w, h,
            None, None, hinstance, None,
        )
        if not self._hwnd:
            raise ctypes.WinError(kernel32.GetLastError())

        # Set color key for transparency
        user32.SetLayeredWindowAttributes(self._hwnd, COLOR_KEY, 0, LWA_COLORKEY)
        user32.ShowWindow(self._hwnd, SW_SHOW)
        user32.UpdateWindow(self._hwnd)

        # Timer for refresh (~60 fps)
        user32.SetTimer(self._hwnd, 1, 16, None)

        # Message loop
        msg = ctypes.wintypes.MSG()
        while self._running:
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        self._hwnd = None

    def _on_paint(self, hwnd) -> None:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        class PAINTSTRUCT(ctypes.Structure):
            _fields_ = [
                ("hdc", ctypes.c_void_p),
                ("fErase", ctypes.c_int),
                ("rcPaint", ctypes.wintypes.RECT),
                ("fRestore", ctypes.c_int),
                ("fIncUpdate", ctypes.c_int),
                ("rgbReserved", ctypes.c_byte * 32),
            ]

        ps = PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))

        with self._draw_lock:
            data = self._draw_data.copy()

        cfg = self._config
        # Draw bboxes
        if cfg.get("draw_bboxes", True):
            detections = data.get("detections", [])
            for det in detections:
                x1, y1 = self._screen_to_client(det.x1, det.y1)
                x2, y2 = self._screen_to_client(det.x2, det.y2)
                self._draw_rect(hdc, gdi32,
                                x1, y1,
                                x2, y2,
                                color=0x00FF00, width=1)

        # Draw locked target
        if cfg.get("draw_locked_target", True):
            locked = data.get("locked_target")
            if locked:
                d = locked.detection
                x1, y1 = self._screen_to_client(d.x1, d.y1)
                x2, y2 = self._screen_to_client(d.x2, d.y2)
                self._draw_rect(hdc, gdi32,
                                x1, y1,
                                x2, y2,
                                color=0x0000FF, width=3)

        # Draw crosshair
        if cfg.get("draw_crosshair", True):
            cx = data.get("roi_cx", 0)
            cy = data.get("roi_cy", 0)
            if cx > 0:
                cx, cy = self._screen_to_client(cx, cy)
                size = 15
                self._draw_line(hdc, gdi32, cx - size, cy, cx + size, cy,
                                color=0x00FFFF, width=1)
                self._draw_line(hdc, gdi32, cx, cy - size, cx, cy + size,
                                color=0x00FFFF, width=1)

        # Draw aim line
        if cfg.get("draw_aim_line", True):
            locked = data.get("locked_target")
            if locked:
                cx = data.get("roi_cx", 0)
                cy = data.get("roi_cy", 0)
                d = locked.detection
                cx, cy = self._screen_to_client(cx, cy)
                tx, ty = self._screen_to_client(d.cx, d.cy)
                self._draw_line(hdc, gdi32,
                                cx, cy,
                                tx, ty,
                                color=0xFF00FF, width=1)

        # Draw metrics text
        if cfg.get("show_metrics", True):
            metrics_text = data.get("metrics_text", "")
            if metrics_text:
                self._draw_text(hdc, gdi32, 10, 10, metrics_text, color=0x00FFFF)

        user32.EndPaint(hwnd, ctypes.byref(ps))

    def _draw_rect(self, hdc, gdi32, x1, y1, x2, y2,
                   color: int = 0x00FF00, width: int = 1) -> None:
        pen = gdi32.CreatePen(PS_SOLID, width, color)
        old_pen = gdi32.SelectObject(hdc, pen)
        old_brush = gdi32.SelectObject(hdc, gdi32.GetStockObject(5))  # NULL_BRUSH
        gdi32.Rectangle(hdc, x1, y1, x2, y2)
        gdi32.SelectObject(hdc, old_pen)
        gdi32.SelectObject(hdc, old_brush)
        gdi32.DeleteObject(pen)

    def _draw_line(self, hdc, gdi32, x1, y1, x2, y2,
                   color: int = 0x00FF00, width: int = 1) -> None:
        pen = gdi32.CreatePen(PS_SOLID, width, color)
        old_pen = gdi32.SelectObject(hdc, pen)
        gdi32.MoveToEx(hdc, x1, y1, None)
        gdi32.LineTo(hdc, x2, y2)
        gdi32.SelectObject(hdc, old_pen)
        gdi32.DeleteObject(pen)

    def _draw_text(self, hdc, gdi32, x, y, text: str,
                   color: int = 0x00FFFF) -> None:
        gdi32.SetTextColor(hdc, color)
        gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
        font = gdi32.CreateFontW(
            14, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 0, 0, "Consolas",
        )
        old_font = gdi32.SelectObject(hdc, font)

        rect = ctypes.wintypes.RECT(x, y, x + 600, y + 300)
        ctypes.windll.user32.DrawTextW(
            hdc, text, -1, ctypes.byref(rect),
            DT_LEFT | DT_NOCLIP,
        )

        gdi32.SelectObject(hdc, old_font)
        gdi32.DeleteObject(font)
