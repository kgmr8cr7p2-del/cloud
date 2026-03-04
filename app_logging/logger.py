"""
Application logger: file + in-memory ring buffer for GUI log window.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


class LogEntry(NamedTuple):
    timestamp: str
    level: str
    message: str


_MAX_BUFFER = 2000
_LOG_DIR = Path("logs")


class AppLogger:
    """Singleton-style logger with file output and ring buffer for GUI."""

    _instance: AppLogger | None = None
    _lock = threading.Lock()

    def __new__(cls) -> AppLogger:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._buffer: deque[LogEntry] = deque(maxlen=_MAX_BUFFER)
        self._buf_lock = threading.Lock()

        _LOG_DIR.mkdir(exist_ok=True)
        log_file = _LOG_DIR / f"app_{datetime.now():%Y%m%d_%H%M%S}.log"

        self._logger = logging.getLogger("app")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%H:%M:%S")
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

    def _push(self, level: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with self._buf_lock:
            self._buffer.append(LogEntry(ts, level, msg))

    def info(self, msg: str) -> None:
        self._logger.info(msg)
        self._push("INFO", msg)

    def warn(self, msg: str) -> None:
        self._logger.warning(msg)
        self._push("WARN", msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)
        self._push("ERROR", msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def get_entries(self, last_n: int = 200) -> list[LogEntry]:
        with self._buf_lock:
            items = list(self._buffer)
            return items[-last_n:]


def get_logger() -> AppLogger:
    return AppLogger()
