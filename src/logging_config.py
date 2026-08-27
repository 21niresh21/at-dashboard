"""Logging configuration.

Provides a single ``setup_logging()`` call that configures the root logger
for the entire application (both stdlib logging and Streamlit's logger).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config import settings

_CONFIGURED = False


def setup_logging() -> None:
    """Initialise application-wide logging.

    Safe to call multiple times; only the first call has effect.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # Ensure logs directory exists
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # ── Stream handler (stdout) ──────────────────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    stream_handler.setLevel(settings.log_level)

    # ── File handler ─────────────────────────────────────────────────────────
    log_file = settings.logs_dir / "app.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    file_handler.setLevel(settings.log_level)

    # ── Root logger ──────────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    # Quiet noisy third-party loggers in development
    if settings.is_development:
        logging.getLogger("watchdog").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for use inside a module.

    Usage:
        from src.logging_config import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
