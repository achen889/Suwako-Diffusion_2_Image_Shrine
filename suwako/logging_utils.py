"""Persistent, per-process application logging."""
from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs"


def configure_logging() -> tuple[logging.Logger, Path]:
    logger = logging.getLogger("suwako")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return logger, Path(handler.baseFilename)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_ROOT / f"suwako_{stamp}_{os.getpid()}_{threading.get_ident()}.log"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
    )
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    file_handler = logging.FileHandler(path, encoding="utf-8", delay=False)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Logging started; session file: %s", path)
    return logger, path
