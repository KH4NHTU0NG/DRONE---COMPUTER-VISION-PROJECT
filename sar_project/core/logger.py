"""
=========================================================
Logging System
=========================================================
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import config

class Logger:
    def __init__(self):
        self.logger = logging.getLogger("SAR")
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:
            return

        log_dir = config.path.logs
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "sar.log"
        formatter = logging.Formatter(
            "[%(asctime)s] "
            "[%(levelname)s] "
            "%(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

logger = Logger()
