from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str, level: str | int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    lvl = level or os.getenv("CXR_LOG_LEVEL", "INFO")
    logger.setLevel(lvl)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
