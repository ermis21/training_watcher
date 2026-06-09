"""A dedicated ``training_watcher`` logger, independent of the host app's logging."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s  %(levelname)-7s  [training_watcher] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(log_path: str | None = None) -> logging.Logger:
    """Return the package logger, attaching a stderr handler (and optional file) once."""
    logger = logging.getLogger("training_watcher")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    have_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not have_stream:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(sh)

    if log_path:
        already = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_path
            for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
            logger.addHandler(fh)

    return logger
