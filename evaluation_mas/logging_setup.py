"""
Basic structured logging setup.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pythonjsonlogger import jsonlogger


def setup_logging(log_path: Path, verbose: bool = False, quiet: bool = False) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler
    if not quiet:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(ch)

    # File handler (JSON)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


__all__ = ["setup_logging"]
