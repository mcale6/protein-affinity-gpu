"""Consistent logging + timing helpers for protein-affinity-gpu.

Provides ``setup_logging`` for configuring a package-scoped logger and
``log_duration`` — a context manager that emits a debug-level "phase took
Xs" line and the total for a named pipeline.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional


PACKAGE_LOGGER = logging.getLogger("protein_affinity_gpu")


def setup_logging(level: int | str = "INFO", *, propagate: bool = True) -> logging.Logger:
    """Attach a ``StreamHandler`` to the package logger if none exists."""
    logger = PACKAGE_LOGGER
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logger.setLevel(level)
    logger.propagate = propagate

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, e.g. ``protein_affinity_gpu.tinygrad``."""
    if name.startswith(PACKAGE_LOGGER.name):
        return logging.getLogger(name)
    return PACKAGE_LOGGER.getChild(name.removeprefix("protein_affinity_gpu."))


@contextmanager
def log_duration(
    logger: logging.Logger,
    label: str,
    *,
    level: int = logging.DEBUG,
    extra: Optional[str] = None,
) -> Iterator[None]:
    """Context manager that logs ``label`` completion with elapsed seconds."""
    start = time.perf_counter()
    logger.log(level, "%s: start%s", label, f" ({extra})" if extra else "")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s: done in %.3fs", label, elapsed)
