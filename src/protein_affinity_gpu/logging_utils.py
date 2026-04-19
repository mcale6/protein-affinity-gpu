"""Consistent logging + timing helpers for protein-affinity-gpu.

Provides ``setup_logging`` for configuring a package-scoped logger and
``log_duration`` — a context manager that emits a debug-level "phase took
Xs" line and the total for a named pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from typing import IO, Iterator, Optional


PACKAGE_LOGGER = logging.getLogger("protein_affinity_gpu")

_ANSI_RESET = "\x1b[0m"
_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\x1b[36m",       # cyan
    logging.INFO: "\x1b[32m",        # green
    logging.WARNING: "\x1b[33m",     # yellow
    logging.ERROR: "\x1b[31m",       # red
    logging.CRITICAL: "\x1b[1;31m",  # bold red
}
_TIMING_COLOR = "\x1b[35m"           # magenta for "done in X.XXXs"
_DURATION_RE = re.compile(r"(done in )(\d+\.\d+s)")


def supports_color(stream: IO[str]) -> bool:
    """ANSI colors allowed only on a real TTY and when NO_COLOR is unset."""
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


class _ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str, *, use_color: bool) -> None:
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self.use_color:
            return formatted
        color = _LEVEL_COLORS.get(record.levelno, "")
        if color:
            tag = f"[{record.levelname}]"
            formatted = formatted.replace(
                tag, f"[{color}{record.levelname}{_ANSI_RESET}]", 1
            )
        return _DURATION_RE.sub(rf"\1{_TIMING_COLOR}\2{_ANSI_RESET}", formatted)


def setup_logging(
    level: int | str = "INFO",
    *,
    propagate: bool = True,
    stream: Optional[IO[str]] = None,
) -> logging.Logger:
    """Attach a colored ``StreamHandler`` to the package logger if none exists."""
    logger = PACKAGE_LOGGER
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logger.setLevel(level)
    logger.propagate = propagate

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        target_stream = stream if stream is not None else sys.stderr
        handler = logging.StreamHandler(target_stream)
        handler.setFormatter(
            _ColorFormatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "%H:%M:%S",
                use_color=supports_color(target_stream),
            )
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
