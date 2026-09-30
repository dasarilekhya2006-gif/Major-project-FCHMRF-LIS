"""Structured logging configuration for reproducible experiment tracking and system monitoring."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_to_stdout: bool = True,
) -> None:
    """Configure project-wide logging format, level, and destination.

    Args:
        level: Logging level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to append log file output.
        log_to_stdout: Whether to stream logs to sys.stdout.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers = []

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Root logger setup
    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance configured for the specified module namespace.

    Args:
        name: Name of the calling module (typically __name__).

    Returns:
        logging.Logger: Configured logger.
    """
    return logging.getLogger(name)
