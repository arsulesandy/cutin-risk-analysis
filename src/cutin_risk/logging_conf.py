"""Project-level logging configuration entrypoint."""

from __future__ import annotations
from pathlib import Path
import logging


def configure_logging(config_path: str | Path | None = None) -> None:
    """Configure logging for scripts; currently uses a conservative basic config."""
    _ = config_path
    logging.basicConfig(level=logging.INFO)
