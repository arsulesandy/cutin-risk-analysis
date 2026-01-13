from __future__ import annotations
from pathlib import Path
import logging


def configure_logging(config_path: str | Path | None = None) -> None:
    """Placeholder logging setup hook."""
    _ = config_path
    logging.basicConfig(level=logging.INFO)
