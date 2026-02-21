"""Minimal settings container used by scripts that need explicit config wiring."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Settings:
    """Global settings object placeholder for future centralized configuration."""
    config_path: str | None = None


DEFAULT_SETTINGS = Settings()
