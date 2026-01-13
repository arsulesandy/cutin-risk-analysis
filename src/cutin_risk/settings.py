from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Settings:
    """Placeholder settings container."""
    config_path: str | None = None


DEFAULT_SETTINGS = Settings()
