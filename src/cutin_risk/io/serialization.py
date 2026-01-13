from __future__ import annotations
from pathlib import Path


def save_placeholder(path: str | Path):
    """Placeholder for serialization helpers."""
    Path(path).write_text("placeholder", encoding="utf-8")
