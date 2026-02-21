"""Placeholder serialization utilities for intermediate artifacts."""

from __future__ import annotations
from pathlib import Path


def save_placeholder(path: str | Path):
    """Write a placeholder artifact used while serialization is being designed."""
    Path(path).write_text("placeholder", encoding="utf-8")
