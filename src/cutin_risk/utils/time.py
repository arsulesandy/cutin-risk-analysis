"""Time-conversion helpers."""

from __future__ import annotations


def to_seconds(timestamp_ms: float) -> float:
    """Convert millisecond timestamps to seconds."""
    return timestamp_ms / 1000.0
