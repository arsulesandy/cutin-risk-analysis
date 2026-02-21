"""Small mathematical helpers shared across modules."""

from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value between lower and upper bounds."""
    return max(lower, min(value, upper))
