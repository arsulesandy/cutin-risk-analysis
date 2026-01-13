from __future__ import annotations


def safe_divide(numerator: float, denominator: float, *, default: float | None = None) -> float:
    """Divide while avoiding ZeroDivisionError."""
    if denominator == 0:
        if default is None:
            return float("inf")
        return default
    return numerator / denominator
