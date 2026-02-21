"""Lightweight terminal progress helpers for long-running loops."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


def iter_with_progress(
    items: Sequence[T],
    *,
    label: str,
    item_name: str | None = "item",
    emit: Callable[[str], None] = print,
) -> Iterator[tuple[int, int, T]]:
    """Yield ``(index, total, item)`` while emitting progress lines."""
    total = len(items)
    if total == 0:
        emit(f"[{label}] 0/0 (100.0%)")
        return

    for idx, item in enumerate(items, start=1):
        pct = (idx / total) * 100.0
        suffix = f" {item_name}={item}" if item_name else ""
        emit(f"[{label}] {idx}/{total} ({pct:5.1f}%){suffix}")
        yield idx, total, item

