"""Markdown formatting helpers."""

from __future__ import annotations

from typing import Literal, Sequence

Alignment = Literal["left", "right", "center"]


def _separator_cell(width: int, align: Alignment) -> str:
    if align == "right":
        return "-" * (width + 1) + ":"
    if align == "center":
        return ":" + "-" * width + ":"
    return "-" * (width + 2)


def markdown_table(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    align: Sequence[Alignment] | None = None,
) -> str:
    """Build a strict markdown table with fixed-width cells and explicit alignment."""
    header_cells = [str(h) for h in headers]
    string_rows = [[str(cell) for cell in row] for row in rows]
    n_cols = len(header_cells)
    if n_cols == 0:
        return ""

    if align is None:
        alignments: list[Alignment] = ["left"] * n_cols
    else:
        alignments = list(align)
        if len(alignments) != n_cols:
            raise ValueError("align length must match headers length")

    widths = [len(h) for h in header_cells]
    for row in string_rows:
        if len(row) != n_cols:
            raise ValueError("each row must have the same number of columns as headers")
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_cell(value: str, width: int, a: Alignment) -> str:
        if a == "right":
            return value.rjust(width)
        if a == "center":
            return value.center(width)
        return value.ljust(width)

    out: list[str] = []
    out.append(
        "| "
        + " | ".join(format_cell(header_cells[i], widths[i], alignments[i]) for i in range(n_cols))
        + " |"
    )
    out.append("|" + "|".join(_separator_cell(widths[i], alignments[i]) for i in range(n_cols)) + "|")
    for row in string_rows:
        out.append(
            "| "
            + " | ".join(format_cell(row[i], widths[i], alignments[i]) for i in range(n_cols))
            + " |"
        )
    return "\n".join(out)
