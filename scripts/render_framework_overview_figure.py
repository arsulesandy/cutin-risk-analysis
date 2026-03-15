#!/usr/bin/env python3
"""Render a cleaner framework overview figure for the thesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
LATEX_DIR = ROOT / "latex"
PDF_OUT = LATEX_DIR / "fig_framework_overview_academic.pdf"
PNG_OUT = LATEX_DIR / "fig_framework_overview_academic.png"


PALETTE = {
    "ink": "#243447",
    "method_fill": "#EAF2FB",
    "method_group": "#9FB8D6",
    "validation_fill": "#F3F4F6",
    "validation_group": "#B8C0CC",
    "metrics_fill": "#EDF6EE",
    "metrics_edge": "#7A9E7E",
    "accent": "#486581",
}


def add_group(ax, x: float, y: float, w: float, h: float, title: str, edge: str, fill: str) -> None:
    rect = Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.4,
        edgecolor=edge,
        facecolor=fill,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.012,
        y + h + 0.018,
        title,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 9.3,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.5,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    title_lines = title.count("\n") + 1
    title_y = 0.69 if title_lines == 1 else 0.72
    body_y = 0.27

    ax.text(
        x + w / 2.0,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        x + w / 2.0,
        y + h * body_y,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["ink"],
        linespacing=1.15,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    lw: float = 1.6,
    connectionstyle: str = "arc3",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color or PALETTE["accent"],
        shrinkA=6,
        shrinkB=6,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def render() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_group(
        ax,
        0.03,
        0.53,
        0.76,
        0.30,
        "Minimal-input reconstruction path (method)",
        PALETTE["method_group"],
        PALETTE["method_fill"],
    )
    add_group(
        ax,
        0.03,
        0.16,
        0.54,
        0.20,
        "Reference-label path (validation only)",
        PALETTE["validation_group"],
        PALETTE["validation_fill"],
    )

    x_positions = [0.055, 0.18, 0.305, 0.43, 0.555, 0.68]
    y_top = 0.61
    box_w = 0.10
    box_h = 0.14
    method_face = "#FFFFFF"

    add_box(
        ax,
        x_positions[0],
        y_top,
        box_w,
        box_h,
        "Minimal inputs",
        ["trajectories,", "kinematics,", "vehicle size,", "lane markings"],
        facecolor=method_face,
        edgecolor=PALETTE["method_group"],
    )
    add_box(
        ax,
        x_positions[1],
        y_top,
        box_w,
        box_h,
        "Preliminary\nprocessing",
        ["harmonisation,", "sanity checks,", "time alignment"],
        facecolor=method_face,
        edgecolor=PALETTE["method_group"],
    )
    add_box(
        ax,
        x_positions[2],
        y_top,
        box_w,
        box_h,
        "Traj2Rel\nreconstruction",
        ["lane index,", "same-lane", "leader / follower"],
        facecolor=method_face,
        edgecolor=PALETTE["method_group"],
    )
    add_box(
        ax,
        x_positions[3],
        y_top,
        box_w,
        box_h,
        "Cut-in mining",
        ["cutter-follower", "pairing,", "onset $t_0$,", "stages"],
        facecolor=method_face,
        edgecolor=PALETTE["method_group"],
    )
    add_box(
        ax,
        x_positions[4],
        y_top,
        box_w,
        box_h,
        "Safety indicators\n+ context",
        ["DHW, THW, TTC,", "DRAC,", "occupancy grid,", "mirrored SFC"],
        facecolor=method_face,
        edgecolor=PALETTE["method_group"],
        fontsize=9.0,
    )
    add_box(
        ax,
        x_positions[5],
        y_top,
        box_w,
        box_h,
        "Analysis-ready\noutputs",
        ["cut-in table,", "stage features,", "context codes"],
        facecolor=method_face,
        edgecolor=PALETTE["method_group"],
    )

    for idx in range(len(x_positions) - 1):
        add_arrow(
            ax,
            (x_positions[idx] + box_w, y_top + box_h / 2.0),
            (x_positions[idx + 1], y_top + box_h / 2.0),
        )

    add_box(
        ax,
        0.09,
        0.205,
        0.14,
        0.11,
        "Reference labels",
        ["highD lane IDs,", "highD neighbour IDs"],
        facecolor="#FFFFFF",
        edgecolor=PALETTE["validation_group"],
    )
    add_box(
        ax,
        0.31,
        0.205,
        0.18,
        0.11,
        "Reference outputs",
        ["reference relations,", "reference events,", "reference signatures"],
        facecolor="#FFFFFF",
        edgecolor=PALETTE["validation_group"],
    )
    add_arrow(ax, (0.23, 0.26), (0.31, 0.26), color=PALETTE["validation_group"])

    ax.text(
        0.30,
        0.145,
        "Reference labels are excluded from the method path and used only for comparison.",
        ha="center",
        va="center",
        fontsize=9.2,
        style="italic",
        color=PALETTE["ink"],
    )

    add_box(
        ax,
        0.82,
        0.41,
        0.15,
        0.20,
        "Agreement\nmetrics",
        [
            "frame: lane / neighbour",
            "event: cut-in $F_1$",
            "representation: exact",
            "match, Hamming",
        ],
        facecolor=PALETTE["metrics_fill"],
        edgecolor=PALETTE["metrics_edge"],
        fontsize=8.9,
    )

    add_arrow(
        ax,
        (x_positions[5] + box_w, y_top + box_h * 0.58),
        (0.82, 0.54),
        connectionstyle="arc3,rad=-0.03",
    )
    add_arrow(
        ax,
        (0.49, 0.26),
        (0.82, 0.47),
        color=PALETTE["validation_group"],
        connectionstyle="arc3,rad=0.02",
    )

    ax.text(
        0.895,
        0.64,
        "same output schema\ncompared across paths",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=PALETTE["ink"],
    )

    fig.tight_layout()
    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    render()
