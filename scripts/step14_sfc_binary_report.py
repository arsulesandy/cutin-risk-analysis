"""Step 14 report: decode and aggregate binary SFC codes into occupancy heatmaps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.encoding.sfc_binary import decode_grid_4x4_bits
from cutin_risk.io.step_reports import mirror_file_to_step, step_reports_dir, write_step_markdown
from cutin_risk.paths import step14_codes_csv_path

STAGE_ORDER = ["intention", "decision", "execution"]
STAGE_LABELS = {
    "intention": "Intention",
    "decision": "Decision",
    "execution": "Execution",
}
CELL_ORDER = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
CELL_LABELS = {
    (0, 0): "Ahead\nleft adj.",
    (0, 1): "Ahead\ncutter lane",
    (0, 2): "Ahead\ntarget lane",
    (1, 0): "Alongside\nleft adj.",
    (1, 2): "Alongside\ntarget lane",
    (2, 0): "Behind\nleft adj.",
    (2, 1): "Behind\ncutter lane",
    (2, 2): "Behind\ntarget lane",
}
POS_COLOR = "#c44e52"
NEG_COLOR = "#4c72b0"
ZERO_COLOR = "#8c8c8c"


def mean_grid(df: pd.DataFrame, *, order: str) -> np.ndarray:
    acc = np.zeros((3, 3), dtype=float)
    n = 0
    for code in df["code"].to_numpy(dtype=int):
        g4 = decode_grid_4x4_bits(int(code), order=order)  # 4x4
        g3 = g4[:3, :3]  # 3x3
        acc += g3
        n += 1
    return acc / max(n, 1)


def _load_pyplot(out_dir: Path):
    # Keep plotting reproducible in headless/sandboxed environments.
    cache_home = out_dir / ".cache"
    mpl_config = out_dir / ".mplconfig"
    cache_home.mkdir(parents=True, exist_ok=True)
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_home))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_grid(
    plt,
    g: np.ndarray,
    title: str,
    out: Path,
    *,
    diff_mode: bool = False,
    diff_limit: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    xtick_labels = ["left adj.", "cutter lane", "target lane"]
    ytick_labels = ["ahead", "alongside", "behind"]

    if diff_mode:
        from matplotlib.colors import TwoSlopeNorm

        scale = float(diff_limit) if diff_limit is not None else float(np.max(np.abs(g)))
        scale = max(scale, 1e-6)
        display = g.astype(float).copy()
        display[1, 1] = np.nan
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad(color="#f0f0f0")
        im = ax.imshow(
            display * 100.0,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-scale * 100.0, vcenter=0.0, vmax=scale * 100.0),
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.03)
        cbar.set_label("risk - non-risk occupancy difference (percentage points)")
    else:
        im = ax.imshow(g, vmin=0.0, vmax=1.0, cmap="YlGnBu")
        cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.03)
        cbar.set_label("occupancy probability")

    ax.set_title(title)
    ax.set_xticks([0, 1, 2], xtick_labels)
    ax.set_yticks([0, 1, 2], ytick_labels)

    # Emphasize the 3x3 semantic cells so small shifts remain legible.
    ax.set_xticks(np.arange(-0.5, 3.0, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, 3.0, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(3):
        for c in range(3):
            value = float(g[r, c])
            if diff_mode and r == 1 and c == 1:
                ax.text(
                    c,
                    r,
                    "cutter\nfixed",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="semibold",
                    color="#4a4a4a",
                )
                continue

            if diff_mode:
                label = f"{value * 100:+.1f}\npp"
                contrast_cutoff = max((diff_limit or 0.0) * 0.55, 0.02)
                text_color = "white" if abs(value) >= contrast_cutoff else "#1f1f1f"
            else:
                label = f"{value:.2f}"
                text_color = "white" if value >= 0.55 else "#1f1f1f"

            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="semibold",
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stage_diff_trends(plt, stage_diffs: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), sharex=True, sharey=True)
    axes = axes.flatten()
    stage_positions = np.arange(len(STAGE_ORDER))
    all_values = np.array(
        [stage_diffs[stage][r, c] * 100.0 for stage in STAGE_ORDER for (r, c) in CELL_ORDER],
        dtype=float,
    )
    y_limit = max(float(np.max(np.abs(all_values))) * 1.18, 3.0)

    for ax, (r, c) in zip(axes, CELL_ORDER):
        values = np.array([stage_diffs[stage][r, c] * 100.0 for stage in STAGE_ORDER], dtype=float)
        colors = [POS_COLOR if v > 0 else NEG_COLOR if v < 0 else ZERO_COLOR for v in values]

        ax.axhline(0.0, color="#aaaaaa", linewidth=1.2, linestyle="--", zorder=1)
        ax.plot(stage_positions, values, color="#303030", linewidth=2.0, zorder=2)
        ax.scatter(stage_positions, values, c=colors, s=70, zorder=3, edgecolors="white", linewidths=0.8)

        for x, y in zip(stage_positions, values):
            offset = 0.55 if y >= 0 else -0.75
            va = "bottom" if y >= 0 else "top"
            ax.text(
                x,
                y + offset,
                f"{y:+.1f}",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="semibold",
                color="#202020",
            )

        ax.set_title(CELL_LABELS[(r, c)], fontsize=11, fontweight="semibold")
        ax.set_xticks(stage_positions, [STAGE_LABELS[stage] for stage in STAGE_ORDER], rotation=20)
        ax.set_ylim(-y_limit, y_limit)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.9)
        ax.set_axisbelow(True)

    for idx in (0, 4):
        axes[idx].set_ylabel("difference (pp)")

    fig.suptitle(
        "How occupancy differences move across stages",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Positive values mean more occupancy in risky events; negative values mean more occupancy in non-risk events.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.95))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_diff_ranked(plt, stage_diffs: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 7.0), sharex=True)
    all_values = np.array(
        [stage_diffs[stage][r, c] * 100.0 for stage in STAGE_ORDER for (r, c) in CELL_ORDER],
        dtype=float,
    )
    x_limit = max(float(np.max(np.abs(all_values))) * 1.20, 3.0)

    for ax, stage in zip(axes, STAGE_ORDER):
        entries = [
            (CELL_LABELS[(r, c)].replace("\n", " "), stage_diffs[stage][r, c] * 100.0)
            for (r, c) in CELL_ORDER
        ]
        entries.sort(key=lambda item: item[1])
        labels = [item[0] for item in entries]
        values = np.array([item[1] for item in entries], dtype=float)
        positions = np.arange(len(entries))
        colors = [POS_COLOR if v > 0 else NEG_COLOR if v < 0 else ZERO_COLOR for v in values]

        ax.axvline(0.0, color="#777777", linewidth=1.2)
        ax.barh(positions, values, color=colors, edgecolor="none", height=0.72)
        ax.set_yticks(positions, labels)
        ax.set_title(f"{STAGE_LABELS[stage]} stage", fontsize=12, fontweight="semibold")
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.set_xlim(-x_limit, x_limit)

        for y, v in zip(positions, values):
            x = v + (0.35 if v >= 0 else -0.35)
            ha = "left" if v >= 0 else "right"
            ax.text(
                x,
                y,
                f"{v:+.1f}",
                va="center",
                ha=ha,
                fontsize=9,
                fontweight="semibold",
                color="#202020",
            )

    axes[0].set_ylabel("semantic cell")
    axes[1].set_xlabel("difference (percentage points)")
    fig.suptitle(
        "Which cells shift the most at each stage",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.95))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes-csv", default=str(step14_codes_csv_path()))
    ap.add_argument("--out-dir", default=str(step_reports_dir(14)))
    ap.add_argument(
        "--make-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate PNG heatmaps in addition to CSV grids.",
    )
    args = ap.parse_args()

    codes_csv = Path(args.codes_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(codes_csv)
    if df.empty:
        raise ValueError("codes csv is empty")

    order = str(df["sfc_order"].iloc[0])

    plt = _load_pyplot(out_dir) if args.make_plot else None

    stage_results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for stage in STAGE_ORDER:
        d_stage = df[df["stage"] == stage]
        if d_stage.empty:
            continue

        risk_mask = d_stage["risk_thw"].astype(bool)
        d_risk = d_stage[risk_mask]
        d_safe = d_stage[~risk_mask]

        g_risk = mean_grid(d_risk, order=order)
        g_safe = mean_grid(d_safe, order=order)
        g_diff = g_risk - g_safe

        stage_results.append((stage, g_safe, g_risk, g_diff))

    diff_limit = max((float(np.max(np.abs(g_diff))) for _, _, _, g_diff in stage_results), default=0.0)
    stage_diff_map = {stage: g_diff for stage, _, _, g_diff in stage_results}

    for stage, g_safe, g_risk, g_diff in stage_results:

        if plt is not None:
            stage_title = stage.capitalize()
            plot_grid(
                plt,
                g_safe,
                f"{stage_title} stage: non-risk mean occupancy",
                out_dir / f"{stage}_nonrisk.png",
            )
            plot_grid(
                plt,
                g_risk,
                f"{stage_title} stage: risk mean occupancy",
                out_dir / f"{stage}_risk.png",
            )
            plot_grid(
                plt,
                g_diff,
                f"{stage_title} stage: risk - non-risk",
                out_dir / f"{stage}_diff.png",
                diff_mode=True,
                diff_limit=diff_limit,
            )

        # also save raw numbers for the thesis
        np.savetxt(out_dir / f"{stage}_nonrisk.csv", g_safe, delimiter=",", fmt="%.6f")
        np.savetxt(out_dir / f"{stage}_risk.csv", g_risk, delimiter=",", fmt="%.6f")
        np.savetxt(out_dir / f"{stage}_diff.csv", g_diff, delimiter=",", fmt="%.6f")

    if plt is not None and stage_diff_map:
        plot_stage_diff_trends(plt, stage_diff_map, out_dir / "stage_diff_trends.png")
        plot_stage_diff_ranked(plt, stage_diff_map, out_dir / "stage_diff_ranked.png")

    canonical_dir = step_reports_dir(14)
    mirrored: list[Path] = []
    if out_dir.resolve() != canonical_dir.resolve():
        for p in sorted(out_dir.glob("*.csv")):
            mirrored.append(mirror_file_to_step(p, 14))
        for p in sorted(out_dir.glob("*.png")):
            mirrored.append(mirror_file_to_step(p, 14, kind="figures"))
    else:
        mirrored = sorted(out_dir.glob("*.csv")) + sorted(out_dir.glob("*.png"))

    details_md = write_step_markdown(
        14,
        "sfc_binary_report_details.md",
        [
            "# Step 14 Binary SFC Report",
            "",
            f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Input codes CSV: `{codes_csv}`",
            f"- Local report directory: `{out_dir}`",
            f"- Plot generation enabled: `{bool(args.make_plot)}`",
            f"- Artifacts mirrored: `{len(mirrored)}`",
        ],
    )

    print("Saved report to:", out_dir)
    print("Saved summary markdown:", details_md)


if __name__ == "__main__":
    main()
