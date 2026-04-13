#!/usr/bin/env python3
"""Step 21: exploratory NGSIM freeway feasibility report for thesis support."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cutin_risk.datasets.ngsim.feasibility import NGSIMFeasibilityConfig, analyze_location
from cutin_risk.io.step_reports import mirror_file_to_step, step_figures_dir, step_reports_dir, write_step_markdown
from cutin_risk.paths import project_root

STEP_ID = 21
DEFAULT_DATASET = (
    project_root()
    / "data"
    / "raw"
    / "NGSIM"
    / "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260411.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 21: NGSIM freeway feasibility summary.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--locations", nargs="+", default=["us-101", "i-80"])
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--min-stable-before", type=int, default=10)
    parser.add_argument("--min-stable-after", type=int, default=10)
    parser.add_argument("--search-window", type=int, default=20)
    parser.add_argument("--max-relation-delay", type=int, default=6)
    parser.add_argument("--min-relation-frames", type=int, default=5)
    parser.add_argument("--precheck-frames", type=int, default=10)
    return parser.parse_args()


def _plot_counts(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(summary_df))
    width = 0.34

    lane = summary_df["lane_changes"].to_numpy(dtype=float)
    cutin = summary_df["cutin_candidates"].to_numpy(dtype=float)

    ax.bar(x - width / 2, lane, width=width, label="Lane changes", color="#5b8e7d")
    ax.bar(x + width / 2, cutin, width=width, label="Cut-in candidates", color="#d17b49")

    ax.set_xticks(x, labels=summary_df["location"].tolist())
    ax.set_ylabel("Event count")
    ax.set_title("Exploratory NGSIM freeway feasibility check", weight="bold")
    ax.legend(frameon=False)
    ax.set_ylim(0.0, max(lane.max(), cutin.max()) * 1.22)

    for idx, row in summary_df.reset_index(drop=True).iterrows():
        ax.text(
            x[idx] + width / 2,
            float(row["cutin_candidates"]) + max(lane.max(), cutin.max()) * 0.03,
            f"{float(row['cutin_ratio_pct']):.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.text(
        0.0,
        -0.18,
        "Labels above the cut-in bars show cut-ins / lane changes.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    cols = [str(c) for c in df.columns.tolist()]
    header = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return [header, rule, *rows]


def main() -> None:
    args = parse_args()
    reports_dir = step_reports_dir(STEP_ID)
    figures_dir = step_figures_dir(STEP_ID)

    config = NGSIMFeasibilityConfig(
        chunksize=int(args.chunksize),
        min_stable_before_frames=int(args.min_stable_before),
        min_stable_after_frames=int(args.min_stable_after),
        search_window_frames=int(args.search_window),
        max_relation_delay_frames=int(args.max_relation_delay),
        min_relation_frames=int(args.min_relation_frames),
        precheck_frames=int(args.precheck_frames),
    )

    summary_rows: list[dict[str, object]] = []
    for location in args.locations:
        result = analyze_location(args.csv, location=str(location), config=config)
        lane_df = result["lane_changes_df"]
        cutin_df = result["cutin_events_df"]
        summary = dict(result["summary"])

        lane_csv = reports_dir / f"{location}_lane_changes.csv"
        cutin_csv = reports_dir / f"{location}_cutin_events.csv"
        lane_df.to_csv(lane_csv, index=False)
        cutin_df.to_csv(cutin_csv, index=False)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values("location").reset_index(drop=True)
    summary_csv = reports_dir / "ngsim_freeway_feasibility_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    fig_path = figures_dir / "ngsim_feasibility_counts.png"
    _plot_counts(summary_df, fig_path)
    canonical_fig = mirror_file_to_step(fig_path, STEP_ID, kind="figures")

    latex_copy = project_root() / "latex" / "ngsim_feasibility_counts.png"
    shutil.copy2(canonical_fig, latex_copy)

    details = [
        "# Step 21 NGSIM Feasibility Report",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Source CSV: `{args.csv}`",
        f"- Locations: `{', '.join(map(str, args.locations))}`",
        "- Scope: freeway subsets only (`us-101`, `i-80`) to remain aligned with the highway cut-in problem.",
        "- Key cleaning assumptions:",
        "  - process one location at a time",
        "  - drop exact duplicate rows on `(Vehicle_ID, Frame_ID, Global_Time)`",
        "  - split repeated source ids when `Global_Time` jumps by more than 1 second",
        "  - remap `Preceding` and `Following` to the split track-instance ids",
        "  - treat `Time_Headway > 100 s` as placeholder / invalid",
        "- Important limitation: this is an exploratory transferability check, not a second full benchmark.",
        f"- Summary CSV: `{summary_csv}`",
        f"- Figure: `{canonical_fig}`",
        f"- Thesis figure copy: `{latex_copy}`",
        "",
        "## Summary table",
        "",
        *_markdown_table(summary_df),
    ]
    details_md = write_step_markdown(STEP_ID, "ngsim_feasibility_details.md", details)

    print("== Step 21: NGSIM freeway feasibility report ==")
    print(summary_df.to_string(index=False))
    print("Saved summary:", summary_csv)
    print("Saved figure:", canonical_fig)
    print("Copied thesis figure:", latex_copy)
    print("Saved markdown:", details_md)


if __name__ == "__main__":
    main()
