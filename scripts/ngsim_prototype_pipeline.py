#!/usr/bin/env python3
"""Minimal CLI wrapper for the exploratory NGSIM feasibility scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from cutin_risk.datasets.ngsim.feasibility import NGSIMFeasibilityConfig, analyze_location


DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "NGSIM"
    / "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260411.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "ngsim_prototype"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype NGSIM lane-change/cut-in scan.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--location", type=str, default="us-101")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--min-stable-before", type=int, default=10)
    parser.add_argument("--min-stable-after", type=int, default=10)
    parser.add_argument("--search-window", type=int, default=20)
    parser.add_argument("--max-relation-delay", type=int, default=6)
    parser.add_argument("--min-relation-frames", type=int, default=5)
    parser.add_argument("--precheck-frames", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = NGSIMFeasibilityConfig(
        chunksize=int(args.chunksize),
        min_stable_before_frames=int(args.min_stable_before),
        min_stable_after_frames=int(args.min_stable_after),
        search_window_frames=int(args.search_window),
        max_relation_delay_frames=int(args.max_relation_delay),
        min_relation_frames=int(args.min_relation_frames),
        precheck_frames=int(args.precheck_frames),
    )
    result = analyze_location(args.csv, location=str(args.location), config=config)
    lane_df = result["lane_changes_df"]
    cutin_df = result["cutin_events_df"]
    summary = result["summary"]

    lane_path = args.output_dir / f"{args.location}_lane_changes.csv"
    cutin_path = args.output_dir / f"{args.location}_cutin_events.csv"
    lane_df.to_csv(lane_path, index=False)
    cutin_df.to_csv(cutin_path, index=False)

    print(f"Location: {summary['location']}")
    print(f"Rows loaded: {summary['cleaned_rows']}")
    print(f"Track instances: {summary['track_instances']}")
    print(f"Lane changes: {summary['lane_changes']}")
    print(f"Cut-in candidates: {summary['cutin_candidates']}")
    if not cutin_df.empty:
        print("Cut-in summary:")
        print(cutin_df[["gap_ft_min", "thw_s_min", "ttc_s_min", "drac_ftps2_max"]].describe().to_string())
    print(f"Lane-change CSV: {lane_path}")
    print(f"Cut-in CSV: {cutin_path}")


if __name__ == "__main__":
    main()
