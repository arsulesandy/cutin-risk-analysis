"""Exploratory NGSIM utilities for transferability and feasibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.detection.cutin import CutInOptions, detect_cutins
from cutin_risk.detection.lane_change import LaneChangeOptions, detect_lane_changes

USECOLS = [
    "Location",
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
    "v_Vel",
    "v_length",
    "v_Class",
]


@dataclass(frozen=True)
class NGSIMFeasibilityConfig:
    """Thresholds for the NGSIM exploratory feasibility scan."""

    chunksize: int = 500_000
    min_stable_before_frames: int = 10
    min_stable_after_frames: int = 10
    search_window_frames: int = 20
    max_relation_delay_frames: int = 6
    min_relation_frames: int = 5
    precheck_frames: int = 10
    thw_risk_threshold_s: float = 0.7


def build_options(
    config: NGSIMFeasibilityConfig | None = None,
) -> tuple[LaneChangeOptions, CutInOptions]:
    """Build lane-change and cut-in options for 10 Hz NGSIM data."""
    config = config or NGSIMFeasibilityConfig()
    lane_options = LaneChangeOptions(
        min_stable_before_frames=int(config.min_stable_before_frames),
        min_stable_after_frames=int(config.min_stable_after_frames),
        ignore_lane_ids=(0,),
    )
    cutin_options = CutInOptions(
        search_window_frames=int(config.search_window_frames),
        max_relation_delay_frames=int(config.max_relation_delay_frames),
        min_relation_frames=int(config.min_relation_frames),
        precheck_frames=int(config.precheck_frames),
        no_neighbor_ids=(0, -1),
        require_lane_match=True,
        require_preceding_consistency=True,
    )
    return lane_options, cutin_options


def load_location_table(
    csv_path: Path,
    *,
    location: str,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Load and clean one NGSIM location from the mixed CSV export."""
    parts: list[pd.DataFrame] = []
    raw_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=chunksize, thousands=","):
        sub = chunk[chunk["Location"] == location]
        if sub.empty:
            continue
        raw_rows += int(len(sub))
        parts.append(sub)

    if not parts:
        raise ValueError(f"No rows found for location={location!r} in {csv_path}")

    df = pd.concat(parts, ignore_index=True)
    df = df.rename(
        columns={
            "Vehicle_ID": "source_vehicle_id",
            "Frame_ID": "frame",
            "Global_Time": "time_ms",
            "Lane_ID": "laneId",
            "Preceding": "precedingId",
            "Following": "followingId",
            "Space_Headway": "space_headway_ft",
            "Time_Headway": "time_headway_s",
            "v_Vel": "speed_fps",
            "v_length": "length_ft",
            "v_Class": "vehicle_class",
        }
    )

    numeric_cols = [
        "source_vehicle_id",
        "frame",
        "time_ms",
        "laneId",
        "precedingId",
        "followingId",
        "space_headway_ft",
        "time_headway_s",
        "speed_fps",
        "length_ft",
        "vehicle_class",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["source_vehicle_id", "frame", "time_ms", "laneId"]).copy()
    for col in ["source_vehicle_id", "frame", "laneId", "precedingId", "followingId", "vehicle_class"]:
        df[col] = df[col].fillna(0).astype(int)

    df["time"] = df["time_ms"] / 1000.0
    source_vehicle_ids = int(df["source_vehicle_id"].nunique())

    df_dedup = df.drop_duplicates(["source_vehicle_id", "frame", "time_ms"], keep="first").copy()
    duplicate_rows_removed = int(len(df) - len(df_dedup))

    df = df_dedup.sort_values(["source_vehicle_id", "time_ms", "frame"], kind="mergesort").reset_index(drop=True)

    time_diff = df.groupby("source_vehicle_id", sort=False)["time_ms"].diff()
    df["track_instance"] = (
        time_diff.gt(1000).fillna(False).groupby(df["source_vehicle_id"]).cumsum().astype(int)
    )

    keys = pd.Index(list(zip(df["source_vehicle_id"].tolist(), df["track_instance"].tolist())))
    df["id"], _ = pd.factorize(keys, sort=False)
    df["id"] = df["id"].astype(int) + 1

    lookup = df[["source_vehicle_id", "time_ms", "id"]].drop_duplicates(["source_vehicle_id", "time_ms"])
    preceding_lookup = lookup.rename(
        columns={"source_vehicle_id": "precedingId", "id": "preceding_track_id"}
    )
    following_lookup = lookup.rename(
        columns={"source_vehicle_id": "followingId", "id": "following_track_id"}
    )
    df = df.merge(preceding_lookup, on=["precedingId", "time_ms"], how="left")
    df = df.merge(following_lookup, on=["followingId", "time_ms"], how="left")
    df["precedingId"] = df["preceding_track_id"].fillna(0).astype(int)
    df["followingId"] = df["following_track_id"].fillna(0).astype(int)
    df = df.drop(columns=["preceding_track_id", "following_track_id"])

    df = df.drop_duplicates(["id", "frame"], keep="first").copy()
    df = df.sort_values(["id", "frame"], kind="mergesort").reset_index(drop=True)

    meta = {
        "location": location,
        "raw_rows": int(raw_rows),
        "duplicate_rows_removed": duplicate_rows_removed,
        "cleaned_rows": int(len(df)),
        "source_vehicle_ids": source_vehicle_ids,
        "track_instances": int(df["id"].nunique()),
    }
    return df, meta


def event_rows(df: pd.DataFrame, cutins: list) -> pd.DataFrame:
    """Convert cut-in events into event-level summary rows with exploratory metrics."""
    indexed = df.set_index(["id", "frame"], drop=False)
    rows: list[dict[str, float | int]] = []

    for e in cutins:
        metrics: list[dict[str, float]] = []
        for frame in range(int(e.relation_start_frame), int(e.relation_end_frame) + 1):
            try:
                follower = indexed.loc[(int(e.follower_id), frame)]
                cutter = indexed.loc[(int(e.cutter_id), frame)]
            except KeyError:
                continue

            if isinstance(follower, pd.DataFrame):
                follower = follower.iloc[0]
            if isinstance(cutter, pd.DataFrame):
                cutter = cutter.iloc[0]

            spacing = float(follower["space_headway_ft"]) if pd.notna(follower["space_headway_ft"]) else np.nan
            cutter_length = float(cutter["length_ft"]) if pd.notna(cutter["length_ft"]) else np.nan
            gap_ft = spacing - cutter_length if np.isfinite(spacing) and np.isfinite(cutter_length) else np.nan

            thw_s = float(follower["time_headway_s"]) if pd.notna(follower["time_headway_s"]) else np.nan
            if np.isfinite(thw_s) and thw_s > 100.0:
                thw_s = np.nan

            vf = float(follower["speed_fps"]) if pd.notna(follower["speed_fps"]) else np.nan
            vl = float(cutter["speed_fps"]) if pd.notna(cutter["speed_fps"]) else np.nan
            closing_fps = vf - vl if np.isfinite(vf) and np.isfinite(vl) else np.nan

            if np.isfinite(gap_ft) and gap_ft > 0.0 and np.isfinite(closing_fps) and closing_fps > 1e-6:
                ttc_s = gap_ft / closing_fps
                drac_ftps2 = (closing_fps ** 2) / (2.0 * gap_ft)
            else:
                ttc_s = np.inf
                drac_ftps2 = np.inf

            metrics.append(
                {
                    "gap_ft": gap_ft,
                    "thw_s": thw_s,
                    "closing_fps": closing_fps,
                    "ttc_s": ttc_s,
                    "drac_ftps2": drac_ftps2,
                }
            )

        metric_df = pd.DataFrame(metrics)
        finite_ttc = metric_df["ttc_s"].replace([np.inf, -np.inf], np.nan)
        finite_drac = metric_df["drac_ftps2"].replace([np.inf, -np.inf], np.nan)

        rows.append(
            {
                "cutter_track_id": int(e.cutter_id),
                "follower_track_id": int(e.follower_id),
                "from_lane": int(e.from_lane),
                "to_lane": int(e.to_lane),
                "lane_change_start_frame": int(e.lane_change_start_frame),
                "relation_start_frame": int(e.relation_start_frame),
                "relation_end_frame": int(e.relation_end_frame),
                "relation_duration_frames": int(e.relation_end_frame - e.relation_start_frame + 1),
                "gap_ft_min": float(metric_df["gap_ft"].min(skipna=True)) if not metric_df.empty else np.nan,
                "gap_ft_median": float(metric_df["gap_ft"].median(skipna=True)) if not metric_df.empty else np.nan,
                "thw_s_min": float(metric_df["thw_s"].min(skipna=True)) if not metric_df.empty else np.nan,
                "ttc_s_min": float(finite_ttc.min(skipna=True)) if not metric_df.empty else np.nan,
                "drac_ftps2_max": float(finite_drac.max(skipna=True)) if not metric_df.empty else np.nan,
                "closing_fps_max": float(metric_df["closing_fps"].max(skipna=True)) if not metric_df.empty else np.nan,
            }
        )

    return pd.DataFrame(rows)


def analyze_location(
    csv_path: Path,
    *,
    location: str,
    config: NGSIMFeasibilityConfig | None = None,
) -> dict[str, object]:
    """Run the location-scoped feasibility scan and return dataframes plus summary."""
    config = config or NGSIMFeasibilityConfig()
    df, meta = load_location_table(csv_path, location=location, chunksize=int(config.chunksize))
    lane_options, cutin_options = build_options(config)

    lane_changes = detect_lane_changes(df, options=lane_options)
    cutins = detect_cutins(df, lane_changes, options=cutin_options)

    lane_df = pd.DataFrame([asdict(e) for e in lane_changes])
    cutin_df = event_rows(df, cutins)

    thw_valid = cutin_df["thw_s_min"].dropna() if "thw_s_min" in cutin_df.columns else pd.Series(dtype=float)
    finite_ttc = (
        cutin_df["ttc_s_min"].replace([np.inf, -np.inf], np.nan).dropna()
        if "ttc_s_min" in cutin_df.columns
        else pd.Series(dtype=float)
    )
    negative_gap_mask = cutin_df["gap_ft_min"] < 0.0 if "gap_ft_min" in cutin_df.columns else pd.Series(dtype=bool)

    summary: dict[str, object] = {
        **meta,
        "lane_changes": int(len(lane_df)),
        "cutin_candidates": int(len(cutin_df)),
        "cutin_ratio_pct": float(100.0 * len(cutin_df) / len(lane_df)) if len(lane_df) else np.nan,
        "thw_s_min_median": float(thw_valid.median()) if len(thw_valid) else np.nan,
        "thw_risky_pct": float(100.0 * (thw_valid < float(config.thw_risk_threshold_s)).mean())
        if len(thw_valid)
        else np.nan,
        "finite_ttc_pct": float(100.0 * len(finite_ttc) / len(cutin_df)) if len(cutin_df) else np.nan,
        "negative_gap_pct": float(100.0 * negative_gap_mask.mean()) if len(cutin_df) else np.nan,
    }

    return {
        "tracking_df": df,
        "lane_changes_df": lane_df,
        "cutin_events_df": cutin_df,
        "summary": summary,
    }
