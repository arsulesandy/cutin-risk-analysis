"""Interactive NGSIM playback adapter built on the existing visualizer UI.

This wrapper keeps the current highD visualizer untouched. It loads one NGSIM
location, reconstructs track instances, converts the data to the in-memory
shape expected by ``VisualizationPlot``, and optionally focuses on a small
event-centered frame window.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


_PRE_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_PARSER.add_argument("--show", default=True, type=_str_to_bool)
_PRE_ARGS, _ = _PRE_PARSER.parse_known_args()
if not _PRE_ARGS.show:
    os.environ.setdefault("MPLBACKEND", "Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

from read_csv import (
    BACK_SIGHT_DISTANCE,
    BBOX,
    CLASS,
    DHW,
    DRIVING_DIRECTION,
    FINAL_FRAME,
    FOLLOWING_ID,
    FRAME,
    FRAME_RATE,
    FRONT_SIGHT_DISTANCE,
    HEIGHT,
    ID,
    INITIAL_FRAME,
    LANE_ID,
    LEFT_ALONGSIDE_ID,
    LEFT_FOLLOWING_ID,
    LEFT_PRECEDING_ID,
    LOCATION_ID,
    LOWER_LANE_MARKINGS,
    MAX_X_VELOCITY,
    MEAN_X_VELOCITY,
    MIN_DHW,
    MIN_THW,
    MIN_TTC,
    MIN_X_VELOCITY,
    N_CARS,
    N_TRUCKS,
    N_VEHICLES,
    NUMBER_LANE_CHANGES,
    NUM_FRAMES,
    PRECEDING_ID,
    PRECEDING_X_VELOCITY,
    RIGHT_ALONGSIDE_ID,
    RIGHT_FOLLOWING_ID,
    RIGHT_PRECEDING_ID,
    SPEED_LIMIT,
    THW,
    TOTAL_DRIVEN_DISTANCE,
    TOTAL_DRIVEN_TIME,
    TRACK_ID,
    TRAVELED_DISTANCE,
    TTC,
    UPPER_LANE_MARKINGS,
    WEEKDAY,
    WIDTH,
    X_ACCELERATION,
    X_VELOCITY,
    Y_ACCELERATION,
    Y_VELOCITY,
)
from visualize_frame import VisualizationPlot


FEET_TO_METERS = 0.3048
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "data/raw/NGSIM/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260411.csv"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs/cache/ngsim_visualizer"
USECOLS = [
    "Location",
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "v_length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
]


def _default_cache_path(location: str) -> Path:
    return DEFAULT_CACHE_DIR / f"{location}_visualizer.pkl"


def create_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Visualize a small NGSIM freeway slice with the existing highD viewer."
    )
    parser.add_argument(
        "--input_csv",
        default=str(DEFAULT_INPUT_CSV),
        type=str,
        help="Path to the mixed NGSIM CSV export.",
    )
    parser.add_argument(
        "--location",
        default="us-101",
        type=str,
        help="NGSIM site name, for example us-101 or i-80.",
    )
    parser.add_argument(
        "--cache_path",
        default=None,
        type=str,
        help="Optional pickle cache for the cleaned location table.",
    )
    parser.add_argument(
        "--force_reload",
        default=False,
        type=_str_to_bool,
        help="Ignore any existing cache and rebuild from the raw CSV.",
    )
    parser.add_argument(
        "--chunksize",
        default=500_000,
        type=int,
        help="Chunk size used while scanning the raw NGSIM CSV.",
    )
    parser.add_argument(
        "--event_csv",
        default=None,
        type=str,
        help="Optional cut-in event summary CSV created by the feasibility scripts.",
    )
    parser.add_argument(
        "--event_index",
        default=0,
        type=int,
        help="Zero-based row index inside --event_csv.",
    )
    parser.add_argument(
        "--focus_track_id",
        default=None,
        type=int,
        help="Reconstructed NGSIM track id to center the window around.",
    )
    parser.add_argument(
        "--focus_source_vehicle_id",
        default=None,
        type=int,
        help="Raw NGSIM Vehicle_ID to center the window around when a reconstructed track id is not known.",
    )
    parser.add_argument(
        "--focus_frame",
        default=None,
        type=int,
        help="Optional original frame id used as the window center.",
    )
    parser.add_argument(
        "--frame_start",
        default=None,
        type=int,
        help="Optional original frame id for the start of the extracted window.",
    )
    parser.add_argument(
        "--frame_end",
        default=None,
        type=int,
        help="Optional original frame id for the end of the extracted window.",
    )
    parser.add_argument(
        "--window_before",
        default=20,
        type=int,
        help="Frames kept before the focus point or event.",
    )
    parser.add_argument(
        "--window_after",
        default=20,
        type=int,
        help="Frames kept after the focus point or event.",
    )
    parser.add_argument(
        "--export_path",
        default=None,
        type=str,
        help="Optional PNG path for saving a snapshot of the selected frame.",
    )
    parser.add_argument(
        "--snapshot_frame",
        default=None,
        type=int,
        help="Optional original frame id to export instead of the inferred focus frame.",
    )
    parser.add_argument(
        "--show",
        default=True,
        type=_str_to_bool,
        help="Open the interactive matplotlib window. Set false for headless snapshot export.",
    )

    parser.add_argument(
        "--plotBoundingBoxes",
        default=True,
        type=_str_to_bool,
        help="Draw vehicle boxes.",
    )
    parser.add_argument(
        "--plotDirectionTriangle",
        default=True,
        type=_str_to_bool,
        help="Draw direction triangles.",
    )
    parser.add_argument(
        "--plotTextAnnotation",
        default=True,
        type=_str_to_bool,
        help="Draw text labels above vehicles.",
    )
    parser.add_argument(
        "--plotDetailedLabel",
        default=False,
        type=_str_to_bool,
        help="Use detailed labels with class and speed.",
    )
    parser.add_argument(
        "--plotIdOnlyLabel",
        default=True,
        type=_str_to_bool,
        help="Draw the track id on the vehicle body and keep a short lane label above it.",
    )
    parser.add_argument(
        "--plotTrackingLines",
        default=False,
        type=_str_to_bool,
        help="Draw historical motion traces.",
    )
    parser.add_argument(
        "--plotClass",
        default=True,
        type=_str_to_bool,
        help="Show class inside detailed labels.",
    )
    parser.add_argument(
        "--plotVelocity",
        default=True,
        type=_str_to_bool,
        help="Show speed inside detailed labels.",
    )
    parser.add_argument(
        "--plotIDs",
        default=True,
        type=_str_to_bool,
        help="Show ids inside detailed labels.",
    )

    return vars(parser.parse_args())


def _load_location_table(csv_path: Path, location: str, chunksize: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=chunksize, thousands=","):
        sub = chunk[chunk["Location"] == location]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        raise ValueError(f"No rows found for location={location!r} in {csv_path}")

    df = pd.concat(parts, ignore_index=True)
    df = df.rename(
        columns={
            "Vehicle_ID": "source_vehicle_id",
            "Frame_ID": "frame",
            "Global_Time": "time_ms",
            "Local_X": "local_x_ft",
            "Local_Y": "local_y_ft",
            "v_length": "length_ft",
            "v_Width": "width_ft",
            "v_Class": "vehicle_class",
            "v_Vel": "speed_fps",
            "v_Acc": "accel_fps2",
            "Lane_ID": "lane_id",
            "Preceding": "preceding_source_id",
            "Following": "following_source_id",
            "Space_Headway": "space_headway_ft",
            "Time_Headway": "time_headway_s",
        }
    )

    numeric_cols = [
        "source_vehicle_id",
        "frame",
        "time_ms",
        "local_x_ft",
        "local_y_ft",
        "length_ft",
        "width_ft",
        "vehicle_class",
        "speed_fps",
        "accel_fps2",
        "lane_id",
        "preceding_source_id",
        "following_source_id",
        "space_headway_ft",
        "time_headway_s",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["source_vehicle_id", "frame", "time_ms", "local_x_ft", "local_y_ft", "lane_id"]
    ).copy()
    int_cols = [
        "source_vehicle_id",
        "frame",
        "time_ms",
        "vehicle_class",
        "lane_id",
        "preceding_source_id",
        "following_source_id",
    ]
    for col in int_cols:
        df[col] = df[col].fillna(0).astype(int)

    df = df.drop_duplicates(["source_vehicle_id", "frame", "time_ms"], keep="first").copy()
    df = df.sort_values(["source_vehicle_id", "time_ms", "frame"], kind="mergesort").reset_index(drop=True)

    time_gap = df.groupby("source_vehicle_id", sort=False)["time_ms"].diff()
    df["track_instance"] = (
        time_gap.gt(1000).fillna(False).groupby(df["source_vehicle_id"]).cumsum().astype(int)
    )
    keys = pd.Index(list(zip(df["source_vehicle_id"].tolist(), df["track_instance"].tolist())))
    df["track_id"], _ = pd.factorize(keys, sort=False)
    df["track_id"] = df["track_id"].astype(int) + 1

    lookup = df[["source_vehicle_id", "time_ms", "track_id"]].drop_duplicates(
        ["source_vehicle_id", "time_ms"]
    )
    preceding_lookup = lookup.rename(
        columns={"source_vehicle_id": "preceding_source_id", "track_id": "preceding_track_id"}
    )
    following_lookup = lookup.rename(
        columns={"source_vehicle_id": "following_source_id", "track_id": "following_track_id"}
    )
    df = df.merge(preceding_lookup, on=["preceding_source_id", "time_ms"], how="left")
    df = df.merge(following_lookup, on=["following_source_id", "time_ms"], how="left")
    df["precedingId"] = df["preceding_track_id"].fillna(0).astype(int)
    df["followingId"] = df["following_track_id"].fillna(0).astype(int)
    df = df.drop(columns=["preceding_track_id", "following_track_id"])

    df = df.drop_duplicates(["track_id", "frame"], keep="first").copy()
    df = df.sort_values(["track_id", "frame"], kind="mergesort").reset_index(drop=True)

    df["local_x_m"] = df["local_x_ft"] * FEET_TO_METERS
    df["local_y_m"] = df["local_y_ft"] * FEET_TO_METERS
    df["length_m"] = df["length_ft"].fillna(0.0) * FEET_TO_METERS
    df["width_m"] = df["width_ft"].fillna(0.0) * FEET_TO_METERS
    df["speed_mps"] = df["speed_fps"].fillna(0.0) * FEET_TO_METERS
    df["accel_mps2"] = df["accel_fps2"].fillna(0.0) * FEET_TO_METERS
    df["space_headway_m"] = df["space_headway_ft"] * FEET_TO_METERS
    df["time_headway_s"] = df["time_headway_s"].mask(df["time_headway_s"] > 100.0)

    return df


def load_or_build_location_table(arguments: dict) -> pd.DataFrame:
    csv_path = Path(arguments["input_csv"]).resolve()
    cache_path = (
        Path(arguments["cache_path"]).resolve()
        if arguments.get("cache_path")
        else _default_cache_path(arguments["location"]).resolve()
    )

    if cache_path.exists() and not arguments.get("force_reload", False):
        print(f"Loading cached NGSIM table from {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"Scanning raw NGSIM CSV for location '{arguments['location']}'")
    df = _load_location_table(csv_path, arguments["location"], int(arguments["chunksize"]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f"Cached cleaned location table to {cache_path}")
    return df


def _class_name(raw_value: int) -> str:
    mapping = {
        1: "Motorcycle",
        2: "Car",
        3: "Truck",
    }
    return mapping.get(int(raw_value), f"Class{int(raw_value)}")


def _resolve_focus_track_id(df: pd.DataFrame, arguments: dict) -> int | None:
    if arguments.get("focus_track_id") is not None:
        track_id = int(arguments["focus_track_id"])
        if track_id not in set(df["track_id"].unique()):
            raise ValueError(f"focus_track_id={track_id} is not present in the selected location table")
        return track_id

    raw_source_id = arguments.get("focus_source_vehicle_id")
    if raw_source_id is None:
        return None

    candidates = (
        df.loc[df["source_vehicle_id"] == int(raw_source_id), ["track_id", "frame"]]
        .groupby("track_id")
        .size()
        .sort_values(ascending=False)
    )
    if candidates.empty:
        raise ValueError(f"focus_source_vehicle_id={raw_source_id} is not present in the selected location")
    chosen = int(candidates.index[0])
    print(
        f"Resolved raw Vehicle_ID {raw_source_id} to reconstructed track_id {chosen} "
        f"(longest matching track instance)."
    )
    return chosen


def _lane_change_focus_frame(track_df: pd.DataFrame) -> int:
    lane_diff = track_df["lane_id"].ne(track_df["lane_id"].shift())
    changed = track_df.loc[lane_diff & track_df["lane_id"].shift().notna(), "frame"]
    if not changed.empty:
        return int(changed.iloc[0])
    frames = track_df["frame"].to_numpy()
    return int(frames[len(frames) // 2])


def _load_event_row(event_csv: Path, event_index: int) -> pd.Series:
    event_df = pd.read_csv(event_csv)
    if event_index < 0 or event_index >= len(event_df):
        raise IndexError(f"event_index={event_index} is outside the valid range 0..{len(event_df) - 1}")
    return event_df.iloc[event_index]


def resolve_window(df: pd.DataFrame, arguments: dict) -> dict[str, int | None]:
    focus_track_id = _resolve_focus_track_id(df, arguments)
    focus_frame = arguments.get("focus_frame")
    frame_start = arguments.get("frame_start")
    frame_end = arguments.get("frame_end")
    snapshot_frame = arguments.get("snapshot_frame")
    event_row = None

    if arguments.get("event_csv"):
        event_row = _load_event_row(Path(arguments["event_csv"]).resolve(), int(arguments["event_index"]))
        focus_track_id = int(event_row["cutter_track_id"])
        if frame_start is None:
            frame_start = int(event_row["relation_start_frame"]) - int(arguments["window_before"])
        if frame_end is None:
            frame_end = int(event_row["relation_end_frame"]) + int(arguments["window_after"])
        if focus_frame is None:
            focus_frame = int(event_row["lane_change_start_frame"])
        if snapshot_frame is None:
            snapshot_frame = int(round((event_row["relation_start_frame"] + event_row["relation_end_frame"]) / 2.0))

    if focus_track_id is not None and focus_frame is None:
        track_df = df.loc[df["track_id"] == int(focus_track_id)].sort_values("frame")
        if track_df.empty:
            raise ValueError(f"Focused track_id={focus_track_id} has no rows in the selected location")
        focus_frame = _lane_change_focus_frame(track_df)

    global_frame_min = int(df["frame"].min())
    global_frame_max = int(df["frame"].max())
    if focus_frame is not None:
        if frame_start is None:
            frame_start = int(focus_frame) - int(arguments["window_before"])
        if frame_end is None:
            frame_end = int(focus_frame) + int(arguments["window_after"])

    if frame_start is None:
        frame_start = global_frame_min
    if frame_end is None:
        frame_end = min(global_frame_max, int(frame_start) + int(arguments["window_before"]) + int(arguments["window_after"]))

    frame_start = max(global_frame_min, int(frame_start))
    frame_end = min(global_frame_max, int(frame_end))
    if frame_end < frame_start:
        raise ValueError("Resolved frame window is empty")

    if snapshot_frame is None:
        snapshot_frame = focus_frame if focus_frame is not None else frame_start
    snapshot_frame = max(frame_start, min(frame_end, int(snapshot_frame)))

    return {
        "focus_track_id": focus_track_id,
        "focus_frame": focus_frame,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "snapshot_frame": snapshot_frame,
        "event_row": event_row,
    }


def _derive_lane_boundaries(full_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lane_centers = full_df.groupby("lane_id", sort=False)["local_x_m"].median().sort_values()
    if lane_centers.empty:
        raise ValueError("Cannot infer lane boundaries because the selected table is empty")

    centers = lane_centers.to_numpy(dtype=float)
    if len(centers) == 1:
        typical_width = max(float(full_df["width_m"].median()), 3.5)
        lower = np.array([centers[0] - typical_width / 2.0, centers[0] + typical_width / 2.0], dtype=float)
    else:
        midpoints = (centers[:-1] + centers[1:]) / 2.0
        first = centers[0] - (midpoints[0] - centers[0])
        last = centers[-1] + (centers[-1] - midpoints[-1])
        lower = np.concatenate([[first], midpoints, [last]]).astype(float)

    typical_gap = float(np.median(np.diff(lower))) if len(lower) > 2 else max(float(full_df["width_m"].median()), 3.5)
    if not np.isfinite(typical_gap) or typical_gap <= 0.0:
        typical_gap = 3.7
    upper = np.array([lower[0] - typical_gap, lower[0]], dtype=float)
    return upper, lower


def _prepare_window_dataframe(full_df: pd.DataFrame, window: dict[str, int | None]) -> pd.DataFrame:
    frame_start = int(window["frame_start"])
    frame_end = int(window["frame_end"])
    window_df = full_df.loc[full_df["frame"].between(frame_start, frame_end)].copy()
    if window_df.empty:
        raise ValueError(f"No rows found inside frame window [{frame_start}, {frame_end}]")

    focus_track_id = window.get("focus_track_id")
    anchor_frame = int(window.get("snapshot_frame") or frame_start)
    x_offset = None
    if focus_track_id is not None:
        anchor_rows = window_df.loc[
            (window_df["track_id"] == int(focus_track_id)) & (window_df["frame"] == anchor_frame)
        ]
        if not anchor_rows.empty:
            anchor = anchor_rows.iloc[0]
            anchor_center_x = float(anchor["local_y_m"] - (anchor["length_m"] / 2.0))
            x_offset = max(0.0, anchor_center_x - 200.0)
    if x_offset is None:
        x_min = float((window_df["local_y_m"] - window_df["length_m"]).min())
        x_offset = max(0.0, x_min - 5.0)

    window_df["bbox_x_m"] = window_df["local_y_m"] - window_df["length_m"] - x_offset
    window_df["bbox_y_m"] = window_df["local_x_m"] - (window_df["width_m"] / 2.0)
    window_df["frame_local"] = window_df["frame"] - frame_start + 1

    leader_info = window_df[["track_id", "time_ms", "length_m", "speed_mps"]].rename(
        columns={
            "track_id": "precedingId",
            "length_m": "preceding_length_m",
            "speed_mps": "preceding_speed_mps",
        }
    )
    window_df = window_df.merge(leader_info, on=["precedingId", "time_ms"], how="left")
    window_df["dhw_m"] = window_df["space_headway_m"] - window_df["preceding_length_m"]
    missing_leader = window_df["precedingId"] <= 0
    window_df.loc[missing_leader, "dhw_m"] = np.nan
    window_df["preceding_x_velocity_mps"] = window_df["preceding_speed_mps"]

    closing_speed = window_df["speed_mps"] - window_df["preceding_speed_mps"]
    window_df["ttc_s"] = np.where(
        (window_df["dhw_m"] > 0.0) & np.isfinite(closing_speed) & (closing_speed > 1e-3),
        window_df["dhw_m"] / closing_speed,
        np.inf,
    )
    return window_df


def _resample_track(track_df: pd.DataFrame) -> pd.DataFrame:
    start_frame = int(track_df["frame_local"].min())
    end_frame = int(track_df["frame_local"].max())
    full_index = np.arange(start_frame, end_frame + 1, dtype=int)

    base = (
        track_df.sort_values("frame_local")
        .drop_duplicates("frame_local", keep="first")
        .set_index("frame_local")
        .reindex(full_index)
    )
    base.index.name = "frame_local"
    base["track_id"] = int(track_df["track_id"].iloc[0])
    base["source_vehicle_id"] = int(track_df["source_vehicle_id"].iloc[0])
    base["frame"] = base["frame"].interpolate(limit_direction="both").round().astype(int)
    base["time_ms"] = base["time_ms"].interpolate(limit_direction="both").round().astype(int)

    numeric_cols = [
        "bbox_x_m",
        "bbox_y_m",
        "length_m",
        "width_m",
        "speed_mps",
        "accel_mps2",
        "dhw_m",
        "time_headway_s",
        "ttc_s",
        "preceding_x_velocity_mps",
    ]
    for col in numeric_cols:
        base[col] = base[col].astype(float).interpolate(limit_direction="both")

    fill_int_cols = ["lane_id", "precedingId", "followingId", "vehicle_class"]
    for col in fill_int_cols:
        base[col] = base[col].ffill().bfill().fillna(0).astype(int)

    zero_neighbor_cols = [
        LEFT_PRECEDING_ID,
        LEFT_ALONGSIDE_ID,
        LEFT_FOLLOWING_ID,
        RIGHT_PRECEDING_ID,
        RIGHT_ALONGSIDE_ID,
        RIGHT_FOLLOWING_ID,
    ]
    for col in zero_neighbor_cols:
        base[col] = 0

    dt = np.diff(base["time_ms"].to_numpy(dtype=float)) / 1000.0
    dt = np.where(np.isfinite(dt) & (dt > 1e-6), dt, 0.1)
    center_x = base["bbox_x_m"].to_numpy(dtype=float) + (base["length_m"].to_numpy(dtype=float) / 2.0)
    center_y = base["bbox_y_m"].to_numpy(dtype=float) + (base["width_m"].to_numpy(dtype=float) / 2.0)

    if len(base) > 1:
        x_velocity = np.gradient(center_x, edge_order=1) / np.r_[dt[0], dt]
        y_velocity = np.gradient(center_y, edge_order=1) / np.r_[dt[0], dt]
        x_acceleration = np.gradient(x_velocity, edge_order=1) / np.r_[dt[0], dt]
        y_acceleration = np.gradient(y_velocity, edge_order=1) / np.r_[dt[0], dt]
    else:
        x_velocity = np.array([float(base["speed_mps"].iloc[0])], dtype=float)
        y_velocity = np.array([0.0], dtype=float)
        x_acceleration = np.array([float(base["accel_mps2"].iloc[0])], dtype=float)
        y_acceleration = np.array([0.0], dtype=float)

    base["xVelocity"] = x_velocity
    base["yVelocity"] = y_velocity
    base["xAcceleration"] = x_acceleration
    base["yAcceleration"] = y_acceleration
    base["frame_local"] = base.index.astype(int)
    return base.reset_index(drop=True)


def build_visualizer_payload(
    full_df: pd.DataFrame,
    window_df: pd.DataFrame,
    arguments: dict,
) -> tuple[list[dict], dict[int, dict], dict]:
    upper_lane_markings, lower_lane_markings = _derive_lane_boundaries(full_df)

    tracks: list[dict] = []
    static_info: dict[int, dict] = {}
    num_cars = 0
    num_trucks = 0
    total_distance = 0.0
    total_time = 0.0

    for track_id, group in window_df.groupby("track_id", sort=False):
        resampled = _resample_track(group)
        bbox = np.column_stack(
            [
                resampled["bbox_x_m"].to_numpy(dtype=float),
                resampled["bbox_y_m"].to_numpy(dtype=float),
                resampled["length_m"].to_numpy(dtype=float),
                resampled["width_m"].to_numpy(dtype=float),
            ]
        )

        frames_local = resampled["frame_local"].to_numpy(dtype=int)
        lane_ids = resampled["lane_id"].to_numpy(dtype=int)
        track = {
            TRACK_ID: np.int64(track_id),
            FRAME: frames_local,
            BBOX: bbox,
            X_VELOCITY: resampled["xVelocity"].to_numpy(dtype=float),
            Y_VELOCITY: resampled["yVelocity"].to_numpy(dtype=float),
            X_ACCELERATION: resampled["xAcceleration"].to_numpy(dtype=float),
            Y_ACCELERATION: resampled["yAcceleration"].to_numpy(dtype=float),
            FRONT_SIGHT_DISTANCE: np.zeros(len(resampled), dtype=float),
            BACK_SIGHT_DISTANCE: np.zeros(len(resampled), dtype=float),
            THW: resampled["time_headway_s"].to_numpy(dtype=float),
            TTC: resampled["ttc_s"].to_numpy(dtype=float),
            DHW: resampled["dhw_m"].to_numpy(dtype=float),
            PRECEDING_X_VELOCITY: resampled["preceding_x_velocity_mps"].to_numpy(dtype=float),
            PRECEDING_ID: resampled["precedingId"].to_numpy(dtype=int),
            FOLLOWING_ID: resampled["followingId"].to_numpy(dtype=int),
            LEFT_PRECEDING_ID: np.zeros(len(resampled), dtype=int),
            LEFT_ALONGSIDE_ID: np.zeros(len(resampled), dtype=int),
            LEFT_FOLLOWING_ID: np.zeros(len(resampled), dtype=int),
            RIGHT_PRECEDING_ID: np.zeros(len(resampled), dtype=int),
            RIGHT_ALONGSIDE_ID: np.zeros(len(resampled), dtype=int),
            RIGHT_FOLLOWING_ID: np.zeros(len(resampled), dtype=int),
            LANE_ID: lane_ids,
        }
        tracks.append(track)

        vehicle_class_code = int(resampled["vehicle_class"].mode(dropna=True).iloc[0]) if not resampled["vehicle_class"].dropna().empty else 0
        class_name = _class_name(vehicle_class_code)
        if class_name == "Car":
            num_cars += 1
        elif class_name == "Truck":
            num_trucks += 1

        finite_dhw = pd.Series(track[DHW]).replace([np.inf, -np.inf], np.nan).dropna()
        finite_thw = pd.Series(track[THW]).replace([np.inf, -np.inf], np.nan).dropna()
        finite_ttc = pd.Series(track[TTC]).replace([np.inf, -np.inf], np.nan).dropna()
        traveled_distance = float(abs((bbox[-1, 0] + bbox[-1, 2] / 2.0) - (bbox[0, 0] + bbox[0, 2] / 2.0)))
        total_distance += traveled_distance
        total_time += max(0.0, (float(resampled["time_ms"].iloc[-1]) - float(resampled["time_ms"].iloc[0])) / 1000.0)

        static_info[int(track_id)] = {
            TRACK_ID: int(track_id),
            WIDTH: float(np.nanmedian(resampled["width_m"])),
            HEIGHT: float(np.nanmedian(resampled["length_m"])),
            INITIAL_FRAME: int(frames_local[0]),
            FINAL_FRAME: int(frames_local[-1]) + 1,
            NUM_FRAMES: int(len(frames_local)),
            CLASS: class_name,
            DRIVING_DIRECTION: 2.0,
            TRAVELED_DISTANCE: traveled_distance,
            MIN_X_VELOCITY: float(np.nanmin(track[X_VELOCITY])),
            MAX_X_VELOCITY: float(np.nanmax(track[X_VELOCITY])),
            MEAN_X_VELOCITY: float(np.nanmean(track[X_VELOCITY])),
            MIN_TTC: float(finite_ttc.min()) if not finite_ttc.empty else np.nan,
            MIN_THW: float(finite_thw.min()) if not finite_thw.empty else np.nan,
            MIN_DHW: float(finite_dhw.min()) if not finite_dhw.empty else np.nan,
            NUMBER_LANE_CHANGES: int(np.count_nonzero(np.diff(lane_ids))),
        }

    meta_dictionary = {
        ID: 1,
        FRAME_RATE: 10,
        LOCATION_ID: arguments["location"],
        SPEED_LIMIT: float("nan"),
        WEEKDAY: "",
        TOTAL_DRIVEN_DISTANCE: float(total_distance),
        TOTAL_DRIVEN_TIME: float(total_time),
        N_VEHICLES: int(len(static_info)),
        N_CARS: int(num_cars),
        N_TRUCKS: int(num_trucks),
        UPPER_LANE_MARKINGS: np.asarray(upper_lane_markings, dtype=float),
        LOWER_LANE_MARKINGS: np.asarray(lower_lane_markings, dtype=float),
    }
    return tracks, static_info, meta_dictionary


def build_visualizer_arguments(arguments: dict, window: dict[str, int | None]) -> dict:
    return {
        "recording_id": arguments["location"],
        "input_path": str(Path(arguments["input_csv"]).resolve()),
        "input_static_path": None,
        "input_meta_path": None,
        "pickle_path": None,
        "sfc_codes_csv": None,
        "sfc_codes_canonical": False,
        "visualize": bool(arguments["show"]),
        "background_image": None,
        "plotBoundingBoxes": bool(arguments["plotBoundingBoxes"]),
        "plotDirectionTriangle": bool(arguments["plotDirectionTriangle"]),
        "plotTextAnnotation": bool(arguments["plotTextAnnotation"]),
        "plotDetailedLabel": bool(arguments["plotDetailedLabel"]),
        "plotIdOnlyLabel": bool(arguments["plotIdOnlyLabel"]),
        "plotTrackingLines": bool(arguments["plotTrackingLines"]),
        "plotClass": bool(arguments["plotClass"]),
        "plotVelocity": bool(arguments["plotVelocity"]),
        "plotIDs": bool(arguments["plotIDs"]),
        "save_as_pickle": False,
        "ngsim_window_frame_start": int(window["frame_start"]),
        "ngsim_window_frame_end": int(window["frame_end"]),
    }


def _update_header_for_ngsim(plot: VisualizationPlot, arguments: dict, window: dict[str, int | None]):
    header_texts = list(getattr(plot.ax_header, "texts", []))
    if len(header_texts) >= 3:
        header_texts[0].set_text(f"NGSIM Scenario Intelligence Console | {arguments['location']}")
        header_texts[1].set_text(
            "Controls: Space play/pause | Left/Right +/-1 | Up/Down +/-10 | Click vehicle for details"
        )
        header_texts[2].set_text(
            "Window: frames {}-{} | Focus track: {}".format(
                int(window["frame_start"]),
                int(window["frame_end"]),
                window["focus_track_id"] if window["focus_track_id"] is not None else "-",
            )
        )


def _apply_ngsim_viewport(plot: VisualizationPlot, window_df: pd.DataFrame, meta_dictionary: dict):
    y_min = min(
        float(window_df["bbox_y_m"].min()) - 1.0,
        float(np.min(meta_dictionary[UPPER_LANE_MARKINGS])) - 1.0,
    )
    y_max = max(
        float((window_df["bbox_y_m"] + window_df["width_m"]).max()) + 1.0,
        float(np.max(meta_dictionary[LOWER_LANE_MARKINGS])) + 1.0,
    )
    plot.ax.set_xlim(0.0, 400.0)
    plot.ax.set_ylim(-y_max, -y_min)
    plot.default_xlim = tuple(plot.ax.get_xlim())
    plot.default_ylim = tuple(plot.ax.get_ylim())


def _save_snapshot(plot: VisualizationPlot, export_path: Path, local_frame: int):
    plot.current_frame = int(local_frame)
    plot.trigger_update()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    plot.fig.savefig(export_path, bbox_inches="tight", facecolor=plot.fig.get_facecolor())
    print(f"Saved NGSIM visualizer snapshot to {export_path}")


def main():
    arguments = create_args()
    full_df = load_or_build_location_table(arguments)
    window = resolve_window(full_df, arguments)
    window_df = _prepare_window_dataframe(full_df, window)
    tracks, static_info, meta_dictionary = build_visualizer_payload(full_df, window_df, arguments)
    visualizer_arguments = build_visualizer_arguments(arguments, window)

    print(
        "Prepared NGSIM slice | location={} | original frames {}-{} | local frames 1-{} | tracks={}".format(
            arguments["location"],
            int(window["frame_start"]),
            int(window["frame_end"]),
            int(window_df["frame_local"].max()),
            len(tracks),
        )
    )

    visualization_plot = VisualizationPlot(
        visualizer_arguments,
        tracks,
        static_info,
        meta_dictionary,
        recording_loader=None,
        recording_options=None,
    )
    _update_header_for_ngsim(visualization_plot, arguments, window)
    _apply_ngsim_viewport(visualization_plot, window_df, meta_dictionary)

    local_snapshot_frame = int(window["snapshot_frame"]) - int(window["frame_start"]) + 1
    local_snapshot_frame = max(1, min(local_snapshot_frame, int(window_df["frame_local"].max())))
    if arguments.get("export_path"):
        _save_snapshot(visualization_plot, Path(arguments["export_path"]).resolve(), local_snapshot_frame)

    if arguments["show"]:
        visualization_plot.current_frame = int(local_snapshot_frame)
        visualization_plot.trigger_update()
        VisualizationPlot.show()


if __name__ == "__main__":
    main()
