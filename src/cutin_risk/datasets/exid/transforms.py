"""Transform raw exiD recordings into a compact analysis table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from cutin_risk.detection.events import LaneChangeEvent

from .reader import ExiDRecording


MOTOR_VEHICLE_CLASSES = frozenset({"car", "van", "truck", "motorcycle"})


@dataclass(frozen=True)
class BuildOptions:
    """Options for building the exiD exploratory tracking table."""

    keep_optional_tracks_columns: bool = True
    keep_only_motor_vehicles: bool = True


def _get_frame_rate(recording_meta: pd.DataFrame) -> float:
    if recording_meta.empty:
        raise ValueError("recordingMeta is empty")
    frame_rate = float(recording_meta.loc[0, "frameRate"])
    if frame_rate <= 0:
        raise ValueError(f"Invalid frameRate: {frame_rate}")
    return frame_rate


def _assert_unique_key(df: pd.DataFrame, keys: Iterable[str], name: str) -> None:
    dup = df.duplicated(list(keys), keep=False)
    if dup.any():
        examples = df.loc[dup, list(keys)].head(10).to_dict(orient="records")
        raise ValueError(f"{name} contains duplicate keys for {list(keys)}. Examples: {examples}")


def _parse_first_numeric(value, *, default: float = np.nan) -> float:
    if pd.isna(value):
        return float(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    token = str(value).strip()
    if not token:
        return float(default)
    head = token.split(";", 1)[0].strip()
    if not head:
        return float(default)
    try:
        return float(head)
    except ValueError:
        return float(default)


def _parse_first_int(value, *, default: int = 0) -> int:
    parsed = _parse_first_numeric(value, default=float(default))
    if not np.isfinite(parsed):
        return int(default)
    return int(round(parsed))


def build_tracking_table(
    rec: ExiDRecording,
    *,
    options: BuildOptions | None = None,
) -> pd.DataFrame:
    """Build one normalized exiD table for exploratory cut-in mining."""

    options = options or BuildOptions()
    frame_rate = _get_frame_rate(rec.recording_meta)

    tracks = rec.tracks.copy()
    tracks_meta = rec.tracks_meta.copy()
    _assert_unique_key(tracks, keys=("trackId", "frame"), name="tracks")

    tracks["id"] = tracks["trackId"].astype(int)
    tracks["time"] = tracks["frame"] / frame_rate
    tracks["speed"] = np.sqrt(tracks["xVelocity"] ** 2 + tracks["yVelocity"] ** 2)
    tracks["laneId"] = tracks["laneletId"].map(_parse_first_int)
    tracks["precedingId"] = tracks["leadId"].fillna(0).astype(int)
    tracks["followingId"] = tracks["rearId"].fillna(0).astype(int)
    tracks["leftPrecedingId"] = tracks["leftLeadId"].fillna(0).astype(int)
    tracks["leftFollowingId"] = tracks["leftRearId"].fillna(0).astype(int)
    tracks["leftAlongsideId"] = tracks["leftAlongsideId"].map(_parse_first_int)
    tracks["rightPrecedingId"] = tracks["rightLeadId"].fillna(0).astype(int)
    tracks["rightFollowingId"] = tracks["rightRearId"].fillna(0).astype(int)
    tracks["rightAlongsideId"] = tracks["rightAlongsideId"].map(_parse_first_int)
    tracks["laneChange"] = tracks["laneChange"].fillna(0).astype(int)
    tracks["leadDHW"] = pd.to_numeric(tracks["leadDHW"], errors="coerce")
    tracks["leadTHW"] = pd.to_numeric(tracks["leadTHW"], errors="coerce")
    tracks["leadTTC"] = pd.to_numeric(tracks["leadTTC"], errors="coerce")

    meta_cols = [
        "trackId",
        "class",
        "initialFrame",
        "finalFrame",
        "numFrames",
    ]
    merged = tracks.merge(
        tracks_meta[meta_cols],
        on="trackId",
        how="left",
        validate="many_to_one",
    )

    if merged["class"].isna().any():
        missing_ids = merged.loc[merged["class"].isna(), "id"].drop_duplicates().head(20).tolist()
        raise ValueError(
            "tracksMeta join produced missing values for some vehicle ids. "
            f"Examples (up to 20): {missing_ids}"
        )

    if options.keep_only_motor_vehicles:
        merged = merged[merged["class"].isin(MOTOR_VEHICLE_CLASSES)].copy()

    direction_sign = (
        merged.groupby("id", sort=False)["lonVelocity"]
        .transform("median")
        .apply(lambda v: 1 if float(v) >= 0.0 else 2)
    )
    merged["drivingDirection"] = direction_sign.astype(int)

    if not options.keep_optional_tracks_columns:
        keep = [
            "recordingId",
            "frame",
            "id",
            "time",
            "xCenter",
            "yCenter",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "speed",
            "length",
            "width",
            "laneId",
            "laneChange",
            "precedingId",
            "followingId",
            "leftPrecedingId",
            "leftAlongsideId",
            "leftFollowingId",
            "rightPrecedingId",
            "rightAlongsideId",
            "rightFollowingId",
            "leadDHW",
            "leadTHW",
            "leadTTC",
            "class",
            "drivingDirection",
            "initialFrame",
            "finalFrame",
            "numFrames",
        ]
        merged = merged[[c for c in keep if c in merged.columns]].copy()

    merged = merged.sort_values(["id", "frame"]).reset_index(drop=True)
    return merged


def build_lane_change_events(
    df: pd.DataFrame,
    *,
    marker_col: str = "laneChange",
    lane_col: str = "laneId",
    id_col: str = "id",
    frame_col: str = "frame",
    time_col: str = "time",
    merge_gap_frames: int = 2,
) -> list[LaneChangeEvent]:
    """Build lightweight lane-change events from exiD's sparse laneChange markers."""

    required = {marker_col, lane_col, id_col, frame_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_lane_change_events missing required columns: {sorted(missing)}")

    events: list[LaneChangeEvent] = []
    for vehicle_id, group in df.sort_values([id_col, frame_col]).groupby(id_col, sort=False):
        markers = group[group[marker_col].fillna(0).astype(int) > 0].copy()
        if markers.empty:
            continue

        frames = markers[frame_col].astype(int).to_numpy()
        start_idx = 0
        for idx in range(1, len(frames) + 1):
            split = idx == len(frames) or (frames[idx] - frames[idx - 1]) > int(merge_gap_frames)
            if not split:
                continue

            segment = markers.iloc[start_idx:idx]
            start_frame = int(segment[frame_col].iloc[0])
            end_frame = int(segment[frame_col].iloc[-1])
            current_lane = int(segment[lane_col].iloc[0])

            before = group[group[frame_col].astype(int) < start_frame]
            after = group[group[frame_col].astype(int) > end_frame]
            from_lane = int(before[lane_col].iloc[-1]) if not before.empty else current_lane
            to_lane = int(after[lane_col].iloc[0]) if not after.empty else current_lane
            if to_lane == 0:
                to_lane = current_lane
            if from_lane == 0:
                from_lane = current_lane

            events.append(
                LaneChangeEvent(
                    vehicle_id=int(vehicle_id),
                    from_lane=int(from_lane),
                    to_lane=int(to_lane),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=float(segment[time_col].iloc[0]),
                    end_time=float(segment[time_col].iloc[-1]),
                )
            )
            start_idx = idx

    return events
