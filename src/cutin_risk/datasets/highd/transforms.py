from __future__ import annotations

"""
Transform raw highD recording tables into a normalized, analysis-friendly structure.

Responsibilities:
- add continuous time column (seconds) based on recording frame rate
- add basic derived kinematics (e.g., speed magnitude)
- join per-vehicle metadata onto per-frame rows
- enforce basic integrity checks (keys, sorting, duplicates)

Non-responsibilities:
- lane change detection
- cut-in identification
- risk indicator computation
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .reader import HighDRecording


@dataclass(frozen=True)
class BuildOptions:
    """
    Options for building the normalized tracking table.

    keep_optional_tracks_columns:
        If True, keep any optional/highD-version-dependent columns present in tracks.
        If False, keep only core columns used by the pipeline.
    """
    keep_optional_tracks_columns: bool = True


def _get_frame_rate(recording_meta: pd.DataFrame) -> float:
    if recording_meta.empty:
        raise ValueError("recordingMeta is empty")
    if "frameRate" not in recording_meta.columns:
        raise ValueError("recordingMeta does not contain frameRate")
    frame_rate = float(recording_meta.loc[0, "frameRate"])
    if frame_rate <= 0:
        raise ValueError(f"Invalid frameRate: {frame_rate}")
    return frame_rate


def _assert_unique_key(df: pd.DataFrame, keys: Iterable[str], name: str) -> None:
    dup = df.duplicated(list(keys), keep=False)
    if dup.any():
        examples = df.loc[dup, list(keys)].head(10).to_dict(orient="records")
        raise ValueError(f"{name} contains duplicate keys for {list(keys)}. Examples: {examples}")


def build_tracking_table(
        rec: HighDRecording,
        *,
        options: BuildOptions | None = None,
) -> pd.DataFrame:
    """
    Build one normalized table where each row represents one vehicle at one frame.

    Output columns (always present):
      - id, frame, time
      - x, y, xVelocity, yVelocity, xAcceleration, yAcceleration, speed
      - laneId + neighbor ids
      - class, drivingDirection, initialFrame, finalFrame, numFrames (joined from tracksMeta)

    The function is deterministic: sorted by (id, frame), index reset.
    """
    options = options or BuildOptions()

    frame_rate = _get_frame_rate(rec.recording_meta)

    tracks = rec.tracks.copy()
    tracks_meta = rec.tracks_meta.copy()

    # Basic key integrity before doing anything else
    _assert_unique_key(tracks, keys=("id", "frame"), name="tracks")

    # Normalize time and add basic derived kinematics
    tracks["time"] = tracks["frame"] / frame_rate
    tracks["speed"] = np.sqrt(tracks["xVelocity"] ** 2 + tracks["yVelocity"] ** 2)

    # Join selected per-vehicle metadata onto all per-frame rows
    # Keep this conservative; expand only when needed by later steps.
    meta_cols = [
        "id",
        "class",
        "drivingDirection",
        "initialFrame",
        "finalFrame",
        "numFrames",
    ]
    missing_meta_cols = [c for c in meta_cols if c not in tracks_meta.columns]
    if missing_meta_cols:
        raise ValueError(f"tracksMeta missing expected columns: {missing_meta_cols}")

    merged = tracks.merge(
        tracks_meta[meta_cols],
        on="id",
        how="left",
        validate="many_to_one",
    )

    # Join must succeed for all rows; otherwise downstream logic becomes ambiguous.
    if merged["class"].isna().any():
        missing_ids = merged.loc[merged["class"].isna(), "id"].drop_duplicates().head(20).tolist()
        raise ValueError(
            "tracksMeta join produced missing values for some vehicle ids. "
            f"Examples (up to 20): {missing_ids}"
        )

    # Optionally drop columns that are not needed for the pipeline contract.
    # This keeps memory smaller and makes later code easier to reason about.
    if not options.keep_optional_tracks_columns:
        # Keep everything that is part of the core pipeline plus joined meta and derived fields.
        core_cols = [
            "frame",
            "id",
            "time",
            "x",
            "y",
            "width",
            "height",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "speed",
            "laneId",
            "precedingId",
            "followingId",
            "leftPrecedingId",
            "leftAlongsideId",
            "leftFollowingId",
            "rightPrecedingId",
            "rightAlongsideId",
            "rightFollowingId",
            "class",
            "drivingDirection",
            "initialFrame",
            "finalFrame",
            "numFrames",
        ]
        keep = [c for c in core_cols if c in merged.columns]
        merged = merged[keep].copy()

    # Deterministic ordering
    merged = merged.sort_values(["id", "frame"]).reset_index(drop=True)

    # Light-weight sanity: time should be non-decreasing per vehicle
    # (not raising here unless it actually goes backwards; if it does, data is not usable)
    for vid, g in merged.groupby("id", sort=False):
        t = g["time"].to_numpy()
        if (t[1:] < t[:-1]).any():
            raise ValueError(f"Non-monotonic time detected for vehicle id={int(vid)}")

    return merged
