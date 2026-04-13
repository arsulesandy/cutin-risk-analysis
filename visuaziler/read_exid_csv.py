"""CSV parsing utilities used by the interactive exiD visualizer adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas

try:
    from read_csv import (
        BACKGROUND_SCALE,
        BACKGROUND_X_LIMITS,
        BACKGROUND_Y_LIMITS,
        BBOX,
        CLASS,
        DATASET_DISPLAY_NAME,
        DATASET_NAME,
        DHW,
        DRIVING_DIRECTION,
        DURATION,
        FINAL_FRAME,
        FOLLOWING_ID,
        FRAME,
        FRAME_RATE,
        HEIGHT,
        ID,
        INITIAL_FRAME,
        LANE_ID,
        LEFT_ALONGSIDE_ID,
        LEFT_FOLLOWING_ID,
        LEFT_PRECEDING_ID,
        LOCATION_ID,
        MIN_DHW,
        MIN_THW,
        MIN_TTC,
        MIN_X_VELOCITY,
        N_VEHICLES,
        NUM_FRAMES,
        NUMBER_LANE_CHANGES,
        PRECEDING_ID,
        PRECEDING_X_VELOCITY,
        RIGHT_ALONGSIDE_ID,
        RIGHT_FOLLOWING_ID,
        RIGHT_PRECEDING_ID,
        ROAD_INFO_NOTE,
        SPEED_LIMIT,
        THW,
        TRACK_ID,
        TRAVELED_DISTANCE,
        TTC,
        WEEKDAY,
        WIDTH,
        X_ACCELERATION,
        X_VELOCITY,
        Y_ACCELERATION,
        Y_VELOCITY,
    )
except ModuleNotFoundError:
    from visuaziler.read_csv import (
        BACKGROUND_SCALE,
        BACKGROUND_X_LIMITS,
        BACKGROUND_Y_LIMITS,
        BBOX,
        CLASS,
        DATASET_DISPLAY_NAME,
        DATASET_NAME,
        DHW,
        DRIVING_DIRECTION,
        DURATION,
        FINAL_FRAME,
        FOLLOWING_ID,
        FRAME,
        FRAME_RATE,
        HEIGHT,
        ID,
        INITIAL_FRAME,
        LANE_ID,
        LEFT_ALONGSIDE_ID,
        LEFT_FOLLOWING_ID,
        LEFT_PRECEDING_ID,
        LOCATION_ID,
        MIN_DHW,
        MIN_THW,
        MIN_TTC,
        MIN_X_VELOCITY,
        N_VEHICLES,
        NUM_FRAMES,
        NUMBER_LANE_CHANGES,
        PRECEDING_ID,
        PRECEDING_X_VELOCITY,
        RIGHT_ALONGSIDE_ID,
        RIGHT_FOLLOWING_ID,
        RIGHT_PRECEDING_ID,
        ROAD_INFO_NOTE,
        SPEED_LIMIT,
        THW,
        TRACK_ID,
        TRAVELED_DISTANCE,
        TTC,
        WEEKDAY,
        WIDTH,
        X_ACCELERATION,
        X_VELOCITY,
        Y_ACCELERATION,
        Y_VELOCITY,
    )


_VISUALIZER_PARAMS_PATH = (
    Path(__file__).resolve().parents[1]
    / "drone-dataset-tools-master"
    / "data"
    / "visualizer_params"
    / "visualizer_params.json"
)


def _parse_first_numeric(value, *, default=np.nan):
    if pandas.isna(value):
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    token = str(value).strip()
    if not token:
        return default
    head = token.split(";", 1)[0].strip()
    if not head:
        return default
    try:
        return float(head)
    except ValueError:
        return default


def _parse_first_int(value, *, default=0):
    parsed = _parse_first_numeric(value, default=float(default))
    if not np.isfinite(parsed):
        return int(default)
    return int(round(parsed))


def _get_rotated_bbox(x_center, y_center, length, width, heading):
    """Adapted from the official drone-dataset-tools exiD importer."""
    centroids = np.column_stack([x_center, y_center])
    half_length = length / 2.0
    half_width = width / 2.0
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)

    lc = half_length * cos_heading
    ls = half_length * sin_heading
    wc = half_width * cos_heading
    ws = half_width * sin_heading

    corners = np.empty((centroids.shape[0], 4, 2))
    corners[:, 0, 0] = lc - ws
    corners[:, 0, 1] = ls + wc
    corners[:, 1, 0] = -lc - ws
    corners[:, 1, 1] = -ls + wc
    corners[:, 2, 0] = -lc + ws
    corners[:, 2, 1] = -ls - wc
    corners[:, 3, 0] = lc + ws
    corners[:, 3, 1] = ls - wc
    return corners + np.expand_dims(centroids, axis=1)


def _collapse_to_axis_aligned_bbox(rotated_bbox):
    min_xy = rotated_bbox.min(axis=1)
    max_xy = rotated_bbox.max(axis=1)
    return np.column_stack(
        [
            min_xy[:, 0],
            min_xy[:, 1],
            max_xy[:, 0] - min_xy[:, 0],
            max_xy[:, 1] - min_xy[:, 1],
        ]
    )


def _visualizer_crop(location_id):
    if not _VISUALIZER_PARAMS_PATH.exists():
        return None, None
    payload = json.loads(_VISUALIZER_PARAMS_PATH.read_text(encoding="utf-8"))
    exid_block = payload.get("datasets", {}).get("exid", {})
    relevant_area = exid_block.get("relevant_areas", {}).get(str(int(location_id)), {})
    return relevant_area.get("x_lim"), relevant_area.get("y_lim")


def read_track_csv(arguments):
    """Read exiD tracks into the highD-style visualizer contract."""

    meta_df = pandas.read_csv(arguments["input_meta_path"])
    ortho_px_to_meter = float(meta_df["orthoPxToMeter"][0])

    df = pandas.read_csv(arguments["input_path"]).sort_values(["trackId", "frame"])
    grouped = df.groupby(["trackId"], sort=False)
    tracks = [None] * grouped.ngroups

    current_track = 0
    for group_id, rows in grouped:
        x_center_vis = rows["xCenter"].to_numpy(dtype=float) / ortho_px_to_meter
        y_center_vis = -rows["yCenter"].to_numpy(dtype=float) / ortho_px_to_meter
        length_vis = rows["length"].to_numpy(dtype=float) / ortho_px_to_meter
        width_vis = rows["width"].to_numpy(dtype=float) / ortho_px_to_meter
        heading_vis = np.deg2rad((-rows["heading"].to_numpy(dtype=float)) % 360.0)

        rotated_bbox = _get_rotated_bbox(
            x_center_vis,
            y_center_vis,
            length_vis,
            width_vis,
            heading_vis,
        )
        bounding_boxes = _collapse_to_axis_aligned_bbox(rotated_bbox)

        tracks[current_track] = {
            TRACK_ID: np.int64(group_id),
            FRAME: rows["frame"].to_numpy(dtype=int),
            BBOX: bounding_boxes,
            X_VELOCITY: rows["xVelocity"].to_numpy(dtype=float),
            Y_VELOCITY: rows["yVelocity"].to_numpy(dtype=float),
            X_ACCELERATION: rows["xAcceleration"].to_numpy(dtype=float),
            Y_ACCELERATION: rows["yAcceleration"].to_numpy(dtype=float),
            THW: pandas.to_numeric(rows["leadTHW"], errors="coerce").to_numpy(dtype=float),
            TTC: pandas.to_numeric(rows["leadTTC"], errors="coerce").to_numpy(dtype=float),
            DHW: pandas.to_numeric(rows["leadDHW"], errors="coerce").to_numpy(dtype=float),
            PRECEDING_X_VELOCITY: rows["lonVelocity"].to_numpy(dtype=float),
            PRECEDING_ID: rows["leadId"].fillna(0).to_numpy(dtype=int),
            FOLLOWING_ID: rows["rearId"].fillna(0).to_numpy(dtype=int),
            LEFT_FOLLOWING_ID: rows["leftRearId"].fillna(0).to_numpy(dtype=int),
            LEFT_ALONGSIDE_ID: rows["leftAlongsideId"].map(_parse_first_int).to_numpy(dtype=int),
            LEFT_PRECEDING_ID: rows["leftLeadId"].fillna(0).to_numpy(dtype=int),
            RIGHT_FOLLOWING_ID: rows["rightRearId"].fillna(0).to_numpy(dtype=int),
            RIGHT_ALONGSIDE_ID: rows["rightAlongsideId"].map(_parse_first_int).to_numpy(dtype=int),
            RIGHT_PRECEDING_ID: rows["rightLeadId"].fillna(0).to_numpy(dtype=int),
            LANE_ID: rows["laneletId"].map(_parse_first_int).to_numpy(dtype=int),
        }
        current_track += 1
    return tracks


def read_static_info(arguments):
    """Read exiD tracksMeta into the visualizer's static-info contract."""

    df = pandas.read_csv(arguments["input_static_path"]).sort_values("trackId")
    static_dictionary = {}
    for _, row in df.iterrows():
        track_id = int(row["trackId"])
        driving_direction = 1.0
        static_dictionary[track_id] = {
            TRACK_ID: track_id,
            WIDTH: float(row["length"]),
            HEIGHT: float(row["width"]),
            INITIAL_FRAME: int(row["initialFrame"]),
            FINAL_FRAME: int(row["finalFrame"]),
            NUM_FRAMES: int(row["numFrames"]),
            CLASS: str(row["class"]),
            DRIVING_DIRECTION: driving_direction,
            TRAVELED_DISTANCE: float(0.0),
            MIN_X_VELOCITY: float(0.0),
            MIN_TTC: float(np.nan),
            MIN_THW: float(np.nan),
            MIN_DHW: float(np.nan),
            NUMBER_LANE_CHANGES: int(0),
        }
    return static_dictionary


def read_meta_info(arguments):
    """Read exiD recording meta plus viewer crop hints."""

    df = pandas.read_csv(arguments["input_meta_path"])
    row = df.iloc[0]
    x_limits, y_limits = _visualizer_crop(int(row["locationId"]))

    return {
        ID: int(row["recordingId"]),
        FRAME_RATE: int(row["frameRate"]),
        LOCATION_ID: int(row["locationId"]),
        SPEED_LIMIT: float(row["speedLimit"]),
        WEEKDAY: str(row["weekday"]),
        DURATION: float(row["duration"]),
        N_VEHICLES: int(row["numVehicles"]),
        DATASET_NAME: "exid",
        DATASET_DISPLAY_NAME: "exiD",
        BACKGROUND_SCALE: 1.0,
        BACKGROUND_X_LIMITS: np.array(x_limits, dtype=float) if x_limits is not None else None,
        BACKGROUND_Y_LIMITS: np.array(y_limits, dtype=float) if y_limits is not None else None,
        ROAD_INFO_NOTE: "exiD background crop uses location-specific limits from drone-dataset-tools.",
    }
