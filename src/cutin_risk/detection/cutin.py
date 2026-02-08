from __future__ import annotations

"""
Cut-in identification built on top of lane-change events.

Definition used here (baseline):
- Start from a LaneChangeEvent into a target lane.
- Within a short window after the lane-change start, identify a follower vehicle such that:
    cutter.following_col == follower_id
    follower.preceding_col == cutter_id
- Require the relationship to persist for at least N consecutive frames.

This module supports different lane / neighbor columns via CutInOptions:
- lane_col can be "laneId" (highD) or "laneIndex_xy" (inferred lanes)
- following_col / preceding_col can be highD or reconstructed columns
"""

from dataclasses import dataclass

import pandas as pd

from .events import LaneChangeEvent, CutInEvent


@dataclass(frozen=True)
class CutInOptions:
    # Window in which we try to find the follower relation (frames after lane-change start)
    search_window_frames: int = 50
    start_offset_frames: int = 0

    # Relation must hold for this many consecutive frames
    min_relation_frames: int = 10

    # Sentinel values meaning "no neighbor"
    no_neighbor_ids: tuple[int, ...] = (0, -1)

    # Consistency requirements
    require_lane_match: bool = True
    require_preceding_consistency: bool = True

    # Pluggable columns
    lane_col: str = "laneId"
    following_col: str = "followingId"
    preceding_col: str = "precedingId"

    # Base columns
    id_col: str = "id"
    frame_col: str = "frame"
    time_col: str = "time"


def _as_int(value, *, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def _get_row(indexed: pd.DataFrame, id_col: str, frame_col: str, vehicle_id: int, frame: int) -> pd.Series | None:
    try:
        r = indexed.loc[(vehicle_id, frame)]
    except KeyError:
        return None
    if isinstance(r, pd.DataFrame):
        return r.iloc[0]
    return r


def detect_cutins(
        df: pd.DataFrame,
        lane_changes: list[LaneChangeEvent],
        *,
        options: CutInOptions | None = None,
) -> list[CutInEvent]:
    options = options or CutInOptions()

    required = {
        options.id_col,
        options.frame_col,
        options.time_col,
        options.lane_col,
        options.following_col,
        options.preceding_col,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_cutins missing required columns: {sorted(missing)}")

    # MultiIndex for fast (id, frame) lookup
    indexed = df.set_index([options.id_col, options.frame_col], drop=False)

    no_neighbor = set(int(x) for x in options.no_neighbor_ids)
    global_max_frame = int(df[options.frame_col].max())

    events: list[CutInEvent] = []

    for lc in lane_changes:
        cutter_id = int(lc.vehicle_id)
        to_lane = int(lc.to_lane)

        scan_start = int(lc.start_frame) + int(options.start_offset_frames)
        scan_end = min(scan_start + int(options.search_window_frames) - 1, global_max_frame)
        if scan_end < scan_start:
            continue

        found: CutInEvent | None = None
        f = scan_start

        while f <= scan_end:
            cutter_row = _get_row(indexed, options.id_col, options.frame_col, cutter_id, f)
            if cutter_row is None:
                f += 1
                continue

            # IMPORTANT: use lane_col (not hardcoded "laneId")
            if _as_int(cutter_row[options.lane_col]) != to_lane:
                f += 1
                continue

            follower_id = _as_int(cutter_row[options.following_col])
            if follower_id in no_neighbor or follower_id == cutter_id:
                f += 1
                continue

            follower_row = _get_row(indexed, options.id_col, options.frame_col, follower_id, f)
            if follower_row is None:
                f += 1
                continue

            if options.require_lane_match and _as_int(follower_row[options.lane_col]) != to_lane:
                f += 1
                continue

            if options.require_preceding_consistency and _as_int(follower_row[options.preceding_col]) != cutter_id:
                f += 1
                continue

            # Extend relation forward while it remains consistent
            start_frame = f
            end_frame = f

            while end_frame + 1 <= scan_end:
                fn = end_frame + 1

                c_next = _get_row(indexed, options.id_col, options.frame_col, cutter_id, fn)
                if c_next is None:
                    break
                if _as_int(c_next[options.lane_col]) != to_lane:
                    break
                if _as_int(c_next[options.following_col]) != follower_id:
                    break

                fol_next = _get_row(indexed, options.id_col, options.frame_col, follower_id, fn)
                if fol_next is None:
                    break
                if options.require_lane_match and _as_int(fol_next[options.lane_col]) != to_lane:
                    break
                if options.require_preceding_consistency and _as_int(fol_next[options.preceding_col]) != cutter_id:
                    break

                end_frame = fn

            duration = end_frame - start_frame + 1

            if duration >= options.min_relation_frames:
                start_time = float(_get_row(indexed, options.id_col, options.frame_col, cutter_id, start_frame)[options.time_col])
                end_time = float(_get_row(indexed, options.id_col, options.frame_col, cutter_id, end_frame)[options.time_col])

                found = CutInEvent(
                    cutter_id=cutter_id,
                    follower_id=int(follower_id),
                    from_lane=int(lc.from_lane),
                    to_lane=to_lane,
                    lane_change_start_frame=int(lc.start_frame),
                    lane_change_end_frame=int(lc.end_frame),
                    lane_change_start_time=float(lc.start_time),
                    lane_change_end_time=float(lc.end_time),
                    relation_start_frame=int(start_frame),
                    relation_end_frame=int(end_frame),
                    relation_start_time=float(start_time),
                    relation_end_time=float(end_time),
                )
                break

            # If relation was too short, skip past it and keep scanning
            f = end_frame + 1

        if found is not None:
            events.append(found)

    return events
