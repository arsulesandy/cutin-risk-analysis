"""
Cut-in detection based on previously detected lane-change events.

Definition:
A cut-in occurs when a vehicle (the cutter) changes into a target lane and,
shortly after the lane change begins, becomes the preceding vehicle of a
follower in that lane. The follower relationship must:

1. Appear within a limited time window after the lane-change start.
2. Persist for a minimum number of consecutive frames.
3. Be mutually consistent (cutter.following == follower AND
   follower.preceding == cutter).
4. Represent a new follower relationship compared to the period before
   the lane change (optional but enabled by default).

The implementation is dataset-agnostic and supports configurable column names.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .events import LaneChangeEvent, CutInEvent


@dataclass(frozen=True)
class CutInOptions:
    """
    search_window_frames:
        Maximum number of frames after the lane-change start in which a
        follower relationship may be detected.

    start_offset_frames:
        Offset (in frames) added to the lane-change start before scanning begins.

    max_relation_delay_frames:
        The follower relation must begin within this many frames after
        scan_start. This constrains how quickly the cut-in must manifest.

    min_relation_frames:
        Minimum number of consecutive frames that the cutter-follower
        relationship must persist to be accepted.

    require_new_follower:
        If True, the follower detected after the lane change must not have
        been the same follower during the pre-change window.

    precheck_frames:
        Number of frames before the lane-change start used to collect
        prior follower IDs (for novelty check).

    no_neighbor_ids:
        Sentinel values representing "no neighbor".

    require_lane_match:
        If True, both vehicles must be in the target lane.

    require_preceding_consistency:
        If True, follower.preceding must equal cutter_id.

    lane_col, following_col, preceding_col:
        Column names describing lane membership and neighbor relations.

    id_col, frame_col, time_col:
        Base column names.
    """

    search_window_frames: int = 50
    start_offset_frames: int = 0
    max_relation_delay_frames: int = 15
    min_relation_frames: int = 10

    require_new_follower: bool = True
    precheck_frames: int = 25

    no_neighbor_ids: tuple[int, ...] = (0, -1)

    require_lane_match: bool = True
    require_preceding_consistency: bool = True

    lane_col: str = "laneId"
    following_col: str = "followingId"
    preceding_col: str = "precedingId"

    id_col: str = "id"
    frame_col: str = "frame"
    time_col: str = "time"


def _as_int(value, *, default: int = 0) -> int:
    """Convert value to int, handling NaN safely."""
    if pd.isna(value):
        return default
    return int(value)


def _get_row(indexed: pd.DataFrame, vehicle_id: int, frame: int) -> pd.Series | None:
    """
    Retrieve a single row using (vehicle_id, frame) from a MultiIndex DataFrame.
    Returns None if the row does not exist.
    """
    try:
        row = indexed.loc[(vehicle_id, frame)]
    except KeyError:
        return None

    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def _get_time(indexed: pd.DataFrame, vehicle_id: int, frame: int, time_col: str) -> float:
    """Safely extract timestamp for a given vehicle/frame."""
    row = _get_row(indexed, vehicle_id, frame)
    if row is None:
        return float("nan")
    return float(row[time_col])


def detect_cutins(
        df: pd.DataFrame,
        lane_changes: list[LaneChangeEvent],
        *,
        options: CutInOptions | None = None,
) -> list[CutInEvent]:
    """
    Detect cut-in events from a list of lane-change events.

    Parameters
    ----------
    df:
        Trajectory dataframe containing lane and neighbor information.

    lane_changes:
        List of LaneChangeEvent objects previously detected.

    options:
        Configuration parameters for detection logic.

    Returns
    -------
    List[CutInEvent]
        All detected cut-in events.
    """
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

    # Build MultiIndex for efficient (vehicle_id, frame) lookups
    indexed = df.set_index([options.id_col, options.frame_col], drop=False)

    no_neighbor = {int(x) for x in options.no_neighbor_ids}
    global_max_frame = int(df[options.frame_col].max())

    events: list[CutInEvent] = []

    for lc in lane_changes:
        cutter_id = int(lc.vehicle_id)
        from_lane = int(lc.from_lane)
        to_lane = int(lc.to_lane)
        lc_start = int(lc.start_frame)

        scan_start = lc_start + int(options.start_offset_frames)
        scan_end = min(scan_start + options.search_window_frames - 1, global_max_frame)

        if scan_end < scan_start:
            continue

        # Collect prior follower IDs before lane change (for novelty check)
        old_followers: set[int] = set()
        if options.require_new_follower and options.precheck_frames > 0:
            pre_start = max(0, lc_start - options.precheck_frames)
            for fpre in range(pre_start, lc_start):
                row_pre = _get_row(indexed, cutter_id, fpre)
                if row_pre is None:
                    continue
                if _as_int(row_pre[options.lane_col]) != from_lane:
                    continue
                fid = _as_int(row_pre[options.following_col])
                if fid not in no_neighbor and fid != cutter_id:
                    old_followers.add(fid)

        # Restrict how late the relation may begin
        establish_deadline = min(
            scan_end,
            scan_start + options.max_relation_delay_frames
        )

        f = scan_start
        found_event: CutInEvent | None = None

        while f <= establish_deadline:
            cutter_row = _get_row(indexed, cutter_id, f)
            if cutter_row is None:
                f += 1
                continue

            if _as_int(cutter_row[options.lane_col]) != to_lane:
                f += 1
                continue

            follower_id = _as_int(cutter_row[options.following_col])
            if follower_id in no_neighbor or follower_id == cutter_id:
                f += 1
                continue

            if options.require_new_follower and follower_id in old_followers:
                f += 1
                continue

            follower_row = _get_row(indexed, follower_id, f)
            if follower_row is None:
                f += 1
                continue

            if options.require_lane_match and \
                    _as_int(follower_row[options.lane_col]) != to_lane:
                f += 1
                continue

            if options.require_preceding_consistency and \
                    _as_int(follower_row[options.preceding_col]) != cutter_id:
                f += 1
                continue

            # Extend relation forward while it remains consistent
            start_frame = f
            end_frame = f

            while end_frame + 1 <= scan_end:
                fn = end_frame + 1

                c_next = _get_row(indexed, cutter_id, fn)
                if c_next is None:
                    break
                if _as_int(c_next[options.lane_col]) != to_lane:
                    break
                if _as_int(c_next[options.following_col]) != follower_id:
                    break

                fol_next = _get_row(indexed, follower_id, fn)
                if fol_next is None:
                    break
                if options.require_lane_match and \
                        _as_int(fol_next[options.lane_col]) != to_lane:
                    break
                if options.require_preceding_consistency and \
                        _as_int(fol_next[options.preceding_col]) != cutter_id:
                    break

                end_frame = fn

            duration = end_frame - start_frame + 1

            if duration >= options.min_relation_frames:
                found_event = CutInEvent(
                    cutter_id=cutter_id,
                    follower_id=follower_id,
                    from_lane=from_lane,
                    to_lane=to_lane,
                    lane_change_start_frame=int(lc.start_frame),
                    lane_change_end_frame=int(lc.end_frame),
                    lane_change_start_time=float(lc.start_time),
                    lane_change_end_time=float(lc.end_time),
                    relation_start_frame=start_frame,
                    relation_end_frame=end_frame,
                    relation_start_time=_get_time(indexed, cutter_id, start_frame, options.time_col),
                    relation_end_time=_get_time(indexed, cutter_id, end_frame, options.time_col),
                )
                break

            f = end_frame + 1

        if found_event is not None:
            events.append(found_event)

    return events
