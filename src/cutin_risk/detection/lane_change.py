from __future__ import annotations

"""
Lane-change detection based on laneId transitions.

This module provides a baseline, reproducible lane-change segmentation strategy:
a lane change is defined as a transition from a stable lane block (from_lane)
to a stable lane block (to_lane).

The implementation intentionally avoids using lateral position (y) in this baseline.
A refined version can be added later for more precise boundary-crossing timing.
"""

from dataclasses import dataclass
import pandas as pd

from .events import LaneChangeEvent


@dataclass(frozen=True)
class LaneChangeOptions:
    """
    min_stable_before_frames:
      Minimum consecutive frames in the source lane required before a transition is accepted.

    min_stable_after_frames:
      Minimum consecutive frames in the target lane required after a transition is accepted.

    ignore_lane_ids:
      Lane ids that are treated as invalid/unknown (transitions involving these are skipped).
    """
    min_stable_before_frames: int = 25  # 1 second at 25 Hz
    min_stable_after_frames: int = 25   # 1 second at 25 Hz
    ignore_lane_ids: tuple[int, ...] = (0,)


def detect_lane_changes(df: pd.DataFrame, *, options: LaneChangeOptions | None = None) -> list[LaneChangeEvent]:
    """
    Detect lane changes based on laneId transitions per vehicle.

    Requirements:
      - df contains columns: id, frame, time, laneId
      - df is sorted by (id, frame)
    """
    options = options or LaneChangeOptions()

    required = {"id", "frame", "time", "laneId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_lane_changes missing required columns: {sorted(missing)}")

    def is_ignored(lane_id: int) -> bool:
        return int(lane_id) in options.ignore_lane_ids

    events: list[LaneChangeEvent] = []

    for vid, g in df.groupby("id", sort=False):
        lane = g["laneId"].to_numpy()
        frames = g["frame"].to_numpy()
        times = g["time"].to_numpy()

        n = len(lane)
        if n < (options.min_stable_before_frames + options.min_stable_after_frames + 2):
            continue

        i = 0
        while i < n - 1:
            cur_lane = int(lane[i])
            nxt_lane = int(lane[i + 1])

            # No change
            if cur_lane == nxt_lane:
                i += 1
                continue

            # Skip transitions involving ignored lane ids
            if is_ignored(cur_lane) or is_ignored(nxt_lane):
                i += 1
                continue

            from_lane = cur_lane
            to_lane = nxt_lane

            # ---- stability BEFORE: count how long we have been in from_lane ending at i ----
            k = i
            while k >= 0 and int(lane[k]) == from_lane:
                k -= 1
            stable_before_len = i - k  # number of from_lane frames ending at i

            if stable_before_len < options.min_stable_before_frames:
                i += 1
                continue

            # Transition starts at first frame in the target lane
            start_idx = i + 1

            # ---- stability AFTER: count how long we remain in to_lane starting at start_idx ----
            j = start_idx
            while j < n and int(lane[j]) == to_lane:
                j += 1
            stable_after_len = j - start_idx

            if stable_after_len >= options.min_stable_after_frames:
                end_idx = j - 1

                events.append(
                    LaneChangeEvent(
                        vehicle_id=int(vid),
                        from_lane=from_lane,
                        to_lane=to_lane,
                        start_frame=int(frames[start_idx]),
                        end_frame=int(frames[end_idx]),
                        start_time=float(times[start_idx]),
                        end_time=float(times[end_idx]),
                    )
                )

            # Continue after the stable block in the target lane
            i = j

    return events
