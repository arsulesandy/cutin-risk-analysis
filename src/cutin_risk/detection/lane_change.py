from __future__ import annotations

"""
Lane-change detection based on discrete lane transitions.

A lane change is defined as a transition from a stable lane block (from_lane)
to a stable lane block (to_lane). Stability is enforced by requiring a minimum
number of consecutive frames in each lane.

The lane column is configurable via LaneChangeOptions.lane_col, allowing this
detector to run on:
  - dataset-provided lane IDs (e.g., "laneId")
  - reconstructed/inferred lane indices (e.g., "laneIndex_xy")
"""

from dataclasses import dataclass

import numpy as np
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

    lane_col:
      Column used to represent lane membership (e.g., "laneId" or "laneIndex_xy").
    """
    min_stable_before_frames: int = 25  # 1 second at 25 Hz
    min_stable_after_frames: int = 25   # 1 second at 25 Hz
    ignore_lane_ids: tuple[int, ...] = (0,)
    lane_col: str = "laneId"

    id_col: str = "id"
    frame_col: str = "frame"
    time_col: str = "time"


def detect_lane_changes(df: pd.DataFrame, *, options: LaneChangeOptions | None = None) -> list[LaneChangeEvent]:
    """
    Detect lane changes based on lane transitions per vehicle.

    Requirements:
      - df contains columns: id, frame, time, <lane_col>
      - each vehicle's rows must be orderable by frame (we sort within each vehicle)
    """
    options = options or LaneChangeOptions()

    required = {options.id_col, options.frame_col, options.time_col, options.lane_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_lane_changes missing required columns: {sorted(missing)}")

    ignore_set = {int(x) for x in options.ignore_lane_ids}
    events: list[LaneChangeEvent] = []

    for vid, g in df.groupby(options.id_col, sort=False):
        # Ensure temporal order even if upstream data isn't strictly sorted.
        g = g.sort_values(options.frame_col, kind="mergesort")

        lane = g[options.lane_col].fillna(0).astype(int).to_numpy()
        frames = g[options.frame_col].to_numpy(dtype=int)
        times = g[options.time_col].to_numpy(dtype=float)

        n = lane.size
        if n < 2:
            continue

        # Identify constant-lane runs (run-length encoding).
        change_mask = lane[1:] != lane[:-1]
        if not np.any(change_mask):
            continue  # vehicle never changes lane

        run_starts = np.concatenate(([0], np.nonzero(change_mask)[0] + 1))
        run_ends = np.concatenate((run_starts[1:] - 1, [n - 1]))
        run_lanes = lane[run_starts]
        run_lengths = run_ends - run_starts + 1

        # Lane-change candidates are boundaries between run k-1 and run k
        for k in range(1, len(run_starts)):
            from_lane = int(run_lanes[k - 1])
            to_lane = int(run_lanes[k])

            if from_lane == to_lane:
                continue
            if from_lane in ignore_set or to_lane in ignore_set:
                continue

            before_len = int(run_lengths[k - 1])
            after_len = int(run_lengths[k])

            if before_len < options.min_stable_before_frames:
                continue
            if after_len < options.min_stable_after_frames:
                continue

            start_idx = int(run_starts[k])  # first frame in target lane
            end_idx = int(run_ends[k])      # last frame in that stable target-lane block

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

    return events
