from __future__ import annotations

"""
Neighbor reconstruction from geometry (baseline).

Goal:
- Reconstruct same-lane preceding/following IDs using only geometry + lane grouping.
- This provides a "minimal-input" path where we do not rely on dataset-provided neighbor IDs.

Baseline assumptions:
- Vehicles in the same lane are comparable by a normalized longitudinal coordinate s.
- Within each (frame, drivingDirection, laneId) group:
    preceding  = closest vehicle ahead (larger s)
    following  = closest vehicle behind (smaller s)

Notes:
- This baseline still uses laneId as the lane grouping key.
  Later, we can replace laneId with a lane assignment inferred from y + lane markings.
"""

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from cutin_risk.indicators.surrogate_safety import infer_direction_sign_map


@dataclass(frozen=True)
class NeighborReconstructionOptions:
    id_col: str = "id"
    frame_col: str = "frame"
    lane_col: str = "laneId"
    driving_direction_col: str = "drivingDirection"
    x_col: str = "x"
    x_velocity_col: str = "xVelocity"

    ignore_lane_ids: tuple[int, ...] = (0,)
    no_neighbor_id: int = 0

    out_preceding_col: str = "precedingId_xy"
    out_following_col: str = "followingId_xy"


def reconstruct_same_lane_neighbors(
        df: pd.DataFrame,
        *,
        options: NeighborReconstructionOptions | None = None,
        direction_sign_map: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """
    Reconstruct same-lane preceding/following IDs for every (id, frame) row.

    Returns a DataFrame aligned to df.index with two columns:
      - precedingId_xy
      - followingId_xy
    """
    options = options or NeighborReconstructionOptions()

    direction_sign_map = infer_direction_sign_map(df)

    # Output initialized to "no neighbor" everywhere.
    out = pd.DataFrame(index=df.index)
    out[options.out_preceding_col] = int(options.no_neighbor_id)
    out[options.out_following_col] = int(options.no_neighbor_id)

    # Work only on valid lane rows; invalid lanes stay as no-neighbor.
    valid = df[~df[options.lane_col].isin(set(options.ignore_lane_ids))].copy()
    if valid.empty:
        return out

    # Compute normalized longitudinal coordinate s = sign * x
    sign = valid[options.driving_direction_col].map(lambda dd: direction_sign_map.get(int(dd), 1)).astype(int)
    valid["_s"] = sign * valid[options.x_col].astype(float)

    # Sort so neighbors become adjacent within each (frame, direction, lane) group.
    key_cols = [options.frame_col, options.driving_direction_col, options.lane_col]
    valid = valid.sort_values(key_cols + ["_s", options.id_col], kind="mergesort")

    # Shift IDs within group:
    # - preceding: next in sorted order (ahead)
    # - following: previous in sorted order (behind)
    preceding = valid.groupby(key_cols, sort=False)[options.id_col].shift(-1)
    following = valid.groupby(key_cols, sort=False)[options.id_col].shift(1)

    valid[options.out_preceding_col] = preceding.fillna(options.no_neighbor_id).astype(int)
    valid[options.out_following_col] = following.fillna(options.no_neighbor_id).astype(int)

    # Write back into aligned output using original indices.
    out.loc[valid.index, options.out_preceding_col] = valid[options.out_preceding_col].values
    out.loc[valid.index, options.out_following_col] = valid[options.out_following_col].values

    return out
