"""Dataclasses used to exchange lane-change and cut-in events across modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaneChangeEvent:
    """Detected lane-change interval for one vehicle."""
    vehicle_id: int
    from_lane: int
    to_lane: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class CutInEvent:
    """
    A cut-in event derived from a lane change.

    cutter_id:
      The lane-changing vehicle.

    follower_id:
      The vehicle behind the cutter in the target lane (the vehicle that gets cut in front of).

    relation_start_frame/end_frame:
      The contiguous time interval (in frames) where:
        - cutter.followingId == follower_id
        - follower.precedingId == cutter_id
      in the target lane.
    """
    cutter_id: int
    follower_id: int
    from_lane: int
    to_lane: int

    lane_change_start_frame: int
    lane_change_end_frame: int
    lane_change_start_time: float
    lane_change_end_time: float

    relation_start_frame: int
    relation_end_frame: int
    relation_start_time: float
    relation_end_time: float
