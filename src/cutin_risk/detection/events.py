from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class LaneChangeEvent:
    vehicle_id: int
    from_lane: int
    to_lane: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
