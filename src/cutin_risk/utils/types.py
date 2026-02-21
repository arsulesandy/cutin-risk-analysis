"""Shared dataclasses representing trajectory and scenario primitives."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """Simple trajectory point container."""
    track_id: int
    time_s: float
    x: float
    y: float


@dataclass
class Scenario:
    """Minimal scenario descriptor used in prototype APIs."""
    track_id: int
    description: str
