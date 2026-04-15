"""exiD dataset schema contracts used by the exploratory adapter."""

from __future__ import annotations

from typing import FrozenSet


TRACKS_SUFFIX = "tracks"
TRACKS_META_SUFFIX = "tracksMeta"
RECORDING_META_SUFFIX = "recordingMeta"


REQUIRED_TRACKS_COLUMNS: FrozenSet[str] = frozenset(
    {
        "recordingId",
        "trackId",
        "frame",
        "xCenter",
        "yCenter",
        "heading",
        "width",
        "length",
        "xVelocity",
        "yVelocity",
        "xAcceleration",
        "yAcceleration",
        "lonVelocity",
        "latVelocity",
        "lonAcceleration",
        "latAcceleration",
        "laneletId",
        "laneChange",
        "leadDHW",
        "leadTHW",
        "leadTTC",
        "leadId",
        "rearId",
        "leftLeadId",
        "leftRearId",
        "leftAlongsideId",
        "rightLeadId",
        "rightRearId",
        "rightAlongsideId",
    }
)

REQUIRED_TRACKS_META_COLUMNS: FrozenSet[str] = frozenset(
    {
        "recordingId",
        "trackId",
        "initialFrame",
        "finalFrame",
        "numFrames",
        "width",
        "length",
        "class",
    }
)

REQUIRED_RECORDING_META_COLUMNS: FrozenSet[str] = frozenset(
    {
        "recordingId",
        "locationId",
        "frameRate",
        "duration",
        "numVehicles",
        "numVRUs",
        "orthoPxToMeter",
    }
)


def require_schema(df, *, name: str, required: FrozenSet[str]) -> None:
    """Raise when a raw exiD CSV is missing required columns."""
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_str}")
