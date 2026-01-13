from __future__ import annotations

"""
highD dataset schema and validation helpers.

This module defines:
- file naming conventions for a recording
- required columns (minimal contract used by the pipeline)
- optional columns (dataset-version/extractor dependent)
- small helpers to validate and report schema compatibility

The loader should validate the *minimal* contract strictly and treat optional columns
as best-effort inputs.
"""

from dataclasses import dataclass
from typing import FrozenSet, Iterable

import pandas as pd


# -----------------------------
# Dataset conventions
# -----------------------------

TRACKS_SUFFIX = "tracks"
TRACKS_META_SUFFIX = "tracksMeta"
RECORDING_META_SUFFIX = "recordingMeta"


# -----------------------------
# Column contracts
# -----------------------------

# Minimal contract: core kinematics + lane assignment + neighbor relations.
# This is the smallest stable set needed to reproduce detection and indicators.
REQUIRED_TRACKS_COLUMNS: FrozenSet[str] = frozenset(
    {
        "frame",
        "id",
        "x",
        "y",
        "width",
        "height",
        "xVelocity",
        "yVelocity",
        "xAcceleration",
        "yAcceleration",
        "laneId",
        "precedingId",
        "followingId",
        "leftPrecedingId",
        "leftAlongsideId",
        "leftFollowingId",
        "rightPrecedingId",
        "rightAlongsideId",
        "rightFollowingId",
    }
)

# Optional columns that may exist depending on highD export/version.
# These can be useful for cross-checking, debugging, or alternative analyses,
# but the pipeline does not rely on them.
OPTIONAL_TRACKS_COLUMNS: FrozenSet[str] = frozenset(
    {
        "frontSightDistance",
        "backSightDistance",
        "dhw",
        "thw",
        "ttc",
        "precedingXVelocity",
    }
)

REQUIRED_TRACKS_META_COLUMNS: FrozenSet[str] = frozenset(
    {
        "id",
        "class",
        "drivingDirection",
        "initialFrame",
        "finalFrame",
        "numFrames",
    }
)

# Keep this contract small and stable.
# Additional fields can be treated as optional when needed.
REQUIRED_RECORDING_META_COLUMNS: FrozenSet[str] = frozenset(
    {
        "id",
        "frameRate",
        "locationId",
        "duration",
    }
)


# -----------------------------
# Validation results
# -----------------------------

@dataclass(frozen=True)
class SchemaReport:
    """
    A lightweight compatibility report for a DataFrame.
    """
    name: str
    required_missing: tuple[str, ...]
    optional_present: tuple[str, ...]
    unexpected_columns_count: int

    @property
    def ok(self) -> bool:
        return len(self.required_missing) == 0


class SchemaError(ValueError):
    """
    Raised when a required schema contract is not satisfied.
    """
    pass


# -----------------------------
# Helpers
# -----------------------------

def _sorted_tuple(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(items)))


def build_schema_report(
        df: pd.DataFrame,
        *,
        name: str,
        required: FrozenSet[str],
        optional: FrozenSet[str] | None = None,
) -> SchemaReport:
    """
    Build a schema report without raising.

    - required_missing: required columns absent in df
    - optional_present: optional columns that exist in df
    - unexpected_columns_count: columns in df that are neither required nor optional
      (informational only; many dataset versions include extra fields)
    """
    cols = set(df.columns)
    optional = optional or frozenset()

    required_missing = _sorted_tuple(required - cols)
    optional_present = _sorted_tuple(cols & optional)

    known = set(required) | set(optional)
    unexpected = cols - known

    return SchemaReport(
        name=name,
        required_missing=required_missing,
        optional_present=optional_present,
        unexpected_columns_count=len(unexpected),
    )


def require_schema(
        df: pd.DataFrame,
        *,
        name: str,
        required: FrozenSet[str],
) -> None:
    """
    Enforce the required schema contract.
    """
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise SchemaError(f"{name} is missing required columns: {missing_str}")


def list_supported_optional_tracks_columns(df: pd.DataFrame) -> list[str]:
    """
    Convenience helper: returns optional tracks columns that exist in df.
    """
    return sorted(set(df.columns) & set(OPTIONAL_TRACKS_COLUMNS))
