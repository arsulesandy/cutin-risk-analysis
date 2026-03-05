"""
Lane inference from lateral position (y) using per-recording lane markings.

This module does NOT use dataset laneId as input. It infers a lane index based on:
- y (optionally y-center)
- drivingDirection (to select which set of markings applies)

Later, if you want to go fully dataset-agnostic, you can replace markings-based lane
assignment with clustering on y.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


YReference = Literal["raw", "center"]


@dataclass(frozen=True)
class LaneMarkings:
    """Lane marking y-positions for both carriageways (two driving directions)."""
    upper: tuple[float, ...]
    lower: tuple[float, ...]


@dataclass(frozen=True)
class LaneInferenceOptions:
    """Column names and behavior toggles for lane inference."""
    y_col: str = "y"
    height_col: str = "height"
    driving_direction_col: str = "drivingDirection"

    upper_markings_col: str = "upperLaneMarkings"
    lower_markings_col: str = "lowerLaneMarkings"

    y_reference: YReference = "center"
    # Optional directional boundary bias (meters) to reduce one-frame lag around lane boundaries.
    # If >0 and velocity column exists, y_ref is shifted by sign(yVelocity) * lane_boundary_eps.
    lane_boundary_eps: float = 0.0
    y_velocity_col: str = "yVelocity"
    y_velocity_deadband: float = 0.05

    out_lane_index_col: str = "laneIndex_xy"
    unknown_lane: int = 0  # 0 means "unassigned / out of bounds"


def _parse_markings(raw: object) -> tuple[float, ...]:
    """
    Parse markings from typical formats:
    - "0;3.5;7.0;10.5"
    - "0, 3.5, 7.0"
    - list/tuple/ndarray of numbers
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ()

    if isinstance(raw, (list, tuple, np.ndarray)):
        vals = [float(x) for x in raw]
        vals = [v for v in vals if np.isfinite(v)]
        return tuple(vals)

    s = str(raw).strip()
    if not s:
        return ()

    # Remove brackets if present
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")

    # Split on ';' or ','
    parts = []
    for token in s.replace(",", ";").split(";"):
        token = token.strip()
        if token:
            parts.append(token)

    vals: list[float] = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)

    return tuple(vals)


def parse_lane_markings(recording_meta: pd.DataFrame, *, options: LaneInferenceOptions | None = None) -> LaneMarkings:
    """
    Read lane markings from recording metadata.
    Expects one-row DataFrame; uses first row.
    """
    options = options or LaneInferenceOptions()

    if recording_meta is None or recording_meta.empty:
        raise ValueError("recording_meta is empty; cannot parse lane markings.")

    row = recording_meta.iloc[0]
    upper = _parse_markings(row.get(options.upper_markings_col))
    lower = _parse_markings(row.get(options.lower_markings_col))

    if len(upper) < 2 and len(lower) < 2:
        raise ValueError("Could not parse lane markings (need at least 2 positions per side).")

    return LaneMarkings(upper=upper, lower=lower)


def _ensure_ascending(boundaries: np.ndarray) -> np.ndarray:
    """Return finite boundaries in ascending order."""
    b = boundaries.astype(float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return b
    if b[0] > b[-1]:
        b = b[::-1]
    return b


def _interval_index(y: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """
    Given boundaries [b0,b1,...,bk], return interval index i where:
      b[i] <= y < b[i+1]
    Invalid/outside => -1
    """
    if boundaries.size < 2:
        return np.full_like(y, -1, dtype=int)

    idx = np.searchsorted(boundaries, y, side="right") - 1
    valid = (idx >= 0) & (idx < (len(boundaries) - 1))
    return np.where(valid, idx, -1).astype(int)


def infer_lane_index(
        df: pd.DataFrame,
        markings: LaneMarkings,
        *,
        options: LaneInferenceOptions | None = None,
) -> pd.Series:
    """
    Infer lane index per row (1..N within a driving direction). 0 means unknown/outside.
    """
    options = options or LaneInferenceOptions()

    required = {options.y_col, options.driving_direction_col}
    if options.y_reference == "center":
        required.add(options.height_col)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"infer_lane_index missing columns: {sorted(missing)}")

    if options.y_reference == "center":
        y_ref = df[options.y_col].astype(float) + 0.5 * df[options.height_col].astype(float)
    else:
        y_ref = df[options.y_col].astype(float)

    # Optional directional boundary bias to reduce transition lag exactly at lane borders.
    if float(options.lane_boundary_eps) > 0.0 and options.y_velocity_col in df.columns:
        vy = pd.to_numeric(df[options.y_velocity_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sign_vy = np.sign(vy)
        sign_vy[np.abs(vy) < float(options.y_velocity_deadband)] = 0.0
        y_ref = y_ref + (float(options.lane_boundary_eps) * sign_vy)

    dd = df[options.driving_direction_col].astype(int)

    # Prepare boundaries
    upper_b = _ensure_ascending(np.array(markings.upper, dtype=float))
    lower_b = _ensure_ascending(np.array(markings.lower, dtype=float))

    # Decide which marking set belongs to which drivingDirection (robustly)
    # We do this by checking which boundary range contains the direction's median y.
    dd_values = sorted(dd.unique().tolist())
    dd_to_side: dict[int, str] = {}

    upper_min, upper_max = (float(np.min(upper_b)), float(np.max(upper_b))) if upper_b.size else (np.nan, np.nan)
    lower_min, lower_max = (float(np.min(lower_b)), float(np.max(lower_b))) if lower_b.size else (np.nan, np.nan)

    for d in dd_values:
        med = float(np.median(y_ref[dd == d]))
        in_upper = (upper_b.size >= 2) and (upper_min <= med <= upper_max)
        in_lower = (lower_b.size >= 2) and (lower_min <= med <= lower_max)

        if in_upper and not in_lower:
            dd_to_side[d] = "upper"
        elif in_lower and not in_upper:
            dd_to_side[d] = "lower"
        else:
            # Fallback: choose closest range center
            upper_center = 0.5 * (upper_min + upper_max) if np.isfinite(upper_min) else np.inf
            lower_center = 0.5 * (lower_min + lower_max) if np.isfinite(lower_min) else np.inf
            dd_to_side[d] = "upper" if abs(med - upper_center) <= abs(med - lower_center) else "lower"

    lane_index = np.full(len(df), options.unknown_lane, dtype=int)

    for d in dd_values:
        mask = (dd == d).to_numpy()
        boundaries = upper_b if dd_to_side[d] == "upper" else lower_b
        idx = _interval_index(y_ref.to_numpy(dtype=float)[mask], boundaries)

        # Convert interval index 0..N-1 to lane index 1..N
        lane_index[mask] = np.where(idx >= 0, idx + 1, options.unknown_lane)

    return pd.Series(lane_index, index=df.index, name=options.out_lane_index_col)
