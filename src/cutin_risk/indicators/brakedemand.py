"""
Deceleration-based surrogate safety measures for leader-follower pairs.

Indicators:
- DRAC: Deceleration Rate to Avoid Crash [m/s²]
  DRAC = (v_follower - v_leader)² / (2 * DHW)  when follower is closing
  Undefined (inf) when not closing or DHW <= 0.

- Required Deceleration (RD): minimum deceleration follower must apply
  to avoid collision given current gap and closing speed.
  Equivalent to DRAC under constant-speed assumption.

References:
  - Archer (2005), "Indicators for traffic safety assessment and
    prediction and their application in micro-simulation modelling"
  - Wang et al. (2021), "A review of surrogate safety measures and
    their applications in connected and automated vehicles safety modeling"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DRACOptions:
    """Configuration for DRAC computation."""
    min_dhw: float = 0.01  # Avoid division by zero
    closing_speed_epsilon: float = 1e-6  # Min closing speed to compute DRAC


def compute_drac_series(
    dhw: pd.Series | np.ndarray,
    v_follower: pd.Series | np.ndarray,
    v_leader: pd.Series | np.ndarray,
    opts: DRACOptions | None = None,
) -> pd.Series:
    """
    Compute frame-wise DRAC for a leader-follower pair.

    DRAC = (v_follower - v_leader)² / (2 * DHW)

    Only defined when follower is closing (v_follower > v_leader) and DHW > 0.
    Returns np.inf when not closing or gap is non-positive.

    Parameters
    ----------
    dhw : array-like
        Distance headway (bumper-to-bumper gap) in meters.
    v_follower : array-like
        Follower longitudinal speed in m/s.
    v_leader : array-like
        Leader longitudinal speed in m/s.
    opts : DRACOptions, optional
        Configuration options.

    Returns
    -------
    pd.Series
        DRAC values in m/s². np.inf where DRAC is undefined.
    """
    if opts is None:
        opts = DRACOptions()

    dhw = np.asarray(dhw, dtype=float)
    v_f = np.asarray(v_follower, dtype=float)
    v_l = np.asarray(v_leader, dtype=float)

    closing_speed = v_f - v_l
    drac = np.full_like(dhw, np.inf)

    valid = (closing_speed > opts.closing_speed_epsilon) & (dhw > opts.min_dhw)
    drac[valid] = (closing_speed[valid] ** 2) / (2.0 * dhw[valid])

    return pd.Series(drac, name="drac")


def compute_drac_for_stage_features(
    stage_df: pd.DataFrame,
    dhw_col: str = "dhw",
    v_follower_col: str = "follower_speed",
    v_leader_col: str = "cutter_speed",
) -> pd.DataFrame:
    """
    Add DRAC column to a DataFrame with per-frame pair data.

    Returns the original DataFrame with an added 'drac' column.
    """
    drac = compute_drac_series(
        stage_df[dhw_col], stage_df[v_follower_col], stage_df[v_leader_col]
    )
    return stage_df.assign(drac=drac)


def summarize_drac_per_event(
    drac_series: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
) -> dict:
    """
    Summarize DRAC for one event's stage window.

    Parameters
    ----------
    drac_series : array-like
        DRAC values per frame (may include np.inf for non-closing frames).
    thresholds : list of float, optional
        DRAC thresholds to check (default: [3.35, 3.0, 2.0]).
        3.35 m/s² is commonly used in literature as "hard braking" threshold.

    Returns
    -------
    dict with keys:
        drac_max_finite: Maximum finite DRAC value
        drac_mean_finite: Mean of finite DRAC values
        drac_finite_fraction: Fraction of frames with finite DRAC
        drac_exceeds_{threshold}: Whether max DRAC exceeds each threshold
    """
    if thresholds is None:
        thresholds = [3.35, 3.0, 2.0]

    arr = np.asarray(drac_series, dtype=float)
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]

    result = {
        "drac_max_finite": float(finite_vals.max()) if len(finite_vals) > 0 else np.nan,
        "drac_mean_finite": float(finite_vals.mean()) if len(finite_vals) > 0 else np.nan,
        "drac_finite_fraction": float(finite_mask.mean()),
    }

    for thr in thresholds:
        key = f"drac_exceeds_{thr:.2f}"
        result[key] = bool(result["drac_max_finite"] > thr) if not np.isnan(result["drac_max_finite"]) else False

    return result
