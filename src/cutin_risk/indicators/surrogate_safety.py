"""
Surrogate safety indicators for leader–follower vehicle pairs.

Indicators implemented:
- DHW: distance headway (gap) [m] - bumper-to-bumper gap in meters
- THW: time headway [s] - gap divided by follower speed (seconds)
- TTC: time-to-collision [s] (classic closing-speed definition) - gap divided by closing speed when follower is faster (seconds)

This module focuses on longitudinal interaction only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


NoNeighbor = tuple[int, ...]
PositionReference = Literal["center", "rear"]


@dataclass(frozen=True)
class LongitudinalModel:
    """
    length_col:
      Column used as longitudinal vehicle length. In highD, "width" typically corresponds
      to vehicle length along the road (e.g., ~4-5m for cars), while "height" corresponds
      to vehicle width.

    position_reference:
      Interpretation of the x position:
        - "center": x is treated as the longitudinal center of the vehicle
        - "rear":   x is treated as the rear bumper position in travel direction

      Use the validation helper to choose what best matches dataset-provided dhw/thw/ttc.
    """
    length_col: str = "width"
    position_reference: PositionReference = "center"


@dataclass(frozen=True)
class IndicatorOptions:
    """
    min_speed:
      Speeds below this are treated as ~0 to avoid division explosions.

    closing_speed_epsilon:
      Minimum positive closing speed required to compute TTC.
    """
    min_speed: float = 0.1
    closing_speed_epsilon: float = 1e-6


def infer_direction_sign_map(df: pd.DataFrame) -> dict[int, int]:
    """
    Infer a sign per drivingDirection so that longitudinal velocity becomes positive
    in the direction of travel.

    v_long = xVelocity * sign_map[drivingDirection]
    """
    if "drivingDirection" not in df.columns or "xVelocity" not in df.columns:
        raise ValueError("infer_direction_sign_map requires drivingDirection and xVelocity columns")

    sign_map: dict[int, int] = {}
    for dd, g in df.groupby("drivingDirection", sort=False):
        dd_i = int(dd)
        med = float(g["xVelocity"].median())
        sign_map[dd_i] = 1 if med >= 0 else -1

    return sign_map


def _longitudinal_state(
        row: pd.Series,
        *,
        sign: int,
        model: LongitudinalModel,
) -> tuple[float, float, float, float]:
    """
    Returns (s_front, s_rear, v_long, length).
    s_* are in the normalized longitudinal axis where forward travel means increasing s.
    """
    x = float(row["x"])
    v = float(row["xVelocity"])
    length = float(row[model.length_col])

    s = sign * x
    v_long = sign * v

    if model.position_reference == "center":
        s_front = s + 0.5 * length
        s_rear = s - 0.5 * length
    elif model.position_reference == "rear":
        # Treat x as rear position along travel direction
        s_rear = s
        s_front = s + length
    else:
        raise ValueError(f"Unknown position_reference: {model.position_reference}")

    return s_front, s_rear, v_long, length


def compute_pair_indicators_at_frame(
        leader_row: pd.Series,
        follower_row: pd.Series,
        *,
        leader_sign: int,
        follower_sign: int,
        model: LongitudinalModel,
        options: IndicatorOptions,
) -> dict[str, float]:
    """
    Compute DHW/THW/TTC for one leader–follower pair at a single frame.

    leader: vehicle in front
    follower: vehicle behind
    """
    leader_front, leader_rear, v_leader, _ = _longitudinal_state(leader_row, sign=leader_sign, model=model)
    follower_front, follower_rear, v_follower, _ = _longitudinal_state(follower_row, sign=follower_sign, model=model)

    # Gap is rear of leader minus front of follower (bumper-to-bumper, under the chosen x reference model)
    dhw = leader_rear - follower_front

    v_f = max(v_follower, 0.0)
    v_l = max(v_leader, 0.0)

    # THW uses follower speed
    if v_f < options.min_speed:
        thw = float("inf")
    else:
        thw = dhw / v_f

    # TTC uses closing speed (follower faster than leader)
    closing_speed = v_f - v_l
    if dhw <= 0.0 or closing_speed <= options.closing_speed_epsilon:
        ttc = float("inf")
    else:
        ttc = dhw / closing_speed

    return {
        "dhw": float(dhw),
        "thw": float(thw),
        "ttc": float(ttc),
        "v_follower": float(v_f),
        "v_leader": float(v_l),
        "closing_speed": float(closing_speed),
        "follower_front_s": float(follower_front),
        "leader_rear_s": float(leader_rear),
    }


def compute_pair_timeseries(
        indexed: pd.DataFrame,
        *,
        leader_id: int,
        follower_id: int,
        frames: range,
        sign_map: dict[int, int],
        model: LongitudinalModel,
        options: IndicatorOptions,
) -> pd.DataFrame:
    """
    Compute indicators for a leader–follower pair over a frame range.

    indexed must be df.set_index(["id","frame"]) with id/frame retained as columns.
    """
    rows = []

    for f in frames:
        try:
            leader_row = indexed.loc[(leader_id, f)]
            follower_row = indexed.loc[(follower_id, f)]
        except KeyError:
            continue

        if isinstance(leader_row, pd.DataFrame):
            leader_row = leader_row.iloc[0]
        if isinstance(follower_row, pd.DataFrame):
            follower_row = follower_row.iloc[0]

        leader_dd = int(leader_row["drivingDirection"])
        follower_dd = int(follower_row["drivingDirection"])

        out = compute_pair_indicators_at_frame(
            leader_row,
            follower_row,
            leader_sign=sign_map.get(leader_dd, 1),
            follower_sign=sign_map.get(follower_dd, 1),
            model=model,
            options=options,
        )

        out.update(
            {
                "frame": int(f),
                "time": float(follower_row["time"]),
                "leader_id": int(leader_id),
                "follower_id": int(follower_id),
            }
        )
        rows.append(out)

    return pd.DataFrame(rows)


def validate_against_dataset_preceding(
        df: pd.DataFrame,
        *,
        sample_n: int = 20000,
        no_neighbor_ids: NoNeighbor = (0, -1),
        model: LongitudinalModel | None = None,
        options: IndicatorOptions | None = None,
        random_state: int = 7,
) -> dict[str, float]:
    """
    Sanity validation:
    Compare our computed dhw/thw/ttc against dataset-provided columns for the (ego, precedingId) pair.

    This does NOT validate cut-ins directly; it validates the indicator computation model.
    """
    model = model or LongitudinalModel()
    options = options or IndicatorOptions()

    required = {"id", "frame", "precedingId", "dhw", "thw", "ttc", "time", "x", "xVelocity", "drivingDirection", model.length_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"validate_against_dataset_preceding missing columns: {sorted(missing)}")

    sign_map = infer_direction_sign_map(df)
    indexed = df.set_index(["id", "frame"], drop=False)

    no_neighbor = set(no_neighbor_ids)

    candidates = df[~df["precedingId"].isin(no_neighbor)].copy()
    if candidates.empty:
        raise ValueError("No rows with a valid precedingId found for validation.")

    sample = candidates.sample(n=min(sample_n, len(candidates)), random_state=random_state)

    dhw_err = []
    thw_err = []
    ttc_err = []

    for _, row in sample.iterrows():
        ego_id = int(row["id"])
        frame = int(row["frame"])
        leader_id = int(row["precedingId"])

        # ego is follower, preceding is leader
        try:
            leader_row = indexed.loc[(leader_id, frame)]
            follower_row = indexed.loc[(ego_id, frame)]
        except KeyError:
            continue

        if isinstance(leader_row, pd.DataFrame):
            leader_row = leader_row.iloc[0]
        if isinstance(follower_row, pd.DataFrame):
            follower_row = follower_row.iloc[0]

        pred = compute_pair_indicators_at_frame(
            leader_row=leader_row,
            follower_row=follower_row,
            leader_sign=sign_map.get(int(leader_row["drivingDirection"]), 1),
            follower_sign=sign_map.get(int(follower_row["drivingDirection"]), 1),
            model=model,
            options=options,
        )

        # Compare only if dataset values are finite-ish
        dhw_err.append(abs(float(row["dhw"]) - pred["dhw"]))

        # thw/ttc might be inf-like depending on dataset conventions; compare only if finite
        row_thw = float(row["thw"])
        if np.isfinite(row_thw) and np.isfinite(pred["thw"]):
            thw_err.append(abs(row_thw - pred["thw"]))

        row_ttc = float(row["ttc"])
        if np.isfinite(row_ttc) and np.isfinite(pred["ttc"]):
            ttc_err.append(abs(row_ttc - pred["ttc"]))

    def _safe_median(xs: list[float]) -> float:
        return float(np.median(xs)) if xs else float("nan")

    def _safe_mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "dhw_mae": _safe_mean(dhw_err),
        "dhw_median_abs_error": _safe_median(dhw_err),
        "thw_mae": _safe_mean(thw_err),
        "thw_median_abs_error": _safe_median(thw_err),
        "ttc_mae": _safe_mean(ttc_err),
        "ttc_median_abs_error": _safe_median(ttc_err),
        "compared_rows": float(len(dhw_err)),
    }
