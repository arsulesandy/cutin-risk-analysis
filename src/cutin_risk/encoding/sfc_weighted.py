"""Weighted SFC features for local traffic context around cut-in events.

Compared with binary SFC encoding, this module assigns each occupied cell a score
based on distance or TTC-like urgency before vectorizing in SFC order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import numpy as np

from cutin_risk.encoding.sfc_binary import sfc_index_4x4, SFCOrder


ValueMode = Literal["distance", "ttc"]


@dataclass(frozen=True)
class WeightedSFCOptions:
    """Configuration for weighted 3x3 grid extraction and score shaping."""
    lane_col: str = "laneIndex_xy"
    sign_col: str = "sign"
    s_col: str = "s"

    alongside_s_thresh: float = 5.0
    max_range_ahead: Optional[float] = 150.0
    max_range_behind: Optional[float] = 150.0

    # TTC score shaping
    ttc_max: float = 10.0  # seconds

    include_center: bool = True  # keep ego cell present as a constant anchor
    order: SFCOrder = "hilbert"
    mode: ValueMode = "distance"


def _lane_key(frame: int, lane: int, sign: int) -> tuple[int, int, int]:
    """Canonical dictionary key for a weighted-grid snapshot lookup."""
    return (int(frame), int(lane), int(sign))


def _nearest_ahead(s_sorted: np.ndarray, ids_sorted: np.ndarray, s0: float) -> tuple[int, float]:
    """Return nearest vehicle ahead and its forward distance from ego position."""
    pos = int(np.searchsorted(s_sorted, s0, side="right"))
    if pos >= len(s_sorted):
        return 0, float("inf")
    return int(ids_sorted[pos]), float(s_sorted[pos] - s0)


def _nearest_behind(s_sorted: np.ndarray, ids_sorted: np.ndarray, s0: float) -> tuple[int, float]:
    """Return nearest vehicle behind and its backward distance from ego position."""
    pos = int(np.searchsorted(s_sorted, s0, side="left"))
    idx = pos - 1
    if idx < 0:
        return 0, float("inf")
    return int(ids_sorted[idx]), float(s0 - s_sorted[idx])


def _nearest_alongside(s_sorted: np.ndarray, ids_sorted: np.ndarray, s0: float, *, thresh: float) -> tuple[int, float]:
    """Return nearest alongside candidate inside a symmetric longitudinal threshold."""
    pos = int(np.searchsorted(s_sorted, s0, side="left"))
    best_id = 0
    best_abs = float("inf")

    for idx in (pos - 1, pos, pos + 1):
        if 0 <= idx < len(s_sorted):
            ds = float(s_sorted[idx] - s0)
            ads = abs(ds)
            if ads <= thresh and ads < best_abs:
                best_abs = ads
                best_id = int(ids_sorted[idx])

    return best_id, float(best_abs) if best_id != 0 else float("inf")


def _distance_score(ds: float, rng: Optional[float]) -> float:
    """Map distance to [0, 1] with linear gate or exponential fallback."""
    if not np.isfinite(ds):
        return 0.0
    if rng is None:
        # fallback: exponential decay
        return float(np.exp(-ds / 50.0))
    if rng <= 0:
        return 0.0
    return float(max(0.0, 1.0 - (ds / float(rng))))


def _ttc_score(ds: float, v_rel: float, *, ttc_max: float) -> float:
    """Map TTC proxy to [0, 1], where lower TTC yields higher risk score."""
    if not (np.isfinite(ds) and np.isfinite(v_rel)):
        return 0.0
    if v_rel <= 0:
        return 0.0
    ttc = ds / v_rel
    if not np.isfinite(ttc):
        return 0.0
    ttc = min(float(ttc), float(ttc_max))
    return float(max(0.0, 1.0 - (ttc / float(ttc_max))))


def grid3x3_weighted(
        *,
        snapshots: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]],
        indexed,  # pandas df indexed by (id, frame)
        frame: int,
        cutter_id: int,
        cutter_lane: int,
        cutter_sign: int,
        cutter_s: float,
        cutter_v_s: float,
        options: WeightedSFCOptions,
) -> np.ndarray:
    """
    3x3 weighted grid:
      rows: preceding / alongside / following
      cols: left / same / right
    Values are in [0,1] as scores (distance or TTC).
    """
    g = np.zeros((3, 3), dtype=float)
    if options.include_center:
        g[1, 1] = 1.0

    lane_offsets = (-1, 0, +1)
    for col, d_lane in enumerate(lane_offsets):
        lane = int(cutter_lane + d_lane)
        key = _lane_key(frame, lane, cutter_sign)
        if key not in snapshots:
            continue

        s_sorted, ids_sorted = snapshots[key]

        pid, ds_a = _nearest_ahead(s_sorted, ids_sorted, cutter_s)
        fid, ds_b = _nearest_behind(s_sorted, ids_sorted, cutter_s)

        # range gates
        if options.max_range_ahead is not None and pid != 0 and ds_a > float(options.max_range_ahead):
            pid, ds_a = 0, float("inf")
        if options.max_range_behind is not None and fid != 0 and ds_b > float(options.max_range_behind):
            fid, ds_b = 0, float("inf")

        # preceding
        if pid != 0:
            if options.mode == "distance":
                g[0, col] = _distance_score(ds_a, options.max_range_ahead)
            else:
                try:
                    prow = indexed.loc[(pid, frame)]
                    v_p = float(prow["sign"]) * float(prow["xVelocity"])
                    v_rel = cutter_v_s - v_p
                    g[0, col] = _ttc_score(ds_a, v_rel, ttc_max=options.ttc_max)
                except Exception:
                    g[0, col] = 0.0

        # following (risk from behind: follower approaching cutter)
        if fid != 0:
            if options.mode == "distance":
                g[2, col] = _distance_score(ds_b, options.max_range_behind)
            else:
                try:
                    frow = indexed.loc[(fid, frame)]
                    v_f = float(frow["sign"]) * float(frow["xVelocity"])
                    v_rel = v_f - cutter_v_s
                    g[2, col] = _ttc_score(ds_b, v_rel, ttc_max=options.ttc_max)
                except Exception:
                    g[2, col] = 0.0

        # alongside only in adjacent lanes (still use distance score even in TTC mode)
        if d_lane != 0:
            aid, ds_side = _nearest_alongside(
                s_sorted, ids_sorted, cutter_s, thresh=float(options.alongside_s_thresh)
            )
            if aid != 0:
                # use distance score for alongside (interpretable & stable)
                rng = options.max_range_ahead if options.max_range_ahead is not None else options.max_range_behind
                g[1, col] = _distance_score(ds_side, rng)

    return g


def sfc_vector_4x4_from_3x3(g3: np.ndarray, *, order: SFCOrder) -> np.ndarray:
    """
    Embed 3x3 into 4x4 then vectorize into length-16 vector in SFC order.
    """
    if g3.shape != (3, 3):
        raise ValueError(f"Expected (3,3), got {g3.shape}")

    g4 = np.zeros((4, 4), dtype=float)
    g4[:3, :3] = g3

    vec = np.zeros(16, dtype=float)
    for r in range(4):
        for c in range(4):
            k = sfc_index_4x4(r, c, order)
            vec[k] = float(g4[r, c])
    return vec
