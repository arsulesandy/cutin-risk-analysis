"""Binary space-filling-curve (SFC) encoding for local traffic occupancy.

Workflow:
1. Build a 3x3 neighborhood occupancy grid around a cutter vehicle.
2. Embed into a 4x4 grid (padding row/column for fixed-size encoding).
3. Linearize with Hilbert or Morton ordering into a 16-bit integer code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal, Iterable

import numpy as np

SFCOrder = Literal["hilbert", "morton"]


# ============================================================
# 4x4 Space-filling curve indices
# - Hilbert: locality-preserving curve (what people usually mean by SFC)
# - Morton (Z-order): simpler alternative, kept as an option
# ============================================================

def _hilbert_rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    """Hilbert helper: rotate/flip coordinates in one recursion quadrant."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def hilbert_index_2d(n_bits: int, x: int, y: int) -> int:
    """
    Convert (x,y) to Hilbert distance d for grid size 2^n_bits.
    Here we use n_bits=2 for 4x4.
    """
    n = 1 << n_bits
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        s >>= 1
    return int(d)


def _part1by1_2bit(v: int) -> int:
    """Interleave two bits with zeros for 2D Morton coding."""
    # spread 2 bits: b1 b0 -> 0 b1 0 b0
    v &= 0b11
    v = (v | (v << 1)) & 0b0101
    return v


def morton_index_4x4(x: int, y: int) -> int:
    """
    Morton/Z-order index for 4x4.
    We treat x=col, y=row.
    """
    px = _part1by1_2bit(x)
    py = _part1by1_2bit(y)
    return int((py << 1) | px)


def sfc_index_4x4(row: int, col: int, order: SFCOrder) -> int:
    """
    Return SFC index in [0..15] for a 4x4 grid cell.
    row=y, col=x
    """
    r = int(row)
    c = int(col)
    if not (0 <= r < 4 and 0 <= c < 4):
        raise ValueError(f"4x4 coords out of range: row={r}, col={c}")

    if order == "hilbert":
        return hilbert_index_2d(n_bits=2, x=c, y=r)
    if order == "morton":
        return morton_index_4x4(x=c, y=r)

    raise ValueError(f"Unknown SFC order: {order}")


def encode_grid_4x4_bits(grid: np.ndarray, *, order: SFCOrder = "hilbert") -> int:
    """
    Encode a 4x4 binary grid into a 16-bit integer.
    Bit k is set if the cell whose SFC index is k is occupied.
    """
    if grid.shape != (4, 4):
        raise ValueError(f"Expected (4,4), got {grid.shape}")

    code = 0
    for r in range(4):
        for c in range(4):
            if int(grid[r, c]) != 0:
                k = sfc_index_4x4(r, c, order)
                code |= (1 << k)
    return int(code)


def decode_grid_4x4_bits(code: int, *, order: SFCOrder = "hilbert") -> np.ndarray:
    """
    Decode a 16-bit code back into a 4x4 binary grid.
    """
    g = np.zeros((4, 4), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            k = sfc_index_4x4(r, c, order)
            g[r, c] = 1 if ((int(code) >> k) & 1) else 0
    return g


# ============================================================
# Binary neighborhood grid around the cutter
# ============================================================

@dataclass(frozen=True)
class BinarySFCOptions:
    """Configuration for 3x3 neighborhood extraction and range-gating."""
    lane_col: str = "laneIndex_xy"
    sign_col: str = "sign"
    s_col: str = "s"

    # Adjacent-lane "alongside" definition using |Δs|
    alongside_s_thresh: float = 5.0

    # Optional range gates (meters)
    max_range_ahead: Optional[float] = 150.0
    max_range_behind: Optional[float] = 150.0


def _lane_key(frame: int, lane: int, sign: int) -> tuple[int, int, int]:
    """Canonical dictionary key for a lane snapshot."""
    return (int(frame), int(lane), int(sign))


def build_lane_snapshots(
        df,
        frames_needed: set[int],
        *,
        lane_col: str,
        sign_col: str,
        s_col: str,
) -> Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    snapshots[(frame, lane, sign)] = (s_sorted, id_sorted)
    Used for fast neighbor queries without scanning the full DataFrame each time.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    snap = df.loc[df["frame"].isin(frames_needed), ["frame", lane_col, sign_col, s_col, "id"]].copy()
    out: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

    for (frame, lane, sign), g in snap.groupby(["frame", lane_col, sign_col], sort=False):
        s = g[s_col].to_numpy(dtype=float)
        ids = g["id"].to_numpy(dtype=int)
        if len(s) == 0:
            continue
        order = np.argsort(s)
        out[_lane_key(frame, lane, sign)] = (s[order], ids[order])

    return out


def _nearest_ahead(s_sorted: np.ndarray, ids_sorted: np.ndarray, s0: float) -> tuple[int, float]:
    """Return nearest vehicle ahead of `s0` and its forward distance."""
    pos = int(np.searchsorted(s_sorted, s0, side="right"))
    if pos >= len(s_sorted):
        return 0, float("inf")
    return int(ids_sorted[pos]), float(s_sorted[pos] - s0)


def _nearest_behind(s_sorted: np.ndarray, ids_sorted: np.ndarray, s0: float) -> tuple[int, float]:
    """Return nearest vehicle behind `s0` and its backward distance."""
    pos = int(np.searchsorted(s_sorted, s0, side="left"))
    idx = pos - 1
    if idx < 0:
        return 0, float("inf")
    return int(ids_sorted[idx]), float(s0 - s_sorted[idx])


def _nearest_ahead_excluding(
        s_sorted: np.ndarray,
        ids_sorted: np.ndarray,
        s0: float,
        *,
        excluded_ids: set[int],
) -> tuple[int, float]:
    """Return nearest ahead vehicle not in `excluded_ids`."""
    pos = int(np.searchsorted(s_sorted, s0, side="right"))
    while pos < len(s_sorted):
        candidate_id = int(ids_sorted[pos])
        if candidate_id not in excluded_ids and candidate_id != 0:
            return candidate_id, float(s_sorted[pos] - s0)
        pos += 1
    return 0, float("inf")


def _nearest_behind_excluding(
        s_sorted: np.ndarray,
        ids_sorted: np.ndarray,
        s0: float,
        *,
        excluded_ids: set[int],
) -> tuple[int, float]:
    """Return nearest behind vehicle not in `excluded_ids`."""
    idx = int(np.searchsorted(s_sorted, s0, side="left")) - 1
    while idx >= 0:
        candidate_id = int(ids_sorted[idx])
        if candidate_id not in excluded_ids and candidate_id != 0:
            return candidate_id, float(s0 - s_sorted[idx])
        idx -= 1
    return 0, float("inf")


def _nearest_alongside(s_sorted: np.ndarray, ids_sorted: np.ndarray, s0: float, *, thresh: float) -> int:
    """Return closest lateral-neighbor candidate within absolute longitudinal threshold."""
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

    return best_id


def _get_single_row(indexed, vehicle_id: int, frame: int):
    """Retrieve a single row from a (id, frame)-indexed table."""
    try:
        row = indexed.loc[(int(vehicle_id), int(frame))]
    except Exception:
        return None
    try:
        import pandas as pd  # local import to avoid hard dependency here

        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
    except Exception:
        pass
    return row


def _center_and_half_length(indexed, vehicle_id: int, frame: int) -> tuple[float, float] | None:
    """Return longitudinal center and half length from highD bbox x/width fields."""
    row = _get_single_row(indexed, vehicle_id, frame)
    if row is None:
        return None
    try:
        x = float(row["x"])
        width = float(row["width"])
    except Exception:
        return None
    if not (np.isfinite(x) and np.isfinite(width)):
        return None
    return float(x + (0.5 * width)), float(0.5 * width)


def _nearest_alongside_with_geometry(
        *,
        indexed,
        frame: int,
        cutter_id: int | None,
        candidate_ids: Iterable[int],
        s_sorted: np.ndarray,
        ids_sorted: np.ndarray,
        s0: float,
        thresh: float,
        edge_tol: float = 0.25,
) -> int:
    """
    Geometry-aware alongside lookup.

    Accept as alongside if either:
    - center-distance criterion passes (legacy-compatible): |delta_s| <= thresh
    - longitudinal intervals overlap / almost overlap: edge_gap <= edge_tol
    """
    if indexed is None or cutter_id is None:
        return _nearest_alongside(s_sorted, ids_sorted, s0, thresh=thresh)

    cutter_geom = _center_and_half_length(indexed, int(cutter_id), int(frame))
    if cutter_geom is None:
        return _nearest_alongside(s_sorted, ids_sorted, s0, thresh=thresh)
    cutter_center, cutter_half = cutter_geom

    best_id = 0
    best_metric = float("inf")
    for cid in candidate_ids:
        if cid is None or int(cid) == 0:
            continue
        geom = _center_and_half_length(indexed, int(cid), int(frame))
        if geom is None:
            continue
        cand_center, cand_half = geom
        center_gap = abs(cand_center - cutter_center)
        edge_gap = center_gap - (cutter_half + cand_half)
        if (center_gap <= float(thresh)) or (edge_gap <= float(edge_tol)):
            metric = max(0.0, edge_gap)
            if metric < best_metric:
                best_metric = metric
                best_id = int(cid)

    if best_id != 0:
        return best_id
    return _nearest_alongside(s_sorted, ids_sorted, s0, thresh=thresh)


def build_binary_grid_3x3(
        *,
        snapshots: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]],
        indexed=None,
        frame: int,
        cutter_id: int | None = None,
        cutter_lane: int,
        cutter_sign: int,
        cutter_s: float,
        options: BinarySFCOptions,
) -> np.ndarray:
    """
    3x3 grid: rows=[preceding, alongside, following], cols=[left, same, right]
    Center cell is always 1 (the cutter).
    """
    g = np.zeros((3, 3), dtype=np.uint8)

    # cutter itself
    g[1, 1] = 1

    lane_offsets = (-1, 0, +1)
    for col, d_lane in enumerate(lane_offsets):
        lane = int(cutter_lane + d_lane)
        key = _lane_key(frame, lane, cutter_sign)
        if key not in snapshots:
            continue

        s_sorted, ids_sorted = snapshots[key]

        # Same-lane preceding/following are meaningful
        pid, ds_ahead = _nearest_ahead(s_sorted, ids_sorted, cutter_s)
        fid, ds_behind = _nearest_behind(s_sorted, ids_sorted, cutter_s)

        if options.max_range_ahead is not None and pid != 0 and ds_ahead > float(options.max_range_ahead):
            pid = 0
        if options.max_range_behind is not None and fid != 0 and ds_behind > float(options.max_range_behind):
            fid = 0

        if pid != 0:
            g[0, col] = 1
        if fid != 0:
            g[2, col] = 1

        # "Alongside" only makes sense in adjacent lanes (left/right)
        if d_lane != 0:
            pos = int(np.searchsorted(s_sorted, cutter_s, side="left"))
            candidate_ids = []
            for idx in (pos - 1, pos, pos + 1):
                if 0 <= idx < len(s_sorted):
                    candidate_ids.append(int(ids_sorted[idx]))
            aid = _nearest_alongside_with_geometry(
                indexed=indexed,
                frame=int(frame),
                cutter_id=cutter_id,
                candidate_ids=candidate_ids,
                s_sorted=s_sorted,
                ids_sorted=ids_sorted,
                s0=float(cutter_s),
                thresh=float(options.alongside_s_thresh),
            )
            # Adjacent-lane occupancy is exclusive per neighbor assignment.
            # If the same vehicle is selected as alongside and ahead/behind, fall back to
            # the next closest ahead/behind vehicle instead of dropping occupancy to zero.
            if aid != 0 and aid == pid:
                pid, ds_ahead = _nearest_ahead_excluding(
                    s_sorted,
                    ids_sorted,
                    cutter_s,
                    excluded_ids={int(aid)},
                )
                if options.max_range_ahead is not None and pid != 0 and ds_ahead > float(options.max_range_ahead):
                    pid = 0
                g[0, col] = 1 if pid != 0 else 0
            if aid != 0 and aid == fid:
                fid, ds_behind = _nearest_behind_excluding(
                    s_sorted,
                    ids_sorted,
                    cutter_s,
                    excluded_ids={int(aid)},
                )
                if options.max_range_behind is not None and fid != 0 and ds_behind > float(options.max_range_behind):
                    fid = 0
                g[2, col] = 1 if fid != 0 else 0
            if aid != 0:
                g[1, col] = 1

    return g


def embed_3x3_into_4x4(g3: np.ndarray) -> np.ndarray:
    """
    Embed 3x3 into top-left of a 4x4 grid (last row/col stay 0).
    """
    if g3.shape != (3, 3):
        raise ValueError(f"Expected (3,3), got {g3.shape}")
    g4 = np.zeros((4, 4), dtype=np.uint8)
    g4[:3, :3] = g3
    return g4


def encode_frame_binary_sfc(
        *,
        snapshots: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]],
        indexed=None,
        frame: int,
        cutter_id: int | None = None,
        cutter_lane: int,
        cutter_sign: int,
        cutter_s: float,
        options: BinarySFCOptions,
        order: SFCOrder = "hilbert",
) -> tuple[int, np.ndarray]:
    """
    Returns: (code_16bit, grid_3x3)
    """
    g3 = build_binary_grid_3x3(
        snapshots=snapshots,
        indexed=indexed,
        frame=frame,
        cutter_id=cutter_id,
        cutter_lane=cutter_lane,
        cutter_sign=cutter_sign,
        cutter_s=cutter_s,
        options=options,
    )
    g4 = embed_3x3_into_4x4(g3)
    code = encode_grid_4x4_bits(g4, order=order)
    return code, g3
