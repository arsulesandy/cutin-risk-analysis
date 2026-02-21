from __future__ import annotations

"""
Step 16: Predict lane changes and cut-ins using binary SFC (space-filling curve) signatures.

We build a 3x3 occupancy grid around a vehicle for each frame:
  rows = [preceding, alongside, following]
  cols = [left, same, right]

Center cell (alongside,same) is always 1 (ego).

Occupancy is computed from geometry + inferred laneIndex_xy:
  - preceding: nearest vehicle ahead in that lane within ahead_m
  - following: nearest vehicle behind in that lane within behind_m
  - alongside: any vehicle with |Δs| <= alongside_m in that lane

Then we embed 3x3 into a 4x4 grid (top-left) and linearize it with a Hilbert curve
(order=2 -> 4x4 -> 16 positions), producing a 16-dim binary feature vector v00..v15.

Tasks:
  A) Lane-change prediction:
     Decision window (e.g., 2s before t0) -> label: lane change starts at t0 (1) vs non-event (0).
     Negatives are sampled from frames that are not near lane changes.

  B) Cut-in prediction (among lane changes):
     Decision window before lane change start -> label: cut-in (1) vs non cut-in lane change (0).

Mirroring:
  - For lane-change/cut-in positive samples, we optionally mirror LEFT lane changes into RIGHT
    (swap left/right columns) so patterns align.
  - For negative lane-change samples, we randomly mirror 50% to avoid directional bias.

Outputs (default):
  outputs/reports/step16_sfc_predict/
    - sfc_lanechange_dataset.csv
    - sfc_cutin_dataset.csv
    - lanechange_loocv.csv
    - cutin_loocv.csv

Note:
  This is a baseline. You can later extend to distance/TTC weighted occupancy (Step 15B).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.paths import dataset_root_path, output_path
from cutin_risk.reconstruction.lanes import parse_lane_markings, infer_lane_index


# ----------------------------
# Event dataclasses
# ----------------------------

@dataclass(frozen=True)
class LaneChangeEvent:
    vehicle_id: int
    from_lane: int
    to_lane: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class CutInEvent:
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


# ----------------------------
# Options
# ----------------------------

@dataclass(frozen=True)
class LaneChangeOptions:
    min_stable_before_frames: int = 25
    min_stable_after_frames: int = 25
    ignore_lane_ids: tuple[int, ...] = (0,)
    lane_col: str = "laneId"


@dataclass(frozen=True)
class CutInOptions:
    # A cut-in is accepted if the follower relation persists >= min_relation_frames.
    min_relation_frames: int = 19
    # We cap the relation segment to keep computations stable (like your Step 3 report).
    max_relation_frames: int = 50


@dataclass(frozen=True)
class SFCOptions:
    decision_seconds: float = 2.0
    ahead_m: float = 50.0
    behind_m: float = 50.0
    alongside_m: float = 10.0

    lane_col: str = "laneIndex_xy"
    sign_col: str = "sign"
    s_col: str = "s"

    mirror_left_to_right: bool = True
    random_mirror_negatives: bool = True


# ----------------------------
# Metrics
# ----------------------------

@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return Metrics(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)


def best_threshold_for_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    # simple grid search
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        pred = prob >= thr
        m = compute_metrics(y_true, pred)
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_thr = float(thr)
    return best_thr


# ----------------------------
# Step 2-ish: lane change detection
# ----------------------------

def detect_lane_changes(df: pd.DataFrame, *, options: LaneChangeOptions | None = None) -> list[LaneChangeEvent]:
    options = options or LaneChangeOptions()

    required = {"id", "frame", "time", options.lane_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_lane_changes missing required columns: {sorted(missing)}")

    def is_ignored(lane_id: int) -> bool:
        return int(lane_id) in options.ignore_lane_ids

    events: list[LaneChangeEvent] = []
    lane_col = options.lane_col

    for vid, g in df.groupby("id", sort=False):
        lane = g[lane_col].to_numpy()
        frames = g["frame"].to_numpy()
        times = g["time"].to_numpy()

        n = len(lane)
        if n < (options.min_stable_before_frames + options.min_stable_after_frames + 2):
            continue

        i = 0
        while i < n - 1:
            cur_lane = int(lane[i])
            nxt_lane = int(lane[i + 1])

            if cur_lane == nxt_lane:
                i += 1
                continue

            if is_ignored(cur_lane) or is_ignored(nxt_lane):
                i += 1
                continue

            from_lane = cur_lane
            to_lane = nxt_lane

            # stability BEFORE: consecutive from_lane ending at i
            k = i
            while k >= 0 and int(lane[k]) == from_lane:
                k -= 1
            stable_before_len = i - k
            if stable_before_len < options.min_stable_before_frames:
                i += 1
                continue

            start_idx = i + 1

            # stability AFTER: consecutive to_lane starting at start_idx
            j = start_idx
            while j < n and int(lane[j]) == to_lane:
                j += 1
            stable_after_len = j - start_idx

            if stable_after_len >= options.min_stable_after_frames:
                end_idx = j - 1
                events.append(
                    LaneChangeEvent(
                        vehicle_id=int(vid),
                        from_lane=from_lane,
                        to_lane=to_lane,
                        start_frame=int(frames[start_idx]),
                        end_frame=int(frames[end_idx]),
                        start_time=float(times[start_idx]),
                        end_time=float(times[end_idx]),
                    )
                )

            i = j

    return events


# ----------------------------
# Step 3-ish: cut-in detection (from lane changes + followingId persistence)
# ----------------------------

def detect_cutins_from_lane_changes(
        df: pd.DataFrame,
        lane_changes: list[LaneChangeEvent],
        *,
        options: CutInOptions | None = None,
) -> list[CutInEvent]:
    options = options or CutInOptions()

    required = {"id", "frame", "time", "followingId", "laneId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_cutins_from_lane_changes missing required columns: {sorted(missing)}")

    indexed = df.set_index(["id", "frame"], drop=False).sort_index()

    cutins: list[CutInEvent] = []

    for lc in lane_changes:
        vid = lc.vehicle_id
        start = lc.start_frame
        end = lc.end_frame

        # Scan from lane-change start for first followerId != 0
        search_end = min(end, start + options.max_relation_frames - 1)

        follower_id = 0
        rel_start = None

        for f in range(start, search_end + 1):
            try:
                row = indexed.loc[(vid, f)]
            except KeyError:
                continue

            # Must already be in target lane (should be true by construction)
            if int(row["laneId"]) != int(lc.to_lane):
                continue

            fid = int(row["followingId"])
            if fid != 0:
                follower_id = fid
                rel_start = f
                break

        if rel_start is None or follower_id == 0:
            continue

        # Extend while followerId stays the same (and non-zero)
        rel_end = rel_start
        max_end = min(end, rel_start + options.max_relation_frames - 1)

        for f in range(rel_start + 1, max_end + 1):
            try:
                row = indexed.loc[(vid, f)]
            except KeyError:
                break
            fid = int(row["followingId"])
            if fid != follower_id:
                break
            rel_end = f

        duration = rel_end - rel_start + 1
        if duration < options.min_relation_frames:
            continue

        # times
        try:
            t_rel_start = float(indexed.loc[(vid, rel_start)]["time"])
            t_rel_end = float(indexed.loc[(vid, rel_end)]["time"])
        except KeyError:
            continue

        cutins.append(
            CutInEvent(
                cutter_id=vid,
                follower_id=follower_id,
                from_lane=lc.from_lane,
                to_lane=lc.to_lane,
                lane_change_start_frame=lc.start_frame,
                lane_change_end_frame=lc.end_frame,
                lane_change_start_time=lc.start_time,
                lane_change_end_time=lc.end_time,
                relation_start_frame=rel_start,
                relation_end_frame=rel_end,
                relation_start_time=t_rel_start,
                relation_end_time=t_rel_end,
            )
        )

    return cutins


# ----------------------------
# Hilbert (order 2) utilities
# ----------------------------

def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    # Standard Hilbert rotation (Wikipedia)
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    # n must be power of 2
    t = d
    x = 0
    y = 0
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_order_coords(n: int) -> list[tuple[int, int]]:
    # returns coords in Hilbert traversal order
    return [hilbert_d2xy(n, d) for d in range(n * n)]


HILBERT_4x4 = hilbert_order_coords(4)  # 16 (x,y) positions


# ----------------------------
# SFC feature extraction
# ----------------------------

def mirror_3x3(mat: np.ndarray) -> np.ndarray:
    # swap left/right columns
    out = mat.copy()
    out[:, 0], out[:, 2] = mat[:, 2], mat[:, 0]
    return out


def embed_3x3_in_4x4(mat3: np.ndarray) -> np.ndarray:
    mat4 = np.zeros((4, 4), dtype=np.int8)
    mat4[0:3, 0:3] = mat3.astype(np.int8)
    return mat4


def sfc_bits_16_from_3x3(mat3: np.ndarray) -> np.ndarray:
    mat4 = embed_3x3_in_4x4(mat3)
    bits = np.zeros(16, dtype=np.int8)
    for i, (x, y) in enumerate(HILBERT_4x4):
        bits[i] = mat4[y, x]
    return bits


def build_lane_snapshots(
        df: pd.DataFrame,
        frames_needed: set[int],
        *,
        lane_col: str,
        sign_col: str,
        s_col: str,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    """
    snapshots[(frame, lane, sign)] = (s_sorted, ids_sorted)
    """
    use = df.loc[df["frame"].isin(list(frames_needed)), ["frame", lane_col, sign_col, s_col, "id"]].copy()
    use = use.dropna(subset=[lane_col, sign_col, s_col, "id"])

    snapshots: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    for (frame, lane, sign), g in use.groupby(["frame", lane_col, sign_col], sort=False):
        s = g[s_col].to_numpy(dtype=float)
        ids = g["id"].to_numpy(dtype=int)
        if len(s) == 0:
            continue
        order = np.argsort(s)
        snapshots[(int(frame), int(lane), int(sign))] = (s[order], ids[order])

    return snapshots


def _has_any_in_range(s_sorted: np.ndarray, cutter_s: float, band: float) -> bool:
    lo = cutter_s - band
    hi = cutter_s + band
    a = int(np.searchsorted(s_sorted, lo, side="left"))
    b = int(np.searchsorted(s_sorted, hi, side="right"))
    return b > a


def _preceding_within(s_sorted: np.ndarray, cutter_s: float, ahead_m: float) -> bool:
    pos = int(np.searchsorted(s_sorted, cutter_s, side="right"))
    if pos >= len(s_sorted):
        return False
    return (s_sorted[pos] - cutter_s) <= ahead_m


def _following_within(s_sorted: np.ndarray, cutter_s: float, behind_m: float) -> bool:
    pos = int(np.searchsorted(s_sorted, cutter_s, side="left")) - 1
    if pos < 0:
        return False
    return (cutter_s - s_sorted[pos]) <= behind_m


def occupancy_3x3_for_vehicle(
        snapshots: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
        *,
        frame: int,
        lane: int,
        sign: int,
        cutter_s: float,
        ahead_m: float,
        behind_m: float,
        alongside_m: float,
) -> np.ndarray:
    """
    Returns 3x3 matrix:
      rows: preceding, alongside, following
      cols: left, same, right
    Center cell (alongside,same) is ego=1.
    """
    mat = np.zeros((3, 3), dtype=np.int8)
    mat[1, 1] = 1  # ego

    for col, lane_off in enumerate([-1, 0, 1]):
        lane2 = lane + lane_off
        key = (int(frame), int(lane2), int(sign))
        if key not in snapshots:
            continue

        s_sorted, _ids_sorted = snapshots[key]

        # preceding
        if _preceding_within(s_sorted, cutter_s, ahead_m):
            mat[0, col] = 1

        # following
        if _following_within(s_sorted, cutter_s, behind_m):
            mat[2, col] = 1

        # alongside (skip center: already ego)
        if not (lane_off == 0):
            if _has_any_in_range(s_sorted, cutter_s, alongside_m):
                mat[1, col] = 1

    return mat


def extract_sfc_window_feature(
        indexed: pd.DataFrame,
        snapshots: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
        *,
        vehicle_id: int,
        t0_frame: int,
        pre_frames: int,
        sfc_opt: SFCOptions,
        mirror: bool,
) -> np.ndarray:
    """
    Mean 16-dim SFC bits over frames [t0-pre_frames, t0-1].
    """
    bits_list: list[np.ndarray] = []

    start = max(1, int(t0_frame) - int(pre_frames))
    end = int(t0_frame) - 1
    if end < start:
        return np.full(16, np.nan, dtype=float)

    for f in range(start, end + 1):
        try:
            row = indexed.loc[(vehicle_id, f)]
        except KeyError:
            continue

        lane = row.get(sfc_opt.lane_col, np.nan)
        if pd.isna(lane):
            continue
        lane = int(lane)

        sign = int(row[sfc_opt.sign_col])
        cutter_s = float(row[sfc_opt.s_col])

        mat3 = occupancy_3x3_for_vehicle(
            snapshots,
            frame=f,
            lane=lane,
            sign=sign,
            cutter_s=cutter_s,
            ahead_m=sfc_opt.ahead_m,
            behind_m=sfc_opt.behind_m,
            alongside_m=sfc_opt.alongside_m,
        )

        if mirror:
            mat3 = mirror_3x3(mat3)

        bits = sfc_bits_16_from_3x3(mat3).astype(float)
        bits_list.append(bits)

    if not bits_list:
        return np.full(16, np.nan, dtype=float)

    X = np.vstack(bits_list)
    return np.mean(X, axis=0)


# ----------------------------
# Dataset building
# ----------------------------

def normalize_recording_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit():
        return f"{int(s):02d}"
    return s


def infer_sign_and_s(df: pd.DataFrame) -> pd.DataFrame:
    # drivingDirection: {1,2} in highD
    dir_to_sign = {1: -1, 2: 1}
    df = df.copy()
    df["sign"] = df["drivingDirection"].map(dir_to_sign)

    if df["sign"].isna().any():
        vx = pd.to_numeric(df["xVelocity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        vx_sign = np.sign(vx)
        vx_sign[vx_sign == 0] = 1
        df["sign"] = df["sign"].fillna(pd.Series(vx_sign, index=df.index))

    df["sign"] = df["sign"].fillna(1).astype(int)
    df["s"] = df["sign"].astype(float) * df["x"].astype(float)
    return df


def lane_change_direction_mirror_flag(indexed: pd.DataFrame, vehicle_id: int, start_frame: int) -> bool:
    """
    Decide mirroring using inferred laneIndex_xy:
      delta = laneIndex(start_frame) - laneIndex(start_frame-1)
      mirror if delta < 0  (left becomes right)
    """
    try:
        a = indexed.loc[(vehicle_id, start_frame - 1)]["laneIndex_xy"]
        b = indexed.loc[(vehicle_id, start_frame)]["laneIndex_xy"]
    except KeyError:
        return False
    if pd.isna(a) or pd.isna(b):
        return False
    delta = int(b) - int(a)
    return delta < 0


def sample_negative_lanechange_frames(
        df: pd.DataFrame,
        lane_changes: list[LaneChangeEvent],
        *,
        pre_frames: int,
        n_neg: int,
        seed: int = 0,
) -> list[tuple[int, int]]:
    """
    Sample (vehicle_id, t0_frame) negatives not near any lane-change interval.
    """
    rng = np.random.default_rng(seed)

    excluded: set[tuple[int, int]] = set()
    for lc in lane_changes:
        lo = max(1, lc.start_frame - pre_frames)
        hi = lc.end_frame + pre_frames
        for f in range(lo, hi + 1):
            excluded.add((lc.vehicle_id, f))

    ids = df["id"].to_numpy(dtype=int)
    frames = df["frame"].to_numpy(dtype=int)
    n = len(df)

    neg: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(20000, 20 * n_neg)

    while len(neg) < n_neg and attempts < max_attempts:
        attempts += 1
        idx = int(rng.integers(0, n))
        vid = int(ids[idx])
        t0 = int(frames[idx])

        if t0 <= pre_frames:
            continue
        if (vid, t0) in excluded:
            continue

        neg.append((vid, t0))

    return neg


def build_datasets_for_recording(
        dataset_root: Path,
        recording_id: str,
        *,
        lc_opt: LaneChangeOptions,
        ci_opt: CutInOptions,
        sfc_opt: SFCOptions,
        seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      lanechange_df: samples for lane change prediction
      cutin_df: samples for cut-in prediction among lane changes
    """
    rec = load_highd_recording(dataset_root, recording_id)
    df = build_tracking_table(rec)

    # infer lanes from geometry
    markings = parse_lane_markings(rec.recording_meta)
    df = df.join(infer_lane_index(df, markings))  # adds laneIndex_xy

    # sign + s coordinate
    df = infer_sign_and_s(df)

    # lane changes + cut-ins (labels)
    lane_changes = detect_lane_changes(df, options=lc_opt)
    cutins = detect_cutins_from_lane_changes(df, lane_changes, options=ci_opt)

    # map lane-change start to cut-in label
    cutin_key = {(c.cutter_id, c.lane_change_start_frame): True for c in cutins}

    indexed = df.set_index(["id", "frame"], drop=False).sort_index()
    frame_rate = float(rec.recording_meta.loc[0, "frameRate"])
    pre_frames = int(round(sfc_opt.decision_seconds * frame_rate))

    # build samples list
    lanechange_pos = [(lc.vehicle_id, lc.start_frame, True) for lc in lane_changes]
    lanechange_neg_pairs = sample_negative_lanechange_frames(df, lane_changes, pre_frames=pre_frames, n_neg=len(lanechange_pos), seed=seed)
    lanechange_neg = [(vid, t0, False) for (vid, t0) in lanechange_neg_pairs]

    # cut-in samples: all lane changes
    cutin_samples = []
    for lc in lane_changes:
        y = bool(cutin_key.get((lc.vehicle_id, lc.start_frame), False))
        cutin_samples.append((lc.vehicle_id, lc.start_frame, y))

    # Frames needed for SFC snapshots (all decision windows)
    frames_needed: set[int] = set()

    def add_window_frames(t0: int) -> None:
        start = max(1, t0 - pre_frames)
        for f in range(start, t0):
            frames_needed.add(int(f))

    for vid, t0, _y in lanechange_pos + lanechange_neg + cutin_samples:
        add_window_frames(int(t0))

    snapshots = build_lane_snapshots(
        df,
        frames_needed,
        lane_col=sfc_opt.lane_col,
        sign_col=sfc_opt.sign_col,
        s_col=sfc_opt.s_col,
    )

    rng = np.random.default_rng(seed)

    # Build lane-change dataset (pos+neg)
    rows_lc: list[dict[str, object]] = []
    for vid, t0, y in lanechange_pos + lanechange_neg:
        mirror = False
        if sfc_opt.mirror_left_to_right and y:
            mirror = lane_change_direction_mirror_flag(indexed, vid, t0)
        elif sfc_opt.random_mirror_negatives and (not y):
            mirror = bool(rng.integers(0, 2))

        feat = extract_sfc_window_feature(
            indexed,
            snapshots,
            vehicle_id=vid,
            t0_frame=t0,
            pre_frames=pre_frames,
            sfc_opt=sfc_opt,
            mirror=mirror,
        )
        row = {
            "recording_id": recording_id,
            "vehicle_id": int(vid),
            "t0_frame": int(t0),
            "label_lanechange": bool(y),
            "mirror": bool(mirror),
        }
        for i in range(16):
            row[f"v{i:02d}"] = float(feat[i])
        rows_lc.append(row)

    lanechange_df = pd.DataFrame(rows_lc).dropna()

    # Build cut-in dataset (lane changes only)
    rows_ci: list[dict[str, object]] = []
    for vid, t0, y in cutin_samples:
        mirror = False
        if sfc_opt.mirror_left_to_right:
            mirror = lane_change_direction_mirror_flag(indexed, vid, t0)

        feat = extract_sfc_window_feature(
            indexed,
            snapshots,
            vehicle_id=vid,
            t0_frame=t0,
            pre_frames=pre_frames,
            sfc_opt=sfc_opt,
            mirror=mirror,
        )
        row = {
            "recording_id": recording_id,
            "vehicle_id": int(vid),
            "t0_frame": int(t0),
            "label_cutin": bool(y),
            "mirror": bool(mirror),
        }
        for i in range(16):
            row[f"v{i:02d}"] = float(feat[i])
        rows_ci.append(row)

    cutin_df = pd.DataFrame(rows_ci).dropna()

    return lanechange_df, cutin_df


# ----------------------------
# LOOCV evaluation
# ----------------------------

def loocv_eval(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    recs = sorted(df["recording_id"].astype(str).unique().tolist())
    feat_cols = [c for c in df.columns if c.startswith("v")]

    rows = []

    for heldout in recs:
        train = df[df["recording_id"].astype(str) != heldout]
        test = df[df["recording_id"].astype(str) == heldout]

        X_tr = train[feat_cols].to_numpy(dtype=float)
        y_tr = train[label_col].to_numpy(dtype=bool)

        X_te = test[feat_cols].to_numpy(dtype=float)
        y_te = test[label_col].to_numpy(dtype=bool)

        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=2000),
        )
        model.fit(X_tr, y_tr)

        prob_tr = model.predict_proba(X_tr)[:, 1]
        thr = best_threshold_for_f1(y_tr, prob_tr)

        prob_te = model.predict_proba(X_te)[:, 1]
        pred_te = prob_te >= thr

        m = compute_metrics(y_te, pred_te)

        rows.append(
            {
                "heldout_recording": heldout,
                "threshold": thr,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "tn": m.tn,
                "n_test": int(len(y_te)),
                "pos_rate_test": float(np.mean(y_te)) if len(y_te) else 0.0,
            }
        )

    return pd.DataFrame(rows)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(dataset_root_path()))
    parser.add_argument("--recordings", type=str, default="01,02,03,04,05")
    parser.add_argument("--out-dir", type=str, default=str(output_path("reports/step16_sfc_predict")))

    parser.add_argument("--decision-seconds", type=float, default=2.0)
    parser.add_argument("--ahead-m", type=float, default=50.0)
    parser.add_argument("--behind-m", type=float, default=50.0)
    parser.add_argument("--alongside-m", type=float, default=10.0)

    parser.add_argument("--seed", type=int, default=0)

    # lane change / cut-in definition knobs
    parser.add_argument("--min-stable-before", type=int, default=25)
    parser.add_argument("--min-stable-after", type=int, default=25)
    parser.add_argument("--min-relation-frames", type=int, default=19)
    parser.add_argument("--max-relation-frames", type=int, default=50)

    parser.add_argument("--no-mirror", action="store_true", help="Disable mirroring of left lane changes.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = [normalize_recording_id(x) for x in args.recordings.split(",") if x.strip()]

    lc_opt = LaneChangeOptions(
        min_stable_before_frames=int(args.min_stable_before),
        min_stable_after_frames=int(args.min_stable_after),
    )
    ci_opt = CutInOptions(
        min_relation_frames=int(args.min_relation_frames),
        max_relation_frames=int(args.max_relation_frames),
    )
    sfc_opt = SFCOptions(
        decision_seconds=float(args.decision_seconds),
        ahead_m=float(args.ahead_m),
        behind_m=float(args.behind_m),
        alongside_m=float(args.alongside_m),
        mirror_left_to_right=(not args.no_mirror),
        random_mirror_negatives=(not args.no_mirror),
    )

    all_lc = []
    all_ci = []

    for rid in recordings:
        print(f"=== Step 16: building datasets for recording {rid} ===")
        lc_df, ci_df = build_datasets_for_recording(
            dataset_root,
            rid,
            lc_opt=lc_opt,
            ci_opt=ci_opt,
            sfc_opt=sfc_opt,
            seed=args.seed,
        )
        all_lc.append(lc_df)
        all_ci.append(ci_df)

        print(f"  lanechange samples: {len(lc_df)} | cutin samples: {len(ci_df)}")

    lanechange_df = pd.concat(all_lc, ignore_index=True) if all_lc else pd.DataFrame()
    cutin_df = pd.concat(all_ci, ignore_index=True) if all_ci else pd.DataFrame()

    out_lc = out_dir / "sfc_lanechange_dataset.csv"
    out_ci = out_dir / "sfc_cutin_dataset.csv"
    lanechange_df.to_csv(out_lc, index=False)
    cutin_df.to_csv(out_ci, index=False)

    print("\nSaved datasets:")
    print(" ", out_lc)
    print(" ", out_ci)

    # LOOCV evaluation
    if not lanechange_df.empty:
        lc_loocv = loocv_eval(lanechange_df, "label_lanechange")
        out_lc_cv = out_dir / "lanechange_loocv.csv"
        lc_loocv.to_csv(out_lc_cv, index=False)
        print("\nLane-change LOOCV:")
        print(lc_loocv.to_string(index=False))
        print("Saved:", out_lc_cv)

    if not cutin_df.empty:
        ci_loocv = loocv_eval(cutin_df, "label_cutin")
        out_ci_cv = out_dir / "cutin_loocv.csv"
        ci_loocv.to_csv(out_ci_cv, index=False)
        print("\nCut-in LOOCV (among lane changes):")
        print(ci_loocv.to_string(index=False))
        print("Saved:", out_ci_cv)


if __name__ == "__main__":
    main()
