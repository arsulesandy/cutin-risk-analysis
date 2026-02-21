"""
Step 13B: Early-warning evaluation without using the true follower id.

Risk label (same as earlier steps):
  risky := execution_thw_min < 0.7s

Decision-stage (t in [-2s, 0s)) features are computed WITHOUT the oracle follower_id.
Instead, at each frame we select a candidate follower from geometry:

  candidate follower = closest vehicle behind the cutter in the TARGET lane (same direction)

Features per event:
  - decision_lat_v_abs_max: max(|cutter yVelocity|) over decision window
  - decision_speed_delta_median_candidate: median(cutter_speed - candidate_speed) over frames where a candidate exists
  - decision_candidate_ratio: fraction of decision frames where a valid candidate exists
  - candidate_id_mode / candidate_mode_ratio / candidate_id_last: stability diagnostics for candidate selection

If the merged input table contains an oracle follower id column (e.g., follower_id),
we also report how often our geometric candidate matches the oracle follower:
  - match_mode, match_last, match_ratio
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.io.progress import iter_with_progress
from cutin_risk.paths import dataset_root_path, output_path
from cutin_risk.reconstruction.lanes import infer_lane_index, parse_lane_markings
from cutin_risk.thesis_config import thesis_float, thesis_str


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


def normalize_recording_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit():
        return f"{int(s):02d}"
    return s


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


def detect_oracle_follower_column(df: pd.DataFrame) -> str | None:
    candidates = ["follower_id", "followerId", "followerID", "follower"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_laneid_to_inferred_lane_map(df: pd.DataFrame, *, lane_col: str, laneid_col: str = "laneId") -> dict[int, int]:
    if laneid_col not in df.columns or lane_col not in df.columns:
        return {}

    g = df.dropna(subset=[laneid_col, lane_col])
    if g.empty:
        return {}

    mapping: dict[int, int] = {}
    for lane_id, gg in g.groupby(laneid_col, sort=False):
        # mode of inferred lane index for this laneId
        mapping[int(lane_id)] = int(gg[lane_col].value_counts().idxmax())
    return mapping


def build_lane_snapshots(
        df: pd.DataFrame,
        frames_needed: set[int],
        *,
        lane_col: str,
        sign_col: str = "sign",
        s_col: str = "s",
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    """
    snapshots[(frame, lane, sign)] = (s_sorted, id_sorted)

    s_sorted is the signed longitudinal coordinate along the direction of travel.
    """
    cols = ["frame", lane_col, sign_col, s_col, "id"]
    snap = df.loc[df["frame"].isin(frames_needed), cols].copy()

    snapshots: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (frame, lane, sign), g in snap.groupby(["frame", lane_col, sign_col], sort=False):
        s = g[s_col].to_numpy(dtype=float)
        ids = g["id"].to_numpy(dtype=int)
        if len(s) == 0:
            continue

        order = np.argsort(s)
        key = (int(frame), int(lane), int(sign))
        snapshots[key] = (s[order], ids[order])

    return snapshots


def candidate_follower_id(
        snapshots: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
        *,
        frame: int,
        target_lane: int,
        sign: int,
        cutter_s: float,
) -> int:
    """
    Closest vehicle behind the cutter in the target lane at a given frame.
    Returns 0 if none exists.
    """
    key = (int(frame), int(target_lane), int(sign))
    if key not in snapshots:
        return 0

    s_sorted, ids_sorted = snapshots[key]

    # insertion point of cutter_s; element immediately left is behind (smaller s)
    pos = int(np.searchsorted(s_sorted, cutter_s, side="left"))
    idx = pos - 1
    if idx < 0:
        return 0
    return int(ids_sorted[idx])


def grid_search_best_thresholds(
        lat: np.ndarray,
        spd: np.ndarray,
        y: np.ndarray,
        lat_grid: np.ndarray,
        spd_grid: np.ndarray,
) -> tuple[float, float, Metrics]:
    best_lat = float(lat_grid[0])
    best_spd = float(spd_grid[0])
    best_m = Metrics(0, 0, 0, 0, 0.0, 0.0, 0.0)

    for lat_thr in lat_grid:
        lat_mask = lat >= lat_thr
        for spd_thr in spd_grid:
            pred = lat_mask & (spd >= spd_thr)
            m = compute_metrics(y, pred)

            # Maximize F1, with stable tie-breakers
            if (m.f1, m.recall, m.precision) > (best_m.f1, best_m.recall, best_m.precision):
                best_lat = float(lat_thr)
                best_spd = float(spd_thr)
                best_m = m

    return best_lat, best_spd, best_m


def derive_direction_sign(df: pd.DataFrame) -> pd.Series:
    """
    Returns a +/-1 sign per row.
    Prefer drivingDirection, fallback to xVelocity sign if needed.
    """
    # highD drivingDirection is typically {1,2}:
    #   2 -> x increases
    #   1 -> x decreases
    dir_to_sign = {1: -1, 2: 1}
    sign = df["drivingDirection"].map(dir_to_sign)

    if sign.isna().any():
        vx = pd.to_numeric(df["xVelocity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        vx_sign = np.sign(vx)
        vx_sign[vx_sign == 0] = 1
        sign = sign.fillna(pd.Series(vx_sign, index=df.index))

    return sign.fillna(1).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 13B: realistic candidate-follower early warning.")
    parser.add_argument(
        "--merged-csv",
        type=str,
        default=str(output_path("reports/step9_batch/cutin_stage_features_merged.csv")),
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(dataset_root_path()),
    )
    parser.add_argument("--thw-risk", type=float, default=thesis_float("risk_label.thw_risk", 0.7, min_value=0.0))

    parser.add_argument(
        "--decision-seconds",
        type=float,
        default=thesis_float("step13b.decision_seconds", 2.0, min_value=0.0),
        help="Decision window length before t0.",
    )
    parser.add_argument("--lane-col", type=str, default=thesis_str("step13b.lane_col", "laneIndex_xy"))

    # Compare against Step 11 global rule thresholds (for reference)
    parser.add_argument("--baseline-lat-thr", type=float, default=thesis_float("step13b.baseline_lat_thr", 0.80))
    parser.add_argument("--baseline-spd-thr", type=float, default=thesis_float("step13b.baseline_spd_thr", 1.25))

    # Threshold search grid
    parser.add_argument("--lat-min", type=float, default=thesis_float("step13b.lat_min", 0.6))
    parser.add_argument("--lat-max", type=float, default=thesis_float("step13b.lat_max", 1.6))
    parser.add_argument("--lat-step", type=float, default=thesis_float("step13b.lat_step", 0.05, min_value=1e-9))
    parser.add_argument("--spd-min", type=float, default=thesis_float("step13b.spd_min", 0.0))
    parser.add_argument("--spd-max", type=float, default=thesis_float("step13b.spd_max", 10.0))
    parser.add_argument("--spd-step", type=float, default=thesis_float("step13b.spd_step", 0.25, min_value=1e-9))

    parser.add_argument("--out-dir", type=str, default=str(output_path("reports/step13b_warning")))
    args = parser.parse_args()

    merged_csv = Path(args.merged_csv)
    if not merged_csv.exists():
        raise FileNotFoundError(f"Missing input: {merged_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(merged_csv)

    required_cols = {"recording_id", "cutter_id", "to_lane", "t0_frame", "execution_thw_min"}
    missing = required_cols - set(events.columns)
    if missing:
        raise ValueError(f"merged CSV missing required columns: {sorted(missing)}")

    oracle_col = detect_oracle_follower_column(events)

    # Normalize ids / types
    events = events.copy()
    events["recording_id"] = events["recording_id"].apply(normalize_recording_id)
    events["cutter_id"] = pd.to_numeric(events["cutter_id"], errors="coerce").astype("Int64")
    events["to_lane"] = pd.to_numeric(events["to_lane"], errors="coerce").astype("Int64")
    events["t0_frame"] = pd.to_numeric(events["t0_frame"], errors="coerce").astype("Int64")
    events["execution_thw_min"] = pd.to_numeric(events["execution_thw_min"], errors="coerce")

    if oracle_col is not None:
        events[oracle_col] = pd.to_numeric(events[oracle_col], errors="coerce").astype("Int64")

    events = events.dropna(subset=["cutter_id", "to_lane", "t0_frame", "execution_thw_min"]).copy()
    events["risk_thw"] = events["execution_thw_min"] < float(args.thw_risk)

    rows_out: list[dict[str, object]] = []

    dataset_root = Path(args.dataset_root)

    recording_ids = sorted(events["recording_id"].unique().tolist())
    for _, _, rid in iter_with_progress(
        recording_ids,
        label="Step 13B recordings",
        item_name="recording",
    ):
        ev_r = events.loc[events["recording_id"] == rid].copy()
        if ev_r.empty:
            continue

        rec = load_highd_recording(dataset_root, rid)
        df = build_tracking_table(rec)

        # Infer lanes from geometry
        markings = parse_lane_markings(rec.recording_meta)
        df = df.join(infer_lane_index(df, markings))  # adds laneIndex_xy

        # Signed travel direction and signed longitudinal coordinate
        df["sign"] = derive_direction_sign(df)
        df["s"] = df["sign"].astype(float) * df["x"].astype(float)

        # Fast lookup by (id, frame)
        indexed = df.set_index(["id", "frame"], drop=False).sort_index()

        frame_rate = float(rec.recording_meta.loc[0, "frameRate"])
        pre_frames = int(round(float(args.decision_seconds) * frame_rate))

        # Determine if we need to map event lanes to inferred lanes
        inferred_lanes = set(df[args.lane_col].dropna().astype(int).unique().tolist())
        laneid_to_inferred = build_laneid_to_inferred_lane_map(df, lane_col=args.lane_col)

        # Build frame set needed for snapshots (only decision frames)
        frames_needed: set[int] = set()
        for _, e in ev_r.iterrows():
            t0 = int(e["t0_frame"])
            start = max(1, t0 - pre_frames)
            frames_needed.update(range(start, t0))

        snapshots = build_lane_snapshots(df, frames_needed, lane_col=args.lane_col)

        # Compute realistic decision-stage features per event
        for _, e in ev_r.iterrows():
            cutter_id = int(e["cutter_id"])
            to_lane_raw = int(e["to_lane"])
            t0 = int(e["t0_frame"])

            # Map lane if needed (robust across representations)
            to_lane = to_lane_raw
            if inferred_lanes and (to_lane not in inferred_lanes):
                to_lane = int(laneid_to_inferred.get(to_lane_raw, to_lane_raw))

            start = max(1, t0 - pre_frames)
            end = t0 - 1

            lat_vals: list[float] = []
            deltas: list[float] = []
            cand_ids: list[int] = []

            frames_with_cutter = 0

            for f in range(start, end + 1):
                try:
                    cutter_row = indexed.loc[(cutter_id, f)]
                except KeyError:
                    continue

                frames_with_cutter += 1

                lat_vals.append(abs(float(cutter_row["yVelocity"])))
                cutter_speed = float(cutter_row["speed"])
                sign = int(cutter_row["sign"])
                cutter_s = float(cutter_row["s"])

                cand_id = candidate_follower_id(
                    snapshots,
                    frame=f,
                    target_lane=to_lane,
                    sign=sign,
                    cutter_s=cutter_s,
                )
                if cand_id == 0:
                    continue

                # cand_id originates from the same df snapshot; this should almost always exist
                try:
                    cand_row = indexed.loc[(cand_id, f)]
                except KeyError:
                    continue

                cand_speed = float(cand_row["speed"])
                deltas.append(cutter_speed - cand_speed)
                cand_ids.append(int(cand_id))

            decision_lat_v = float(np.max(lat_vals)) if lat_vals else float("nan")
            decision_speed_delta = float(np.median(deltas)) if deltas else 0.0

            frames_with_candidate = len(cand_ids)
            candidate_ratio = (frames_with_candidate / frames_with_cutter) if frames_with_cutter else 0.0

            # Candidate stability diagnostics
            candidate_id_last = cand_ids[-1] if cand_ids else 0
            candidate_id_mode = 0
            candidate_mode_ratio = 0.0
            if cand_ids:
                values, counts = np.unique(np.asarray(cand_ids, dtype=int), return_counts=True)
                i = int(np.argmax(counts))
                candidate_id_mode = int(values[i])
                candidate_mode_ratio = float(counts[i] / frames_with_candidate) if frames_with_candidate else 0.0

            row: dict[str, object] = {
                "recording_id": rid,
                "cutter_id": cutter_id,
                "to_lane": int(to_lane_raw),
                "to_lane_inferred": int(to_lane),
                "t0_frame": t0,
                "risk_thw": bool(e["risk_thw"]),
                "decision_lat_v_abs_max": decision_lat_v,
                "decision_speed_delta_median_candidate": decision_speed_delta,
                "decision_candidate_ratio": float(candidate_ratio),
                "decision_frames": int(pre_frames),
                "decision_frames_with_cutter": int(frames_with_cutter),
                "candidate_frames_with_candidate": int(frames_with_candidate),
                "candidate_id_mode": int(candidate_id_mode),
                "candidate_mode_ratio": float(candidate_mode_ratio),
                "candidate_id_last": int(candidate_id_last),
            }

            if oracle_col is not None:
                oracle = int(e[oracle_col]) if not pd.isna(e[oracle_col]) else 0
                row["oracle_follower_id"] = oracle
                row["match_mode"] = bool(oracle != 0 and candidate_id_mode == oracle)
                row["match_last"] = bool(oracle != 0 and candidate_id_last == oracle)
                row["match_ratio"] = (
                    float(np.mean([cid == oracle for cid in cand_ids])) if (oracle != 0 and cand_ids) else 0.0
                )

            rows_out.append(row)

    feat = pd.DataFrame(rows_out)
    out_feat = out_dir / "realistic_candidate_features.csv"
    feat.to_csv(out_feat, index=False)

    print("== Step 13B: Realistic candidate-follower warning ==")
    print("Events:", len(feat))
    print("Saved features:", out_feat)

    if not feat.empty:
        print("\nCandidate availability (decision_candidate_ratio):")
        desc = feat["decision_candidate_ratio"].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        )
        print(desc.to_string())

        print("\nCandidate stability (candidate_mode_ratio):")
        desc2 = feat["candidate_mode_ratio"].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        )
        print(desc2.to_string())

        # Optional oracle match summary
        if oracle_col is not None and "match_mode" in feat.columns:
            print("\nCandidate vs oracle follower match:")
            print("match_mode rate:", float(feat["match_mode"].mean()))
            print("match_last rate:", float(feat["match_last"].mean()))
            print("match_ratio median:", float(feat["match_ratio"].median()))

            by_rec = (
                feat.groupby("recording_id", sort=False)
                .agg(
                    events=("cutter_id", "count"),
                    cand_ratio_median=("decision_candidate_ratio", "median"),
                    mode_ratio_median=("candidate_mode_ratio", "median"),
                    match_mode_rate=("match_mode", "mean"),
                    match_last_rate=("match_last", "mean"),
                    match_ratio_median=("match_ratio", "median"),
                )
                .reset_index()
            )
            out_match = out_dir / "candidate_match_summary_by_recording.csv"
            by_rec.to_csv(out_match, index=False)
            print("Saved match summary:", out_match)

    # Drop rows with missing lat feature (rare, but safe)
    use = feat.dropna(subset=["decision_lat_v_abs_max"]).copy()

    y = use["risk_thw"].to_numpy(dtype=bool)
    lat = use["decision_lat_v_abs_max"].to_numpy(dtype=float)
    spd = use["decision_speed_delta_median_candidate"].to_numpy(dtype=float)

    # 1) Evaluate Step 11 thresholds as a reference point
    pred_baseline = (lat >= float(args.baseline_lat_thr)) & (spd >= float(args.baseline_spd_thr))
    m_baseline = compute_metrics(y, pred_baseline)

    print("\nReference evaluation (Step 11 thresholds):")
    print(f"  lat_thr={args.baseline_lat_thr:.2f}, spd_thr={args.baseline_spd_thr:.2f}")
    print(
        f"  precision={m_baseline.precision:.3f} recall={m_baseline.recall:.3f} f1={m_baseline.f1:.3f} "
        f"(tp={m_baseline.tp} fp={m_baseline.fp} fn={m_baseline.fn} tn={m_baseline.tn})"
    )

    # 2) Global threshold search on realistic features
    lat_grid = np.arange(args.lat_min, args.lat_max + 1e-9, args.lat_step, dtype=float)
    spd_grid = np.arange(args.spd_min, args.spd_max + 1e-9, args.spd_step, dtype=float)

    best_lat, best_spd, best_m = grid_search_best_thresholds(lat, spd, y, lat_grid, spd_grid)

    print("\nGlobal best rule (realistic features):")
    print(f"  decision_lat_v_abs_max >= {best_lat:.2f} AND decision_speed_delta_median_candidate >= {best_spd:.2f}")
    print(
        f"  precision={best_m.precision:.3f} recall={best_m.recall:.3f} f1={best_m.f1:.3f} "
        f"(tp={best_m.tp} fp={best_m.fp} fn={best_m.fn} tn={best_m.tn})"
    )

    # 3) Leave-one-recording-out evaluation (thresholds learned on train set)
    recs = sorted(use["recording_id"].astype(str).unique().tolist())
    loo_rows = []

    micro_tp = micro_fp = micro_fn = micro_tn = 0

    for _, _, r in iter_with_progress(
        recs,
        label="Step 13B LOO folds",
        item_name="heldout_recording",
    ):
        train = use[use["recording_id"].astype(str) != r]
        test = use[use["recording_id"].astype(str) == r]

        y_tr = train["risk_thw"].to_numpy(dtype=bool)
        lat_tr = train["decision_lat_v_abs_max"].to_numpy(dtype=float)
        spd_tr = train["decision_speed_delta_median_candidate"].to_numpy(dtype=float)

        y_te = test["risk_thw"].to_numpy(dtype=bool)
        lat_te = test["decision_lat_v_abs_max"].to_numpy(dtype=float)
        spd_te = test["decision_speed_delta_median_candidate"].to_numpy(dtype=float)

        lat_thr, spd_thr, _ = grid_search_best_thresholds(lat_tr, spd_tr, y_tr, lat_grid, spd_grid)

        pred = (lat_te >= lat_thr) & (spd_te >= spd_thr)
        m = compute_metrics(y_te, pred)

        micro_tp += m.tp
        micro_fp += m.fp
        micro_fn += m.fn
        micro_tn += m.tn

        loo_rows.append(
            {
                "heldout_recording": r,
                "lat_thr": float(lat_thr),
                "spd_thr": float(spd_thr),
                "precision": float(m.precision),
                "recall": float(m.recall),
                "f1": float(m.f1),
                "tp": int(m.tp),
                "fp": int(m.fp),
                "fn": int(m.fn),
                "tn": int(m.tn),
                "n_test": int(len(y_te)),
            }
        )

    loo = pd.DataFrame(loo_rows)
    out_loo = out_dir / "leave_one_recording_out_realistic.csv"
    loo.to_csv(out_loo, index=False)

    macro = {
        "precision": float(loo["precision"].mean()),
        "recall": float(loo["recall"].mean()),
        "f1": float(loo["f1"].mean()),
    }

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    print("\nLeave-one-recording-out (realistic features):")
    print(loo.to_string(index=False))
    print("\nMacro averages:", macro)
    print(
        "Micro totals:",
        {
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "tn": micro_tn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
    )

    print("\nSaved LOOCV results:", out_loo)


if __name__ == "__main__":
    main()
