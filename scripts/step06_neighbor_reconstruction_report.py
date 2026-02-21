from __future__ import annotations

import argparse
import statistics as stats
from pathlib import Path

import pandas as pd
from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.paths import dataset_root_path
from cutin_risk.reconstruction.neighbors import reconstruct_same_lane_neighbors


def _norm_neighbor(s: pd.Series, *, no_neighbor_ids: tuple[int, ...] = (0, -1)) -> pd.Series:
    # Normalize neighbor IDs so 0/-1/NaN are treated consistently as "no neighbor" = 0
    out = s.copy()
    out = out.fillna(0)
    out = out.astype(int)
    out = out.where(~out.isin(set(no_neighbor_ids)), other=0)
    return out.astype(int)


def _accuracy(truth: pd.Series, pred: pd.Series) -> dict[str, float]:
    truth = truth.astype(int)
    pred = pred.astype(int)

    overall = float((truth == pred).mean())

    mask_has_neighbor = truth != 0
    if mask_has_neighbor.any():
        cond = float((truth[mask_has_neighbor] == pred[mask_has_neighbor]).mean())
    else:
        cond = float("nan")

    return {
        "overall_accuracy": overall,
        "accuracy_when_truth_has_neighbor": cond,
        "rows": float(len(truth)),
        "rows_with_neighbor": float(mask_has_neighbor.sum()),
    }


def _match_cutins(true_events, pred_events, *, frame_tolerance: int = 10) -> dict[str, float]:
    # Greedy matching by (cutter_id, follower_id) and start_frame tolerance.
    true_by_key: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, e in enumerate(true_events):
        key = (int(e.cutter_id), int(e.follower_id))
        true_by_key.setdefault(key, []).append((int(e.relation_start_frame), i))

    matched_true = set()
    deltas: list[int] = []

    tp = 0
    for pe in pred_events:
        key = (int(pe.cutter_id), int(pe.follower_id))
        candidates = true_by_key.get(key, [])
        best = None

        for true_start, idx in candidates:
            if idx in matched_true:
                continue
            d = int(pe.relation_start_frame) - int(true_start)
            if abs(d) <= frame_tolerance:
                if best is None or abs(d) < abs(best[0]):
                    best = (d, idx)

        if best is not None:
            d, idx = best
            matched_true.add(idx)
            deltas.append(d)
            tp += 1

    fp = len(pred_events) - tp
    fn = len(true_events) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    out = {
        "true_events": float(len(true_events)),
        "pred_events": float(len(pred_events)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if deltas:
        out["start_frame_delta_median"] = float(stats.median(deltas))
        out["start_frame_delta_min"] = float(min(deltas))
        out["start_frame_delta_max"] = float(max(deltas))
    else:
        out["start_frame_delta_median"] = float("nan")
        out["start_frame_delta_min"] = float("nan")
        out["start_frame_delta_max"] = float("nan")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: reconstruct neighbors from geometry and evaluate vs highD.")
    parser.add_argument("--dataset-root", type=str, default=str(dataset_root_path()))
    parser.add_argument("--recording-id", type=str, default="01")
    parser.add_argument("--frame-tolerance", type=int, default=10)
    parser.add_argument("--mismatch-examples", type=int, default=10)
    args = parser.parse_args()

    rec = load_highd_recording(Path(args.dataset_root), args.recording_id)
    df = build_tracking_table(rec)

    # Reconstruct neighbors from geometry (baseline: same-lane preceding/following).
    pred_neighbors = reconstruct_same_lane_neighbors(df)
    df = df.join(pred_neighbors)

    # Evaluate neighbor reconstruction accuracy against dataset columns.
    print("== Step 6: Neighbor reconstruction report ==")
    print("Recording:", rec.recording_id)
    print("Rows:", len(df))
    print("Vehicles:", df["id"].nunique())

    truth_pre = _norm_neighbor(df["precedingId"])
    pred_pre = _norm_neighbor(df["precedingId_xy"])
    truth_fol = _norm_neighbor(df["followingId"])
    pred_fol = _norm_neighbor(df["followingId_xy"])

    print("\nSame-lane precedingId accuracy:")
    print(_accuracy(truth_pre, pred_pre))

    print("\nSame-lane followingId accuracy:")
    print(_accuracy(truth_fol, pred_fol))

    # Show a few mismatches to understand edge cases
    mism_pre = df.loc[
        truth_pre != pred_pre, ["frame", "id", "laneId", "drivingDirection", "x", "precedingId", "precedingId_xy"]]
    mism_fol = df.loc[
        truth_fol != pred_fol, ["frame", "id", "laneId", "drivingDirection", "x", "followingId", "followingId_xy"]]

    print(f"\nprecedingId mismatches: {len(mism_pre)}")
    if len(mism_pre) > 0:
        print(mism_pre.head(int(args.mismatch_examples)).to_string(index=False))

    print(f"\nfollowingId mismatches: {len(mism_fol)}")
    if len(mism_fol) > 0:
        print(mism_fol.head(int(args.mismatch_examples)).to_string(index=False))

    # Now evaluate cut-in detection using:
    # - oracle neighbors (highD columns)
    # - reconstructed neighbors (our *_xy columns)
    lane_changes = detect_lane_changes(
        df,
        options=LaneChangeOptions(min_stable_before_frames=25, min_stable_after_frames=25),
    )

    cutins_oracle = detect_cutins(df, lane_changes, options=CutInOptions())
    cutins_xy = detect_cutins(
        df,
        lane_changes,
        options=CutInOptions(
            following_col="followingId_xy",
            preceding_col="precedingId_xy",
            search_window_frames=50,
            min_relation_frames=10,
        ),
    )

    print("\nCut-in detection comparison:")
    print("Lane changes:", len(lane_changes))
    print("Cut-ins (oracle):", len(cutins_oracle))
    print("Cut-ins (reconstructed neighbors):", len(cutins_xy))

    metrics = _match_cutins(cutins_oracle, cutins_xy, frame_tolerance=int(args.frame_tolerance))
    print("\nCut-in matching metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
