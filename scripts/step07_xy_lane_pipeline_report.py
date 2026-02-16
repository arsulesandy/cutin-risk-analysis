from __future__ import annotations

import argparse
from pathlib import Path
import statistics as stats

import pandas as pd

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.reconstruction.lanes import parse_lane_markings, infer_lane_index
from cutin_risk.reconstruction.neighbors import reconstruct_same_lane_neighbors, NeighborReconstructionOptions
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.detection.cutin import detect_cutins, CutInOptions


def _norm_neighbor(s: pd.Series) -> pd.Series:
    out = s.fillna(0).astype(int)
    out = out.where(~out.isin({-1, 0}), other=0)
    return out.astype(int)


def _accuracy(truth: pd.Series, pred: pd.Series) -> dict[str, float]:
    truth = truth.astype(int)
    pred = pred.astype(int)
    overall = float((truth == pred).mean())

    mask = truth != 0
    cond = float((truth[mask] == pred[mask]).mean()) if mask.any() else float("nan")
    return {
        "overall_accuracy": overall,
        "accuracy_when_truth_has_neighbor": cond,
        "rows": float(len(truth)),
        "rows_with_neighbor": float(mask.sum()),
    }


def _match_cutins(true_events, pred_events, *, frame_tolerance: int = 10) -> dict[str, float]:
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

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

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
    parser = argparse.ArgumentParser(description="Step 7: infer lanes from y, reconstruct neighbors, compare cut-ins.")
    parser.add_argument("--dataset-root", type=str, default="/Users/sandeep/IdeaProjects/cutin-risk-analysis/data/raw/highD-dataset-v1.0/data")
    parser.add_argument("--recording-id", type=str, default="01")
    parser.add_argument("--frame-tolerance", type=int, default=10)
    args = parser.parse_args()

    rec = load_highd_recording(Path(args.dataset_root), args.recording_id)
    df = build_tracking_table(rec)

    # --- Infer lane index from y using lane markings ---
    markings = parse_lane_markings(rec.recording_meta)
    lane_idx = infer_lane_index(df, markings)
    df = df.join(lane_idx)

    print("== Step 7: XY-lane pipeline report ==")
    print("Recording:", rec.recording_id)
    print("Rows:", len(df))
    print("Vehicles:", df["id"].nunique())

    # --- Neighbor reconstruction using inferred lane index ---
    pred_neighbors = reconstruct_same_lane_neighbors(
        df,
        options=NeighborReconstructionOptions(
            lane_col="laneIndex_xy",
            out_preceding_col="precedingId_xy_lane",
            out_following_col="followingId_xy_lane",
            ignore_lane_ids=(0,),
            no_neighbor_id=0,
        ),
    )
    df = df.join(pred_neighbors)

    print("\nSame-lane precedingId accuracy (inferred lanes):")
    print(_accuracy(_norm_neighbor(df["precedingId"]), _norm_neighbor(df["precedingId_xy_lane"])))

    print("\nSame-lane followingId accuracy (inferred lanes):")
    print(_accuracy(_norm_neighbor(df["followingId"]), _norm_neighbor(df["followingId_xy_lane"])))

    # --- Compare cut-ins: oracle (highD columns) vs XY-lane reconstructed columns ---
    lane_changes_oracle = detect_lane_changes(df, options=LaneChangeOptions())
    cutins_oracle = detect_cutins(df, lane_changes_oracle, options=CutInOptions())

    lane_changes_xy = detect_lane_changes(df, options=LaneChangeOptions(lane_col="laneIndex_xy"))
    cutins_xy = detect_cutins(
        df,
        lane_changes_xy,
        options=CutInOptions(
            lane_col="laneIndex_xy",
            following_col="followingId_xy_lane",
            preceding_col="precedingId_xy_lane",
            search_window_frames=50,
            min_relation_frames=10,
        ),
    )

    print("\nCut-in detection comparison:")
    print("Lane changes (oracle):", len(lane_changes_oracle))
    print("Cut-ins (oracle):", len(cutins_oracle))
    print("Lane changes (xy-lane):", len(lane_changes_xy))
    print("Cut-ins (xy-lane):", len(cutins_xy))

    metrics = _match_cutins(cutins_oracle, cutins_xy, frame_tolerance=int(args.frame_tolerance))
    print("\nCut-in matching metrics (oracle vs xy-lane):")
    print(metrics)


if __name__ == "__main__":
    main()
