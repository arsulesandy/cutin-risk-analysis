"""Step 06: evaluate geometry-based neighbor reconstruction against dataset neighbors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import statistics as stats
from pathlib import Path

import pandas as pd
from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.io.step_reports import step_reports_dir, write_step_markdown
from cutin_risk.paths import dataset_root_path
from cutin_risk.reconstruction.neighbors import reconstruct_same_lane_neighbors
from cutin_risk.thesis_config import thesis_int, thesis_str

STEP_NUMBER = 6


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
    parser.add_argument("--recording-id", type=str, default=thesis_str("step06.recording_id", "01"))
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=thesis_int("step06.frame_tolerance", 10, min_value=0),
    )
    parser.add_argument(
        "--mismatch-examples",
        type=int,
        default=thesis_int("step06.mismatch_examples", 10, min_value=0),
    )
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
    pre_acc = _accuracy(truth_pre, pred_pre)
    print(pre_acc)

    print("\nSame-lane followingId accuracy:")
    fol_acc = _accuracy(truth_fol, pred_fol)
    print(fol_acc)

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
        options=LaneChangeOptions(),
    )

    cutins_oracle = detect_cutins(df, lane_changes, options=CutInOptions())
    cutins_xy = detect_cutins(
        df,
        lane_changes,
        options=CutInOptions(
            following_col="followingId_xy",
            preceding_col="precedingId_xy",
        ),
    )

    print("\nCut-in detection comparison:")
    print("Lane changes:", len(lane_changes))
    print("Cut-ins (oracle):", len(cutins_oracle))
    print("Cut-ins (reconstructed neighbors):", len(cutins_xy))

    metrics = _match_cutins(cutins_oracle, cutins_xy, frame_tolerance=int(args.frame_tolerance))
    print("\nCut-in matching metrics:")
    print(metrics)

    report_dir = step_reports_dir(STEP_NUMBER)
    metrics_csv = report_dir / "neighbor_reconstruction_metrics.csv"
    pd.DataFrame(
        [
            {"metric": "preceding_overall_accuracy", "value": pre_acc["overall_accuracy"]},
            {
                "metric": "preceding_accuracy_when_truth_has_neighbor",
                "value": pre_acc["accuracy_when_truth_has_neighbor"],
            },
            {"metric": "following_overall_accuracy", "value": fol_acc["overall_accuracy"]},
            {
                "metric": "following_accuracy_when_truth_has_neighbor",
                "value": fol_acc["accuracy_when_truth_has_neighbor"],
            },
            {"metric": "cutin_true_events", "value": metrics["true_events"]},
            {"metric": "cutin_pred_events", "value": metrics["pred_events"]},
            {"metric": "cutin_tp", "value": metrics["tp"]},
            {"metric": "cutin_fp", "value": metrics["fp"]},
            {"metric": "cutin_fn", "value": metrics["fn"]},
            {"metric": "cutin_precision", "value": metrics["precision"]},
            {"metric": "cutin_recall", "value": metrics["recall"]},
            {"metric": "cutin_f1", "value": metrics["f1"]},
        ]
    ).to_csv(metrics_csv, index=False)

    details_md = write_step_markdown(
        STEP_NUMBER,
        "neighbor_reconstruction_details.md",
        [
            "# Step 06 Neighbor Reconstruction Report",
            "",
            f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Recording: `{rec.recording_id}`",
            f"- Dataset root: `{Path(args.dataset_root).resolve()}`",
            f"- Rows: `{len(df)}`",
            f"- Vehicles: `{int(df['id'].nunique())}`",
            "",
            "## Accuracy",
            f"- Same-lane preceding overall: `{pre_acc['overall_accuracy']:.6f}`",
            f"- Same-lane preceding (truth has neighbor): `{pre_acc['accuracy_when_truth_has_neighbor']:.6f}`",
            f"- Same-lane following overall: `{fol_acc['overall_accuracy']:.6f}`",
            f"- Same-lane following (truth has neighbor): `{fol_acc['accuracy_when_truth_has_neighbor']:.6f}`",
            "",
            "## Cut-in Matching",
            f"- Lane changes: `{len(lane_changes)}`",
            f"- Cut-ins oracle: `{len(cutins_oracle)}`",
            f"- Cut-ins reconstructed: `{len(cutins_xy)}`",
            f"- Precision: `{metrics['precision']:.6f}`",
            f"- Recall: `{metrics['recall']:.6f}`",
            f"- F1: `{metrics['f1']:.6f}`",
            "",
            f"- Metrics CSV: `{metrics_csv}`",
        ],
    )
    print("\nSaved:", metrics_csv)
    print("Saved:", details_md)


if __name__ == "__main__":
    main()
