"""Step 06: evaluate geometry-based neighbor reconstruction against dataset neighbors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import statistics as stats
from pathlib import Path

import pandas as pd
from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.io.step_reports import step_reports_dir, write_step_markdown
from cutin_risk.paths import dataset_root_path
from cutin_risk.reconstruction.neighbors import reconstruct_same_lane_neighbors
from cutin_risk.thesis_config import thesis_int

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


def _all_recording_ids(root: Path) -> list[str]:
    pattern = f"*_{RECORDING_META_SUFFIX}.csv"
    return sorted(p.name.split("_", 1)[0] for p in root.glob(pattern))


def _evaluate_recording(
    dataset_root: Path,
    recording_id: str,
    *,
    frame_tolerance: int,
    mismatch_examples: int,
) -> tuple[dict[str, float | str], list[str]]:
    rec = load_highd_recording(dataset_root, recording_id)
    df = build_tracking_table(rec)

    # Reconstruct neighbors from geometry (baseline: same-lane preceding/following).
    pred_neighbors = reconstruct_same_lane_neighbors(df)
    df = df.join(pred_neighbors)

    truth_pre = _norm_neighbor(df["precedingId"])
    pred_pre = _norm_neighbor(df["precedingId_xy"])
    truth_fol = _norm_neighbor(df["followingId"])
    pred_fol = _norm_neighbor(df["followingId_xy"])
    pre_acc = _accuracy(truth_pre, pred_pre)
    fol_acc = _accuracy(truth_fol, pred_fol)

    mism_pre = df.loc[
        truth_pre != pred_pre, ["frame", "id", "laneId", "drivingDirection", "x", "precedingId", "precedingId_xy"]]
    mism_fol = df.loc[
        truth_fol != pred_fol, ["frame", "id", "laneId", "drivingDirection", "x", "followingId", "followingId_xy"]]

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
    metrics = _match_cutins(cutins_oracle, cutins_xy, frame_tolerance=frame_tolerance)

    console_lines = [
        f"\n== Recording {rec.recording_id} ==",
        f"Rows: {len(df)}",
        f"Vehicles: {df['id'].nunique()}",
        "Same-lane precedingId accuracy:",
        str(pre_acc),
        "Same-lane followingId accuracy:",
        str(fol_acc),
        f"precedingId mismatches: {len(mism_pre)}",
    ]
    if len(mism_pre) > 0 and mismatch_examples > 0:
        console_lines.append(mism_pre.head(mismatch_examples).to_string(index=False))

    console_lines.append(f"followingId mismatches: {len(mism_fol)}")
    if len(mism_fol) > 0 and mismatch_examples > 0:
        console_lines.append(mism_fol.head(mismatch_examples).to_string(index=False))

    console_lines.extend(
        [
            "Cut-in detection comparison:",
            f"Lane changes: {len(lane_changes)}",
            f"Cut-ins (oracle): {len(cutins_oracle)}",
            f"Cut-ins (reconstructed neighbors): {len(cutins_xy)}",
            "Cut-in matching metrics:",
            str(metrics),
        ]
    )

    row: dict[str, float | str] = {
        "recording_id": rec.recording_id,
        "rows": float(len(df)),
        "vehicles": float(df["id"].nunique()),
        "preceding_overall_accuracy": pre_acc["overall_accuracy"],
        "preceding_accuracy_when_truth_has_neighbor": pre_acc["accuracy_when_truth_has_neighbor"],
        "following_overall_accuracy": fol_acc["overall_accuracy"],
        "following_accuracy_when_truth_has_neighbor": fol_acc["accuracy_when_truth_has_neighbor"],
        "cutin_true_events": metrics["true_events"],
        "cutin_pred_events": metrics["pred_events"],
        "cutin_tp": metrics["tp"],
        "cutin_fp": metrics["fp"],
        "cutin_fn": metrics["fn"],
        "cutin_precision": metrics["precision"],
        "cutin_recall": metrics["recall"],
        "cutin_f1": metrics["f1"],
    }
    return row, console_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: reconstruct neighbors from geometry and evaluate vs highD.")
    parser.add_argument("--dataset-root", type=str, default=str(dataset_root_path()))
    parser.add_argument(
        "--recording-id",
        type=str,
        default=None,
        help="Optional single recording ID to process. If omitted, process all recordings.",
    )
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

    print("== Step 6: Neighbor reconstruction report ==")
    dataset_root = Path(args.dataset_root)
    if args.recording_id:
        recording_ids = [str(args.recording_id)]
    else:
        recording_ids = _all_recording_ids(dataset_root)

    if not recording_ids:
        raise FileNotFoundError(
            f"No recording metadata files found under {dataset_root} matching *_recordingMeta.csv"
        )

    rows: list[dict[str, float | str]] = []
    for rid in recording_ids:
        row, console_lines = _evaluate_recording(
            dataset_root,
            rid,
            frame_tolerance=int(args.frame_tolerance),
            mismatch_examples=int(args.mismatch_examples),
        )
        rows.append(row)
        for line in console_lines:
            print(line)

    report_dir = step_reports_dir(STEP_NUMBER)
    metrics_csv = report_dir / "neighbor_reconstruction_metrics_by_recording.csv"
    metrics_df = pd.DataFrame(rows).sort_values("recording_id").reset_index(drop=True)
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "mean_preceding_overall_accuracy": float(metrics_df["preceding_overall_accuracy"].mean()),
        "mean_following_overall_accuracy": float(metrics_df["following_overall_accuracy"].mean()),
        "mean_cutin_precision": float(metrics_df["cutin_precision"].mean()),
        "mean_cutin_recall": float(metrics_df["cutin_recall"].mean()),
        "mean_cutin_f1": float(metrics_df["cutin_f1"].mean()),
    }
    summary_csv = report_dir / "neighbor_reconstruction_metrics_summary.csv"
    pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()]).to_csv(summary_csv, index=False)

    details_md = write_step_markdown(
        STEP_NUMBER,
        "neighbor_reconstruction_details.md",
        [
            "# Step 06 Neighbor Reconstruction Report",
            "",
            f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Recordings processed: `{len(metrics_df)}`",
            f"- Dataset root: `{Path(args.dataset_root).resolve()}`",
            "",
            "## Mean Accuracy Across Processed Recordings",
            f"- Same-lane preceding overall: `{summary['mean_preceding_overall_accuracy']:.6f}`",
            f"- Same-lane following overall: `{summary['mean_following_overall_accuracy']:.6f}`",
            "",
            "## Mean Cut-in Matching Metrics",
            f"- Precision: `{summary['mean_cutin_precision']:.6f}`",
            f"- Recall: `{summary['mean_cutin_recall']:.6f}`",
            f"- F1: `{summary['mean_cutin_f1']:.6f}`",
            "",
            f"- Metrics CSV: `{metrics_csv}`",
            f"- Summary CSV: `{summary_csv}`",
        ],
    )
    print("\nSaved:", metrics_csv)
    print("Saved:", summary_csv)
    print("Saved:", details_md)


if __name__ == "__main__":
    main()
