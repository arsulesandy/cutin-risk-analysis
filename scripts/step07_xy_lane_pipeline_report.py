"""Step 07: run detection pipeline with inferred lanes and reconstructed neighbors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics as stats

import pandas as pd

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.reconstruction.lanes import parse_lane_markings, infer_lane_index
from cutin_risk.reconstruction.neighbors import reconstruct_same_lane_neighbors, NeighborReconstructionOptions
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
from cutin_risk.io.step_reports import step_reports_dir, write_step_markdown
from cutin_risk.paths import dataset_root_path
from cutin_risk.thesis_config import thesis_int

STEP_NUMBER = 7


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


def _all_recording_ids(root: Path) -> list[str]:
    pattern = f"*_{RECORDING_META_SUFFIX}.csv"
    return sorted(p.name.split("_", 1)[0] for p in root.glob(pattern))


def _evaluate_recording(
    dataset_root: Path,
    recording_id: str,
    *,
    frame_tolerance: int,
) -> tuple[dict[str, float | str], list[str]]:
    rec = load_highd_recording(dataset_root, recording_id)
    df = build_tracking_table(rec)

    markings = parse_lane_markings(rec.recording_meta)
    lane_idx = infer_lane_index(df, markings)
    df = df.join(lane_idx)

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

    pre_acc = _accuracy(_norm_neighbor(df["precedingId"]), _norm_neighbor(df["precedingId_xy_lane"]))
    fol_acc = _accuracy(_norm_neighbor(df["followingId"]), _norm_neighbor(df["followingId_xy_lane"]))

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
        ),
    )
    metrics = _match_cutins(cutins_oracle, cutins_xy, frame_tolerance=frame_tolerance)

    console_lines = [
        f"\n== Recording {rec.recording_id} ==",
        f"Rows: {len(df)}",
        f"Vehicles: {df['id'].nunique()}",
        "Same-lane precedingId accuracy (inferred lanes):",
        str(pre_acc),
        "Same-lane followingId accuracy (inferred lanes):",
        str(fol_acc),
        "Cut-in detection comparison:",
        f"Lane changes (oracle): {len(lane_changes_oracle)}",
        f"Cut-ins (oracle): {len(cutins_oracle)}",
        f"Lane changes (xy-lane): {len(lane_changes_xy)}",
        f"Cut-ins (xy-lane): {len(cutins_xy)}",
        "Cut-in matching metrics (oracle vs xy-lane):",
        str(metrics),
    ]

    row: dict[str, float | str] = {
        "recording_id": rec.recording_id,
        "rows": float(len(df)),
        "vehicles": float(df["id"].nunique()),
        "preceding_overall_accuracy": pre_acc["overall_accuracy"],
        "preceding_accuracy_when_truth_has_neighbor": pre_acc["accuracy_when_truth_has_neighbor"],
        "following_overall_accuracy": fol_acc["overall_accuracy"],
        "following_accuracy_when_truth_has_neighbor": fol_acc["accuracy_when_truth_has_neighbor"],
        "lane_changes_oracle": float(len(lane_changes_oracle)),
        "lane_changes_xy": float(len(lane_changes_xy)),
        "cutins_oracle": float(len(cutins_oracle)),
        "cutins_xy": float(len(cutins_xy)),
        "cutin_precision": metrics["precision"],
        "cutin_recall": metrics["recall"],
        "cutin_f1": metrics["f1"],
    }
    return row, console_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: infer lanes from y, reconstruct neighbors, compare cut-ins.")
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
        default=thesis_int("step07.frame_tolerance", 10, min_value=0),
    )
    args = parser.parse_args()

    print("== Step 7: XY-lane pipeline report ==")
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
        )
        rows.append(row)
        for line in console_lines:
            print(line)

    report_dir = step_reports_dir(STEP_NUMBER)
    metrics_csv = report_dir / "xy_lane_pipeline_metrics_by_recording.csv"
    metrics_df = pd.DataFrame(rows).sort_values("recording_id").reset_index(drop=True)
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "mean_preceding_overall_accuracy": float(metrics_df["preceding_overall_accuracy"].mean()),
        "mean_following_overall_accuracy": float(metrics_df["following_overall_accuracy"].mean()),
        "mean_cutin_precision": float(metrics_df["cutin_precision"].mean()),
        "mean_cutin_recall": float(metrics_df["cutin_recall"].mean()),
        "mean_cutin_f1": float(metrics_df["cutin_f1"].mean()),
    }
    summary_csv = report_dir / "xy_lane_pipeline_metrics_summary.csv"
    pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()]).to_csv(summary_csv, index=False)

    details_md = write_step_markdown(
        STEP_NUMBER,
        "xy_lane_pipeline_details.md",
        [
            "# Step 07 XY-Lane Pipeline Report",
            "",
            f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Recordings processed: `{len(metrics_df)}`",
            f"- Dataset root: `{Path(args.dataset_root).resolve()}`",
            "",
            "## Mean Neighbor Accuracy (Inferred Lanes)",
            f"- Same-lane preceding overall: `{summary['mean_preceding_overall_accuracy']:.6f}`",
            f"- Same-lane following overall: `{summary['mean_following_overall_accuracy']:.6f}`",
            "",
            "## Mean Cut-in Matching (Oracle vs XY-Lane)",
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
