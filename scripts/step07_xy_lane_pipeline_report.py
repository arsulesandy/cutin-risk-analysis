"""Step 07: run detection pipeline with inferred lanes and reconstructed neighbors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics as stats

import numpy as np
import pandas as pd

from cutin_risk.encoding.sfc_binary import decode_grid_4x4_bits
from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.reconstruction.lanes import parse_lane_markings, infer_lane_index
from cutin_risk.reconstruction.neighbors import reconstruct_same_lane_neighbors, NeighborReconstructionOptions
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
from cutin_risk.io.step_reports import step_reports_dir, write_step_markdown
from cutin_risk.paths import dataset_root_path, step14_codes_csv_path
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


def _normalize_recording_id(v: object) -> str:
    s = str(v).strip()
    return f"{int(s):02d}" if s.isdigit() else s


def _parse_neighbor_id(raw_value: object) -> int:
    try:
        value = int(raw_value)
    except Exception:
        try:
            value = int(float(raw_value))
        except Exception:
            return 0
    return value if value > 0 else 0


def _build_highd_reference_matrix(row: dict[str, object]) -> np.ndarray:
    g = np.zeros((3, 3), dtype=np.uint8)
    g[1, 1] = 1
    id_columns = (
        ("leftPrecedingId", "leftAlongsideId", "leftFollowingId"),
        ("precedingId", None, "followingId"),
        ("rightPrecedingId", "rightAlongsideId", "rightFollowingId"),
    )
    for col, (preceding_col, alongside_col, following_col) in enumerate(id_columns):
        for r, column_name in ((0, preceding_col), (1, alongside_col), (2, following_col)):
            if column_name is None:
                continue
            if _parse_neighbor_id(row.get(column_name, 0)) != 0:
                g[r, col] = 1
    return g


def _decode_code_to_3x3(code: int, order: str, cache: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
    key = (str(order), int(code))
    cached = cache.get(key)
    if cached is not None:
        return cached
    grid_4x4 = decode_grid_4x4_bits(int(code), order=str(order))
    grid_3x3 = np.asarray(grid_4x4[:3, :3], dtype=np.uint8)
    cache[key] = grid_3x3
    return grid_3x3


def _load_sfc_codes_table(sfc_codes_csv: str | None) -> pd.DataFrame:
    expected_cols = ["event_id", "recording_id", "cutter_id", "stage", "frame", "code", "sfc_order"]
    if not sfc_codes_csv:
        return pd.DataFrame(columns=expected_cols)
    path = Path(sfc_codes_csv)
    if not path.exists():
        return pd.DataFrame(columns=expected_cols)

    try:
        df = pd.read_csv(path, usecols=expected_cols)
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=expected_cols)

    required = {"recording_id", "cutter_id", "stage", "frame", "code"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame(columns=expected_cols)

    out = df.copy()
    out["recording_id"] = out["recording_id"].apply(_normalize_recording_id)
    out["cutter_id"] = pd.to_numeric(out["cutter_id"], errors="coerce").astype("Int64")
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce").astype("Int64")
    out["code"] = pd.to_numeric(out["code"], errors="coerce").astype("Int64")
    out["stage"] = out["stage"].astype(str)
    if "event_id" not in out.columns:
        out["event_id"] = out.groupby(["recording_id", "cutter_id", "stage"], sort=False).ngroup() + 1
    out["event_id"] = pd.to_numeric(out["event_id"], errors="coerce").astype("Int64")
    if "sfc_order" not in out.columns:
        out["sfc_order"] = "hilbert"
    out["sfc_order"] = out["sfc_order"].fillna("hilbert").astype(str)
    out = out.dropna(subset=["event_id", "cutter_id", "frame", "code"]).copy()
    return out.reset_index(drop=True)


def _evaluate_matrix_verification(
    df: pd.DataFrame,
    sfc_codes_recording: pd.DataFrame,
    *,
    sfc_codes_canonical: bool,
) -> dict[str, float]:
    out = {
        "matrix_rows_available": float(len(sfc_codes_recording)),
        "matrix_rows_verified": 0.0,
        "matrix_matches": 0.0,
        "matrix_mismatches": 0.0,
        "matrix_row_match_rate": float("nan"),
        "matrix_stage_window_any_match_rate": float("nan"),
        "matrix_stage_window_all_match_rate": float("nan"),
        "matrix_event_any_match_rate": float("nan"),
        "matrix_event_all_match_rate": float("nan"),
        "matrix_intention_row_match_rate": float("nan"),
        "matrix_decision_row_match_rate": float("nan"),
        "matrix_execution_row_match_rate": float("nan"),
    }
    if sfc_codes_recording.empty:
        return out

    track_cols = [
        "id",
        "frame",
        "drivingDirection",
        "leftPrecedingId",
        "leftAlongsideId",
        "leftFollowingId",
        "precedingId",
        "followingId",
        "rightPrecedingId",
        "rightAlongsideId",
        "rightFollowingId",
    ]
    if any(col not in df.columns for col in track_cols):
        return out

    df_track = df[track_cols].copy()
    df_track["id"] = pd.to_numeric(df_track["id"], errors="coerce").astype("Int64")
    df_track["frame"] = pd.to_numeric(df_track["frame"], errors="coerce").astype("Int64")
    df_track = df_track.dropna(subset=["id", "frame"]).copy()

    eval_codes = sfc_codes_recording[["event_id", "cutter_id", "stage", "frame", "code", "sfc_order"]].copy()
    merged = eval_codes.merge(df_track, left_on=["cutter_id", "frame"], right_on=["id", "frame"], how="inner")
    if merged.empty:
        return out

    decode_cache: dict[tuple[str, int], np.ndarray] = {}
    flags: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        order = str(getattr(row, "sfc_order", "hilbert") or "hilbert")
        code_value = getattr(row, "code", None)
        if code_value is None:
            continue
        try:
            derived = np.array(_decode_code_to_3x3(int(code_value), order, decode_cache), copy=True)
        except Exception:
            continue
        if not bool(sfc_codes_canonical):
            try:
                driving_direction = int(getattr(row, "drivingDirection", 0))
            except Exception:
                driving_direction = 0
            if driving_direction == 1:
                derived = np.fliplr(derived)
        reference = _build_highd_reference_matrix(row._asdict())
        flags.append(
            {
                "event_id": int(getattr(row, "event_id")),
                "stage": str(getattr(row, "stage", "")),
                "match": bool(np.array_equal(derived, reference)),
            }
        )

    if not flags:
        return out

    eval_df = pd.DataFrame(flags)
    out["matrix_rows_verified"] = float(len(eval_df))
    out["matrix_matches"] = float(eval_df["match"].sum())
    out["matrix_mismatches"] = float(len(eval_df) - int(eval_df["match"].sum()))
    out["matrix_row_match_rate"] = float(eval_df["match"].mean())

    stage_match = eval_df.groupby("stage", as_index=False)["match"].mean()
    stage_rate_map = {str(row["stage"]): float(row["match"]) for _, row in stage_match.iterrows()}
    out["matrix_intention_row_match_rate"] = stage_rate_map.get("intention", float("nan"))
    out["matrix_decision_row_match_rate"] = stage_rate_map.get("decision", float("nan"))
    out["matrix_execution_row_match_rate"] = stage_rate_map.get("execution", float("nan"))

    event_stage_df = (
        eval_df.groupby(["event_id", "stage"], as_index=False)["match"]
        .agg(stage_any="max", stage_all="min")
        .copy()
    )
    if not event_stage_df.empty:
        out["matrix_stage_window_any_match_rate"] = float(event_stage_df["stage_any"].mean())
        out["matrix_stage_window_all_match_rate"] = float(event_stage_df["stage_all"].mean())

    event_df = eval_df.groupby("event_id", as_index=False)["match"].agg(any_match="max", all_match="min")
    if not event_df.empty:
        out["matrix_event_any_match_rate"] = float(event_df["any_match"].mean())
        out["matrix_event_all_match_rate"] = float(event_df["all_match"].mean())

    return out


def _evaluate_recording(
    dataset_root: Path,
    recording_id: str,
    *,
    frame_tolerance: int,
    sfc_codes_recording: pd.DataFrame,
    sfc_codes_canonical: bool,
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
    matrix_metrics = _evaluate_matrix_verification(
        df,
        sfc_codes_recording,
        sfc_codes_canonical=sfc_codes_canonical,
    )

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
        "SFC matrix verification (Step14 codes vs highD IDs):",
        (
            "rows={rows_verified:.0f}, row_match_rate={row_rate:.2%}, "
            "event_stage_any={stage_any:.2%}, event_any={event_any:.2%}"
        ).format(
            rows_verified=float(matrix_metrics["matrix_rows_verified"]),
            row_rate=float(matrix_metrics["matrix_row_match_rate"])
            if pd.notna(matrix_metrics["matrix_row_match_rate"])
            else 0.0,
            stage_any=float(matrix_metrics["matrix_stage_window_any_match_rate"])
            if pd.notna(matrix_metrics["matrix_stage_window_any_match_rate"])
            else 0.0,
            event_any=float(matrix_metrics["matrix_event_any_match_rate"])
            if pd.notna(matrix_metrics["matrix_event_any_match_rate"])
            else 0.0,
        ),
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
    row.update(matrix_metrics)
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
    parser.add_argument(
        "--sfc-codes-csv",
        type=str,
        default=str(step14_codes_csv_path()),
        help="Step14 long-code CSV for matrix verification against highD raw IDs.",
    )
    parser.add_argument(
        "--sfc-codes-canonical",
        default=False,
        type=lambda x: (str(x).strip().lower() == "true"),
        help="Set true if provided SFC code table is already canonical (Step15A-style).",
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

    sfc_codes_table = _load_sfc_codes_table(args.sfc_codes_csv)
    if sfc_codes_table.empty:
        print("SFC matrix verification input missing/empty; matrix metrics will be n/a.")
    else:
        print(f"SFC matrix verification input loaded: {len(sfc_codes_table)} rows")

    rows: list[dict[str, float | str]] = []
    for rid in recording_ids:
        rid_norm = _normalize_recording_id(rid)
        sfc_codes_recording = (
            sfc_codes_table.loc[sfc_codes_table["recording_id"] == rid_norm].copy()
            if not sfc_codes_table.empty
            else pd.DataFrame()
        )
        row, console_lines = _evaluate_recording(
            dataset_root,
            rid,
            frame_tolerance=int(args.frame_tolerance),
            sfc_codes_recording=sfc_codes_recording,
            sfc_codes_canonical=bool(args.sfc_codes_canonical),
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
        "mean_matrix_row_match_rate": float(metrics_df["matrix_row_match_rate"].mean(skipna=True)),
        "mean_matrix_stage_window_any_match_rate": float(
            metrics_df["matrix_stage_window_any_match_rate"].mean(skipna=True)
        ),
        "mean_matrix_stage_window_all_match_rate": float(
            metrics_df["matrix_stage_window_all_match_rate"].mean(skipna=True)
        ),
        "mean_matrix_event_any_match_rate": float(metrics_df["matrix_event_any_match_rate"].mean(skipna=True)),
        "mean_matrix_event_all_match_rate": float(metrics_df["matrix_event_all_match_rate"].mean(skipna=True)),
    }
    summary_csv = report_dir / "xy_lane_pipeline_metrics_summary.csv"
    pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()]).to_csv(summary_csv, index=False)
    matrix_csv = report_dir / "xy_lane_matrix_verification_by_recording.csv"
    matrix_cols = [
        "recording_id",
        "matrix_rows_available",
        "matrix_rows_verified",
        "matrix_matches",
        "matrix_mismatches",
        "matrix_row_match_rate",
        "matrix_intention_row_match_rate",
        "matrix_decision_row_match_rate",
        "matrix_execution_row_match_rate",
        "matrix_stage_window_any_match_rate",
        "matrix_stage_window_all_match_rate",
        "matrix_event_any_match_rate",
        "matrix_event_all_match_rate",
    ]
    metrics_df[matrix_cols].to_csv(matrix_csv, index=False)

    def _fmt_pct(v: float) -> str:
        return "n/a" if pd.isna(v) else f"{100.0 * float(v):.2f}%"

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
            "## Mean SFC Matrix Verification (Step14 Codes vs highD Raw IDs)",
            f"- SFC codes CSV: `{Path(args.sfc_codes_csv).resolve()}`",
            f"- SFC codes canonical: `{bool(args.sfc_codes_canonical)}`",
            f"- Row-level match rate: `{_fmt_pct(summary['mean_matrix_row_match_rate'])}`",
            f"- Event-stage window ANY-frame match: `{_fmt_pct(summary['mean_matrix_stage_window_any_match_rate'])}`",
            f"- Event-stage window ALL-frame match: `{_fmt_pct(summary['mean_matrix_stage_window_all_match_rate'])}`",
            f"- Event ANY-frame match: `{_fmt_pct(summary['mean_matrix_event_any_match_rate'])}`",
            f"- Event ALL-frame match: `{_fmt_pct(summary['mean_matrix_event_all_match_rate'])}`",
            "",
            f"- Metrics CSV: `{metrics_csv}`",
            f"- Matrix verification CSV: `{matrix_csv}`",
            f"- Summary CSV: `{summary_csv}`",
        ],
    )
    print("\nSaved:", metrics_csv)
    print("Saved:", matrix_csv)
    print("Saved:", summary_csv)
    print("Saved:", details_md)


if __name__ == "__main__":
    main()
