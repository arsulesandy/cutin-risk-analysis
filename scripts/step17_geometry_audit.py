"""Step 17 geometry audit: verify Step-04 indicator model consistency for highD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import math

import pandas as pd

from cutin_risk.io.markdown import markdown_table
from cutin_risk.io.step_reports import mirror_file_to_step
from cutin_risk.paths import output_path
from cutin_risk.thesis_config import thesis_float, thesis_str


VALIDATION_COLS = {
    "center": "validation_dhw_median_abs_error_center",
    "rear": "validation_dhw_median_abs_error_rear",
    "bbox_topleft": "validation_dhw_median_abs_error_bbox_topleft",
}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _best_model(row: pd.Series) -> tuple[str | None, dict[str, float]]:
    errs: dict[str, float] = {}
    for model, col in VALIDATION_COLS.items():
        if col not in row.index:
            continue
        v = _to_float(row[col])
        if math.isfinite(v):
            errs[model] = v
    if not errs:
        return None, errs
    return min(errs, key=errs.get), errs


def _normalize_recording_id(value: object) -> str:
    s = str(value).strip()
    if not s:
        return s
    try:
        f = float(s)
    except Exception:
        return s
    if f.is_integer():
        return str(int(f))
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 17: geometry/model consistency audit for Step 04 outputs.")
    ap.add_argument(
        "--step04-by-recording-csv",
        type=str,
        default=str(output_path("reports/Step 04/risk_metrics_by_recording.csv")),
    )
    ap.add_argument(
        "--step04-events-csv",
        type=str,
        default=str(output_path("reports/Step 04/risk_metrics_events.csv")),
    )
    ap.add_argument(
        "--expected-position-reference",
        type=str,
        default=thesis_str(
            "indicators.position_reference",
            "bbox_topleft",
            allowed={"center", "rear", "bbox_topleft"},
        ),
    )
    ap.add_argument(
        "--suspicious-raw-dhw-threshold",
        type=float,
        default=thesis_float("step17.suspicious_raw_dhw_threshold", 1.0, min_value=0.0),
    )
    ap.add_argument(
        "--suspicious-raw-thw-threshold",
        type=float,
        default=thesis_float("step17.suspicious_raw_thw_threshold", 0.05, min_value=0.0),
    )
    ap.add_argument("--out-dir", type=str, default=str(output_path("reports/final")))
    args = ap.parse_args()

    by_csv = Path(args.step04_by_recording_csv)
    events_csv = Path(args.step04_events_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not by_csv.exists():
        raise FileNotFoundError(f"Missing Step-04 by-recording CSV: {by_csv}")
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing Step-04 events CSV: {events_csv}")

    by = pd.read_csv(by_csv)
    events = pd.read_csv(events_csv)

    required_by_cols = {
        "recording_id",
        "model_position_reference",
        "validation_dhw_median_abs_error_center",
        "validation_dhw_median_abs_error_rear",
        "validation_dhw_median_abs_error_bbox_topleft",
    }
    missing_by = sorted(required_by_cols - set(by.columns))
    if missing_by:
        raise ValueError(f"Missing required columns in {by_csv}: {missing_by}")

    required_event_cols = {
        "recording_id",
        "cutter_id",
        "follower_id",
        "relation_start_frame",
        "dhw_min",
        "thw_min",
        "dhw_min_raw_window",
        "thw_min_raw_window",
    }
    missing_events = sorted(required_event_cols - set(events.columns))
    if missing_events:
        raise ValueError(f"Missing required columns in {events_csv}: {missing_events}")

    per_recording_rows: list[dict[str, object]] = []
    for _, row in by.iterrows():
        rec_id = _normalize_recording_id(row["recording_id"])
        chosen = str(row.get("model_position_reference", "")).strip()
        best_model, errs = _best_model(row)
        chosen_err = errs.get(chosen, float("nan"))
        best_err = errs.get(best_model, float("nan")) if best_model is not None else float("nan")

        position_match = chosen == args.expected_position_reference
        chosen_is_best = best_model is not None and chosen == best_model
        recording_pass = bool(position_match and chosen_is_best)

        per_recording_rows.append(
            {
                "recording_id": rec_id,
                "chosen_model": chosen,
                "best_model_by_validation": best_model or "",
                "position_match_expected": position_match,
                "chosen_matches_best": chosen_is_best,
                "chosen_error": chosen_err,
                "best_error": best_err,
                "error_center": errs.get("center", float("nan")),
                "error_rear": errs.get("rear", float("nan")),
                "error_bbox_topleft": errs.get("bbox_topleft", float("nan")),
                "recording_pass": recording_pass,
            }
        )

    per_df = pd.DataFrame(per_recording_rows).sort_values("recording_id").reset_index(drop=True)

    events = events.copy()
    events["dhw_min"] = pd.to_numeric(events["dhw_min"], errors="coerce")
    events["thw_min"] = pd.to_numeric(events["thw_min"], errors="coerce")
    events["dhw_min_raw_window"] = pd.to_numeric(events["dhw_min_raw_window"], errors="coerce")
    events["thw_min_raw_window"] = pd.to_numeric(events["thw_min_raw_window"], errors="coerce")

    suspicious = events.loc[
        (
            (events["dhw_min"] <= 0.0) | (events["thw_min"] <= 0.0)
        )
        & (
            (events["dhw_min_raw_window"] < -abs(float(args.suspicious_raw_dhw_threshold)))
            | (events["thw_min_raw_window"] < -abs(float(args.suspicious_raw_thw_threshold)))
        )
    ].copy()

    suspicious = suspicious.sort_values(
        ["dhw_min_raw_window", "thw_min_raw_window"],
        ascending=[True, True],
    )
    suspicious_out = out_dir / "geometry_audit_suspicious_events.csv"
    suspicious.to_csv(suspicious_out, index=False)

    recording_pass_all = bool(per_df["recording_pass"].fillna(False).all()) if not per_df.empty else False
    suspicious_count = int(len(suspicious))
    overall_pass = bool(recording_pass_all and suspicious_count == 0)

    out_csv = out_dir / "geometry_audit_by_recording.csv"
    per_df.to_csv(out_csv, index=False)

    rows = [
        ["recordings_checked", int(len(per_df))],
        ["recording_pass_all", recording_pass_all],
        ["suspicious_events", suspicious_count],
        ["overall_pass", overall_pass],
        ["expected_position_reference", args.expected_position_reference],
        ["step04_by_recording_csv", str(by_csv)],
        ["step04_events_csv", str(events_csv)],
    ]
    summary_table = markdown_table(headers=["check", "value"], rows=rows, align=["left", "left"])

    top_susp = suspicious.head(10)
    if top_susp.empty:
        suspicious_table = "No suspicious events found."
    else:
        suspicious_table = markdown_table(
            headers=[
                "Recording",
                "Cutter",
                "Follower",
                "Start frame",
                "dhw_min",
                "thw_min",
                "dhw_raw_min",
                "thw_raw_min",
            ],
            rows=[
                [
                    _normalize_recording_id(r.recording_id),
                    int(r.cutter_id),
                    int(r.follower_id),
                    int(r.relation_start_frame),
                    f"{float(r.dhw_min):.3f}",
                    f"{float(r.thw_min):.3f}",
                    f"{float(r.dhw_min_raw_window):.3f}",
                    f"{float(r.thw_min_raw_window):.3f}",
                ]
                for _, r in top_susp.iterrows()
            ],
            align=["left", "right", "right", "right", "right", "right", "right", "right"],
        )

    detail_table = markdown_table(
        headers=[
            "Recording",
            "Chosen model",
            "Best model",
            "Chosen err",
            "Best err",
            "Position match",
            "Chosen is best",
            "Pass",
        ],
        rows=[
            [
                str(r.recording_id),
                str(r.chosen_model),
                str(r.best_model_by_validation),
                f"{float(r.chosen_error):.6g}" if math.isfinite(float(r.chosen_error)) else "nan",
                f"{float(r.best_error):.6g}" if math.isfinite(float(r.best_error)) else "nan",
                str(bool(r.position_match_expected)),
                str(bool(r.chosen_matches_best)),
                str(bool(r.recording_pass)),
            ]
            for _, r in per_df.iterrows()
        ],
        align=["left", "left", "left", "right", "right", "left", "left", "left"],
    )

    now = datetime.now(timezone.utc).isoformat()
    out_md = out_dir / "geometry_audit.md"
    out_md.write_text(
        "\n".join(
            [
                "# Geometry Audit Report",
                "",
                f"Generated: `{now}`",
                "",
                "## Summary",
                "",
                summary_table,
                "",
                "## Per-recording checks",
                "",
                detail_table,
                "",
                "## Suspicious events (top 10)",
                "",
                suspicious_table,
                "",
                f"- Full suspicious-event CSV: `{suspicious_out}`",
                f"- Full per-recording CSV: `{out_csv}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    canonical_csv = mirror_file_to_step(out_csv, 17)
    canonical_md = mirror_file_to_step(out_md, 17)
    canonical_susp = mirror_file_to_step(suspicious_out, 17)

    print("== Step 17: Geometry audit ==")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Saved: {suspicious_out}")
    print(f"Mirrored: {canonical_csv}")
    print(f"Mirrored: {canonical_md}")
    print(f"Mirrored: {canonical_susp}")
    print(f"Pass: {overall_pass}")

    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
