"""Step 22: compact exiD exploratory cut-in report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math

import numpy as np
import pandas as pd

from cutin_risk.datasets.exid.reader import load_exid_recording
from cutin_risk.datasets.exid.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.exid.transforms import (
    BuildOptions,
    build_lane_change_events,
    build_tracking_table,
)
from cutin_risk.detection.cutin import CutInOptions, detect_cutins
from cutin_risk.io.step_reports import step_figures_dir, step_reports_dir, write_step_markdown
from cutin_risk.paths import exid_dataset_root_path
from cutin_risk.thesis_config import thesis_bool, thesis_int, thesis_str


STEP_NUMBER = 22


@dataclass(frozen=True)
class RecordingSelection:
    recording_id: str
    location_id: int
    duration_s: float
    num_vehicles: int
    num_vrus: int
    lanechange_rows: int
    lanechange_tracks: int


def _all_recording_ids(root: Path) -> list[str]:
    pattern = f"*_{RECORDING_META_SUFFIX}.csv"
    ids = {
        (rid.zfill(2) if rid.isdigit() else rid)
        for path in root.glob(pattern)
        for rid in [path.stem.split("_", 1)[0]]
        if rid
    }
    return sorted(ids)


def _parse_recordings_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    token = value.strip()
    if not token:
        return []
    parts = [part.strip() for part in token.split(",") if part.strip()]
    out: list[str] = []
    for part in parts:
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            out.extend([f"{index:02d}" for index in range(start, end + 1)])
        else:
            out.append(f"{int(part):02d}" if part.isdigit() else part)
    return out


def _recording_selection_row(dataset_root: Path, recording_id: str) -> RecordingSelection:
    recording_meta = pd.read_csv(dataset_root / f"{recording_id}_recordingMeta.csv").iloc[0]
    tracks = pd.read_csv(dataset_root / f"{recording_id}_tracks.csv", usecols=["trackId", "laneChange"])
    lanechange_rows = int(pd.to_numeric(tracks["laneChange"], errors="coerce").fillna(0).gt(0).sum())
    lanechange_tracks = int(
        tracks.loc[pd.to_numeric(tracks["laneChange"], errors="coerce").fillna(0).gt(0), "trackId"]
        .astype(int)
        .nunique()
    )
    return RecordingSelection(
        recording_id=recording_id,
        location_id=int(recording_meta["locationId"]),
        duration_s=float(recording_meta["duration"]),
        num_vehicles=int(recording_meta["numVehicles"]),
        num_vrus=int(recording_meta["numVRUs"]),
        lanechange_rows=lanechange_rows,
        lanechange_tracks=lanechange_tracks,
    )


def _auto_select_recordings(dataset_root: Path, *, max_recordings: int) -> list[RecordingSelection]:
    rows = [_recording_selection_row(dataset_root, rid) for rid in _all_recording_ids(dataset_root)]
    active_rows = [row for row in rows if row.lanechange_tracks > 0]
    active_rows.sort(key=lambda row: (row.location_id, row.duration_s, row.recording_id))

    selected: list[RecordingSelection] = []
    selected_ids: set[str] = set()

    for location_id in sorted({row.location_id for row in active_rows}):
        candidates = [row for row in active_rows if row.location_id == location_id]
        if not candidates:
            continue
        chosen = candidates[0]
        selected.append(chosen)
        selected_ids.add(chosen.recording_id)

    remaining = [row for row in active_rows if row.recording_id not in selected_ids]
    remaining.sort(key=lambda row: (row.duration_s, -row.lanechange_tracks, row.recording_id))
    for row in remaining:
        if len(selected) >= int(max_recordings):
            break
        selected.append(row)
        selected_ids.add(row.recording_id)

    selected.sort(key=lambda row: (row.location_id, row.duration_s, row.recording_id))
    return selected[: int(max_recordings)]


def _finite_min(series: pd.Series, *, lower_bound: float | None = None) -> float:
    values: list[float] = []
    for value in pd.to_numeric(series, errors="coerce").tolist():
        number = float(value)
        if not math.isfinite(number):
            continue
        if lower_bound is not None:
            number = max(number, float(lower_bound))
        values.append(number)
    return min(values) if values else float("inf")


def _safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    return float(values.median())


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _finite_median_from_rows(rows: list[dict[str, float | int | str]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if key in row and math.isfinite(float(row[key]))
    ]
    if not values:
        return float("nan")
    return float(np.median(values))


def _process_recording(
    *,
    dataset_root: Path,
    selection: RecordingSelection,
    cutin_options: CutInOptions,
    lane_marker_merge_gap_frames: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    rec = load_exid_recording(dataset_root, selection.recording_id)
    df = build_tracking_table(
        rec,
        options=BuildOptions(
            keep_optional_tracks_columns=True,
            keep_only_motor_vehicles=True,
        ),
    )
    lane_changes = build_lane_change_events(df, merge_gap_frames=int(lane_marker_merge_gap_frames))
    cutins = detect_cutins(df, lane_changes, options=cutin_options)

    indexed = df.set_index(["id", "frame"], drop=False).sort_index()
    event_rows: list[dict[str, float | int | str]] = []

    for event in cutins:
        try:
            cutter_slice = indexed.loc[
                (int(event.cutter_id), slice(int(event.relation_start_frame), int(event.relation_end_frame))),
                :,
            ]
            follower_slice = indexed.loc[
                (int(event.follower_id), slice(int(event.relation_start_frame), int(event.relation_end_frame))),
                :,
            ]
        except KeyError:
            continue

        if isinstance(cutter_slice, pd.Series):
            cutter_slice = cutter_slice.to_frame().T
        if isinstance(follower_slice, pd.Series):
            follower_slice = follower_slice.to_frame().T
        cutter_slice = cutter_slice.reset_index(drop=True)
        follower_slice = follower_slice.reset_index(drop=True)

        follower_slice = follower_slice.copy()
        follower_slice = follower_slice[follower_slice["precedingId"].astype(int) == int(event.cutter_id)]
        if follower_slice.empty:
            continue

        joined = follower_slice.merge(
            cutter_slice[["frame", "time", "speed", "latVelocity", "yCenter"]],
            on=["frame", "time"],
            how="left",
            suffixes=("", "_cutter"),
        )

        if joined.empty:
            continue

        ttc_series = pd.to_numeric(joined["leadTTC"], errors="coerce")
        ttc_series = ttc_series.where(ttc_series > 0.0)
        execution_rows = float(len(joined))
        execution_dhw_min = float(_finite_min(joined["leadDHW"], lower_bound=0.0))
        execution_thw_min = float(_finite_min(joined["leadTHW"], lower_bound=0.0))
        execution_ttc_min = float(_finite_min(ttc_series))
        y_values = pd.to_numeric(joined["yCenter_cutter"], errors="coerce").dropna()
        cutter_dy = float(y_values.iloc[-1] - y_values.iloc[0]) if len(y_values) >= 2 else 0.0

        event_rows.append(
            {
                "dataset": "exid",
                "recording_id": rec.recording_id,
                "location_id": int(rec.recording_meta.loc[0, "locationId"]),
                "cutter_id": int(event.cutter_id),
                "follower_id": int(event.follower_id),
                "from_lane": int(event.from_lane),
                "to_lane": int(event.to_lane),
                "t0_frame": int(event.relation_start_frame),
                "t0_time": float(event.relation_start_time),
                "lane_change_start_frame": int(event.lane_change_start_frame),
                "lane_change_end_frame": int(event.lane_change_end_frame),
                "relation_start_frame": int(event.relation_start_frame),
                "relation_end_frame": int(event.relation_end_frame),
                "relation_duration_frames": int(event.relation_end_frame - event.relation_start_frame + 1),
                "relation_duration_s": float(event.relation_end_time - event.relation_start_time),
                "dhw_min_total": execution_dhw_min,
                "thw_min_total": execution_thw_min,
                "ttc_min_total": execution_ttc_min,
                "execution_rows": execution_rows,
                "execution_dhw_min": execution_dhw_min,
                "execution_thw_min": execution_thw_min,
                "execution_ttc_min": execution_ttc_min,
                "execution_cutter_speed_mean": float(_safe_mean(joined["speed_cutter"])),
                "execution_follower_speed_mean": float(_safe_mean(joined["speed"])),
                "execution_cutter_lat_v_abs_max": float(
                    pd.to_numeric(joined["latVelocity_cutter"], errors="coerce").abs().max()
                ),
                "execution_cutter_dy": cutter_dy,
            }
        )

    summary = {
        "recording_id": rec.recording_id,
        "location_id": int(rec.recording_meta.loc[0, "locationId"]),
        "duration_s": float(rec.recording_meta.loc[0, "duration"]),
        "num_vehicles": int(rec.recording_meta.loc[0, "numVehicles"]),
        "num_vrus": int(rec.recording_meta.loc[0, "numVRUs"]),
        "lane_changes": int(len(lane_changes)),
        "cutins": int(len(cutins)),
        "exported_events": int(len(event_rows)),
        "execution_thw_median": _finite_median_from_rows(event_rows, "execution_thw_min"),
        "execution_dhw_median": _finite_median_from_rows(event_rows, "execution_dhw_min"),
        "execution_ttc_finite_share": float(
            np.mean([math.isfinite(float(row["execution_ttc_min"])) for row in event_rows])
        )
        if event_rows
        else float("nan"),
    }
    return summary, event_rows


def _build_details_markdown(
    *,
    dataset_root: Path,
    selected_recordings: pd.DataFrame,
    recording_summary: pd.DataFrame,
    merged: pd.DataFrame,
    exploratory_summary: pd.DataFrame,
    cutin_options: CutInOptions,
    lane_marker_merge_gap_frames: int,
) -> list[str]:
    lines = [
        "# Step 22 exiD Exploratory Report",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Dataset root: `{dataset_root}`",
        "- Scope: exploratory exiD sidecar only; not part of the main highD benchmark pipeline.",
        "- Metric source: exiD native `leadDHW/leadTHW/leadTTC` on the explicit cutter-follower relation window.",
        "",
        "## Detection configuration",
        f"- `search_window_frames`: `{cutin_options.search_window_frames}`",
        f"- `max_relation_delay_frames`: `{cutin_options.max_relation_delay_frames}`",
        f"- `min_relation_frames`: `{cutin_options.min_relation_frames}`",
        f"- `require_lane_match`: `{cutin_options.require_lane_match}`",
        f"- `require_new_follower`: `{cutin_options.require_new_follower}`",
        f"- `lane_marker_merge_gap_frames`: `{int(lane_marker_merge_gap_frames)}`",
        "",
        "## Exploratory summary",
    ]
    if not exploratory_summary.empty:
        row = exploratory_summary.iloc[0]
        lines.extend(
            [
                f"- Recordings processed: `{int(row['recordings_processed'])}`",
                f"- Locations covered: `{int(row['locations_covered'])}`",
                f"- Lane changes: `{int(row['lane_changes'])}`",
                f"- Candidate cut-ins: `{int(row['cutins'])}`",
                f"- Exported events: `{int(row['exported_events'])}`",
                f"- Execution THW median: `{float(row['execution_thw_median']):.3f}` s",
                f"- Execution DHW median: `{float(row['execution_dhw_median']):.3f}` m",
                f"- Finite TTC share: `{float(row['execution_ttc_finite_share']):.3f}`",
            ]
        )
    else:
        lines.append("- No events exported.")

    lines.extend(
        [
            "",
            "## Selected recordings",
            "```text",
            selected_recordings.to_string(index=False),
            "```",
            "",
            "## Recording summary",
            "```text",
            recording_summary.to_string(index=False) if not recording_summary.empty else "No recording summary rows.",
            "```",
            "",
            "## Lowest execution THW events",
        ]
    )
    if merged.empty:
        lines.append("No event rows exported.")
    else:
        focus = (
            merged.replace([np.inf, -np.inf], np.nan)
            .sort_values(["execution_thw_min", "execution_dhw_min"], na_position="last")
            .head(5)[
                [
                    "recording_id",
                    "location_id",
                    "cutter_id",
                    "follower_id",
                    "t0_frame",
                    "execution_dhw_min",
                    "execution_thw_min",
                    "execution_ttc_min",
                ]
            ]
        )
        lines.extend(["```text", focus.to_string(index=False), "```"])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 22: compact exiD exploratory cut-in report.")
    parser.add_argument("--dataset-root", type=str, default=str(exid_dataset_root_path()))
    parser.add_argument(
        "--recordings",
        type=str,
        default=thesis_str("step22_exid.recordings", "18,07,28,29,50,42,60,70,77,83"),
        help="Comma-separated recording ids, range syntax, or 'auto'.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=thesis_int("step22_exid.max_recordings", 10, min_value=1),
    )
    parser.add_argument(
        "--lane-marker-merge-gap-frames",
        type=int,
        default=thesis_int("step22_exid.lane_marker_merge_gap_frames", 2, min_value=0),
    )
    parser.add_argument(
        "--search-window-frames",
        type=int,
        default=thesis_int("step22_exid.search_window_frames", 40, min_value=1),
    )
    parser.add_argument(
        "--max-relation-delay-frames",
        type=int,
        default=thesis_int("step22_exid.max_relation_delay_frames", 20, min_value=0),
    )
    parser.add_argument(
        "--min-relation-frames",
        type=int,
        default=thesis_int("step22_exid.min_relation_frames", 5, min_value=1),
    )
    parser.add_argument(
        "--make-plot",
        action=argparse.BooleanOptionalAction,
        default=thesis_bool("step22_exid.make_plot", True),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing exiD dataset root: {dataset_root}")

    if args.recordings.strip().lower() == "auto":
        selected_rows = _auto_select_recordings(dataset_root, max_recordings=int(args.max_recordings))
        recording_ids = [row.recording_id for row in selected_rows]
    else:
        recording_ids = _parse_recordings_arg(args.recordings)[: int(args.max_recordings)]
        selected_rows = [_recording_selection_row(dataset_root, rid) for rid in recording_ids]

    if not recording_ids:
        raise ValueError("No exiD recordings selected.")

    selected_recordings_df = pd.DataFrame([row.__dict__ for row in selected_rows]).sort_values("recording_id")

    reports_dir = step_reports_dir(STEP_NUMBER)
    figures_dir = step_figures_dir(STEP_NUMBER)

    cutin_options = CutInOptions(
        search_window_frames=int(args.search_window_frames),
        start_offset_frames=0,
        max_relation_delay_frames=int(args.max_relation_delay_frames),
        min_relation_frames=int(args.min_relation_frames),
        require_new_follower=False,
        precheck_frames=0,
        require_lane_match=False,
        require_preceding_consistency=True,
        lane_col="laneId",
        following_col="followingId",
        preceding_col="precedingId",
    )

    summary_rows: list[dict[str, float | int | str]] = []
    merged_rows: list[dict[str, float | int | str]] = []
    for row in selected_rows:
        summary, events = _process_recording(
            dataset_root=dataset_root,
            selection=row,
            cutin_options=cutin_options,
            lane_marker_merge_gap_frames=int(args.lane_marker_merge_gap_frames),
        )
        summary_rows.append(summary)
        merged_rows.extend(events)

    recording_summary_df = pd.DataFrame(summary_rows).sort_values("recording_id").reset_index(drop=True)
    merged_df = pd.DataFrame(merged_rows).sort_values(
        ["recording_id", "relation_start_frame", "cutter_id", "follower_id"]
    ).reset_index(drop=True) if merged_rows else pd.DataFrame()

    exploratory_summary = pd.DataFrame(
        [
            {
                "recordings_processed": int(len(recording_summary_df)),
                "locations_covered": int(recording_summary_df["location_id"].nunique()) if not recording_summary_df.empty else 0,
                "lane_changes": int(recording_summary_df["lane_changes"].sum()) if not recording_summary_df.empty else 0,
                "cutins": int(recording_summary_df["cutins"].sum()) if not recording_summary_df.empty else 0,
                "exported_events": int(recording_summary_df["exported_events"].sum()) if not recording_summary_df.empty else 0,
                "execution_thw_median": _safe_median(merged_df["execution_thw_min"]) if not merged_df.empty else float("nan"),
                "execution_dhw_median": _safe_median(merged_df["execution_dhw_min"]) if not merged_df.empty else float("nan"),
                "execution_ttc_finite_share": float(
                    pd.to_numeric(merged_df["execution_ttc_min"], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean()
                )
                if not merged_df.empty
                else float("nan"),
            }
        ]
    )

    selected_csv = reports_dir / "exid_selected_recordings.csv"
    summary_csv = reports_dir / "exid_recording_summary.csv"
    merged_csv = reports_dir / "exid_stage_features_merged.csv"
    overview_csv = reports_dir / "exid_exploratory_summary.csv"
    examples_csv = reports_dir / "exid_example_events.csv"

    selected_recordings_df.to_csv(selected_csv, index=False)
    recording_summary_df.to_csv(summary_csv, index=False)
    exploratory_summary.to_csv(overview_csv, index=False)
    if merged_df.empty:
        merged_df = pd.DataFrame(
            columns=[
                "dataset",
                "recording_id",
                "location_id",
                "cutter_id",
                "follower_id",
                "from_lane",
                "to_lane",
                "t0_frame",
                "t0_time",
                "lane_change_start_frame",
                "lane_change_end_frame",
                "relation_start_frame",
                "relation_end_frame",
                "relation_duration_frames",
                "relation_duration_s",
                "dhw_min_total",
                "thw_min_total",
                "ttc_min_total",
                "execution_rows",
                "execution_dhw_min",
                "execution_thw_min",
                "execution_ttc_min",
                "execution_cutter_speed_mean",
                "execution_follower_speed_mean",
                "execution_cutter_lat_v_abs_max",
                "execution_cutter_dy",
            ]
        )
    merged_df.to_csv(merged_csv, index=False)
    (
        merged_df.replace([np.inf, -np.inf], np.nan)
        .sort_values(["execution_thw_min", "execution_dhw_min"], na_position="last")
        .head(2)
        .to_csv(examples_csv, index=False)
    )

    if bool(args.make_plot) and not recording_summary_df.empty:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            pass
        else:
            plot_df = recording_summary_df.copy().sort_values("recording_id")
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
            axes[0].bar(plot_df["recording_id"], plot_df["exported_events"], color="#0ea5e9")
            axes[0].set_title("exiD exported cut-ins")
            axes[0].set_xlabel("Recording")
            axes[0].set_ylabel("Events")
            axes[1].bar(plot_df["recording_id"], plot_df["execution_thw_median"], color="#f59e0b")
            axes[1].set_title("Median execution THW")
            axes[1].set_xlabel("Recording")
            axes[1].set_ylabel("THW [s]")
            for axis in axes:
                axis.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(figures_dir / "exid_exploratory_overview.png", dpi=180)
            plt.close(fig)

    details_path = write_step_markdown(
        STEP_NUMBER,
        "exid_exploratory_details.md",
        _build_details_markdown(
            dataset_root=dataset_root,
            selected_recordings=selected_recordings_df,
            recording_summary=recording_summary_df,
            merged=merged_df,
            exploratory_summary=exploratory_summary,
            cutin_options=cutin_options,
            lane_marker_merge_gap_frames=int(args.lane_marker_merge_gap_frames),
        ),
    )

    print("== Step 22: exiD exploratory report ==")
    print("Selected recordings:", ", ".join(recording_ids))
    print("Saved:", selected_csv)
    print("Saved:", summary_csv)
    print("Saved:", merged_csv)
    print("Saved:", overview_csv)
    print("Saved:", examples_csv)
    print("Saved:", details_path)
    if not exploratory_summary.empty:
        print(exploratory_summary.to_string(index=False))


if __name__ == "__main__":
    main()
