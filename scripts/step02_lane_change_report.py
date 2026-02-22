"""Step 02: lane-change detection report with overall dataset totals."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
import statistics as stats

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.io.markdown import markdown_table
from cutin_risk.io.progress import iter_with_progress
from cutin_risk.paths import dataset_root_path, step_display_name, step_output_dir

STEP_NUMBER = 2


def _all_recording_ids(root: Path) -> list[str]:
    pattern = f"*_{RECORDING_META_SUFFIX}.csv"
    ids = {
        (rid.zfill(2) if rid.isdigit() else rid)
        for p in root.glob(pattern)
        for rid in [p.stem.split("_", 1)[0]]
        if rid
    }
    return sorted(ids)


def _percentile(values_sorted: list[float], q: float) -> float:
    if not values_sorted:
        raise ValueError("values_sorted must not be empty")
    if len(values_sorted) == 1:
        return values_sorted[0]

    idx = (len(values_sorted) - 1) * q
    low = int(idx)
    high = min(low + 1, len(values_sorted) - 1)
    weight = idx - low
    return values_sorted[low] * (1.0 - weight) + values_sorted[high] * weight


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    v = sorted(values)
    return {
        "count": float(len(v)),
        "min": float(v[0]),
        "p25": float(_percentile(v, 0.25)),
        "median": float(stats.median(v)),
        "mean": float(stats.fmean(v)),
        "p75": float(_percentile(v, 0.75)),
        "max": float(v[-1]),
    }


def _summary_lines(
    label: str,
    summary: dict[str, float],
    *,
    digits: int = 2,
    include_count: bool = True,
) -> list[str]:
    if not summary:
        return [f"- {label}: n/a"]
    lines = []
    if include_count:
        lines.append(f"- {label} count: {int(summary['count'])}")
    lines.extend(
        [
            f"- {label} min: {summary['min']:.{digits}f}",
            f"- {label} p25: {summary['p25']:.{digits}f}",
            f"- {label} median: {summary['median']:.{digits}f}",
            f"- {label} mean: {summary['mean']:.{digits}f}",
            f"- {label} p75: {summary['p75']:.{digits}f}",
            f"- {label} max: {summary['max']:.{digits}f}",
        ]
    )
    return lines


def _transition_table_lines(counts: Counter[tuple[int, int]], *, top_n: int = 15) -> list[str]:
    if not counts:
        return ["No transitions detected."]
    rows = [[a, b, c] for (a, b), c in counts.most_common(top_n)]
    return markdown_table(
        headers=["From lane", "To lane", "Count"],
        rows=rows,
        align=["left", "left", "right"],
    ).splitlines()


def _write_per_recording_csv(rows: list[dict[str, int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["recording_id", "vehicles", "lane_changes"])
        writer.writeheader()
        for row in sorted(rows, key=lambda x: str(x["recording_id"])):
            writer.writerow(row)


def _per_recording_counts_table_lines(rows: list[dict[str, int | str]]) -> list[str]:
    sorted_rows = sorted(rows, key=lambda x: str(x["recording_id"]))
    body_rows: list[list[object]] = (
        [[row["recording_id"], row["vehicles"], row["lane_changes"]] for row in sorted_rows]
        if sorted_rows
        else [["n/a", "n/a", "n/a"]]
    )
    return markdown_table(
        headers=["Recording", "Vehicles", "Lane changes"],
        rows=body_rows,
        align=["left", "right", "right"],
    ).splitlines()


def _build_details_markdown(
    *,
    dataset_root: Path,
    options: LaneChangeOptions,
    overall_transition_counts: Counter[tuple[int, int]],
    recording_ids: list[str],
    per_recording_rows: list[dict[str, int | str]],
    failed_recordings: list[tuple[str, str]],
    total_lane_changes: int,
    overall_duration_frames: list[float],
    overall_duration_seconds: list[float],
    timestamp: str,
) -> str:
    lines = [
        f"# {step_display_name(STEP_NUMBER)} Lane Change Detailed Statistics",
        "",
        f"- Generated at: {timestamp}",
        f"- Dataset root: `{dataset_root}`",
        "",
        "## Detection configuration",
        f"- `min_stable_before_frames`: {options.min_stable_before_frames}",
        f"- `min_stable_after_frames`: {options.min_stable_after_frames}",
        f"- `ignore_lane_ids`: {list(options.ignore_lane_ids)}",
    ]

    lines.extend(
        [
            "",
            "## Dataset-wide summary",
            f"- Recordings discovered: {len(recording_ids)}",
            f"- Recordings processed: {len(per_recording_rows)}",
            f"- Recordings failed: {len(failed_recordings)}",
            f"- Total lane change events: {total_lane_changes}",
        ]
    )
    if per_recording_rows:
        lines.append(f"- Mean lane changes per processed recording: {total_lane_changes / len(per_recording_rows):.2f}")

    lines.extend(["", "### Top transitions (dataset-wide)"])
    lines.extend(_transition_table_lines(overall_transition_counts, top_n=20))
    lines.extend(["", "### Lane-change duration stats (dataset-wide)"])
    lines.append(f"- Events included: {len(overall_duration_frames)} lane changes")
    lines.extend(
        _summary_lines(
            "Duration (frames)",
            _numeric_summary(overall_duration_frames),
            digits=0,
            include_count=False,
        )
    )
    lines.extend(
        _summary_lines(
            "Duration (seconds)",
            _numeric_summary(overall_duration_seconds),
            digits=3,
            include_count=False,
        )
    )

    lines.extend(
        [
            "",
            "## Per-recording counts",
        ]
    )
    lines.extend(_per_recording_counts_table_lines(per_recording_rows))

    lines.extend(["", "## Failed recordings"])
    if failed_recordings:
        for rid, err in failed_recordings:
            lines.append(f"- `{rid}`: {err}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> None:
    root = dataset_root_path()
    report_dir = step_output_dir(STEP_NUMBER, kind="reports")
    lane_change_options = LaneChangeOptions()

    print(f"{step_display_name(STEP_NUMBER)}: Lane Change Report")
    all_recording_ids = _all_recording_ids(root)
    total_lane_changes = 0
    failed_recordings: list[tuple[str, str]] = []
    per_recording_rows: list[dict[str, int | str]] = []
    overall_transition_counts: Counter[tuple[int, int]] = Counter()
    overall_duration_frames: list[float] = []
    overall_duration_seconds: list[float] = []
    for _, _, rid in iter_with_progress(
        all_recording_ids,
        label=f"{step_display_name(STEP_NUMBER)} recordings",
        item_name="recording",
    ):
        try:
            rec_all = load_highd_recording(root, rid)
            df_all = build_tracking_table(rec_all)
            all_events = detect_lane_changes(df_all, options=lane_change_options)
            total_lane_changes += len(all_events)
            overall_transition_counts.update((e.from_lane, e.to_lane) for e in all_events)

            duration_frames = [float(e.end_frame - e.start_frame + 1) for e in all_events]
            duration_seconds = [float(max(0.0, e.end_time - e.start_time)) for e in all_events]
            overall_duration_frames.extend(duration_frames)
            overall_duration_seconds.extend(duration_seconds)

            per_recording_rows.append(
                {
                    "recording_id": rid,
                    "vehicles": int(df_all["id"].nunique()),
                    "lane_changes": len(all_events),
                }
            )
        except Exception as exc:
            failed_recordings.append((rid, str(exc)))

    print("\nOverall stats (all recordings):")
    print("Recordings discovered:", len(all_recording_ids))
    print("Recordings processed:", len(per_recording_rows))
    print("Total lane change events:", total_lane_changes)
    if failed_recordings:
        print("Recordings failed:", ", ".join(rid for rid, _ in failed_recordings))

    summary_csv_path = report_dir / "lane_change_counts_by_recording.csv"
    _write_per_recording_csv(per_recording_rows, summary_csv_path)

    details_md_path = report_dir / "lane_change_details.md"
    details_md_path.write_text(
        _build_details_markdown(
            dataset_root=root,
            options=lane_change_options,
            overall_transition_counts=overall_transition_counts,
            recording_ids=all_recording_ids,
            per_recording_rows=per_recording_rows,
            failed_recordings=failed_recordings,
            total_lane_changes=total_lane_changes,
            overall_duration_frames=overall_duration_frames,
            overall_duration_seconds=overall_duration_seconds,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        ),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(f"  Details markdown: {details_md_path}")
    print(f"  Per-recording CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
