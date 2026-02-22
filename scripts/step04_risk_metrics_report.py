"""Step 04: compute surrogate risk indicators around detected cut-ins."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics as stats

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.cutin import CutInOptions, detect_cutins
from cutin_risk.detection.lane_change import LaneChangeOptions, detect_lane_changes
from cutin_risk.indicators.surrogate_safety import (
    IndicatorOptions,
    LongitudinalModel,
    compute_pair_timeseries,
    infer_direction_sign_map,
    validate_against_dataset_preceding,
)
from cutin_risk.io.markdown import markdown_table
from cutin_risk.io.progress import iter_with_progress
from cutin_risk.paths import dataset_root_path, step_display_name, step_output_dir
from cutin_risk.thesis_config import thesis_float, thesis_int

STEP_NUMBER = 4


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
    digits: int = 3,
    include_count: bool = True,
) -> list[str]:
    if not summary:
        return [f"- {label}: n/a"]
    lines: list[str] = []
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


def _fmt_value(value: float, *, digits: int = 3) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def _finite_min(values: list[float] | tuple[float, ...], *, lower_bound: float | None = None) -> float:
    finite: list[float] = []
    for x in values:
        fx = float(x)
        if not math.isfinite(fx):
            continue
        if lower_bound is not None:
            fx = max(fx, float(lower_bound))
        finite.append(fx)
    return min(finite) if finite else float("inf")


def _event_minima(
    ts,
    *,
    relation_start_frame: int,
    relation_end_frame: int,
) -> dict[str, float]:
    raw_dhw_min = float(ts["dhw"].min())
    raw_thw_min = _finite_min(ts["thw"].tolist())
    raw_ttc_min = _finite_min(ts["ttc"].tolist())

    relation_mask = (
        (ts["frame"].astype(int) >= int(relation_start_frame))
        & (ts["frame"].astype(int) <= int(relation_end_frame))
    )
    relation_ts = ts.loc[relation_mask].copy()

    # Fallback keeps metrics defined if relation frames are unexpectedly missing.
    metric_ts = relation_ts if not relation_ts.empty else ts
    metric_dhw = metric_ts["dhw"].clip(lower=0.0)

    return {
        "relation_rows": float(len(relation_ts)),
        "dhw_min": float(metric_dhw.min()),
        "thw_min": _finite_min(metric_ts["thw"].tolist(), lower_bound=0.0),
        "ttc_min": _finite_min(metric_ts["ttc"].tolist()),
        "dhw_min_raw_window": raw_dhw_min,
        "thw_min_raw_window": raw_thw_min,
        "ttc_min_raw_window": raw_ttc_min,
    }


def _write_per_recording_csv(rows: list[dict[str, int | float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recording_id",
                "vehicles",
                "lane_changes",
                "cutins",
                "events_with_metrics",
                "dhw_min",
                "thw_min",
                "ttc_min",
                "dhw_min_raw_window",
                "thw_min_raw_window",
                "ttc_min_raw_window",
                "finite_thw_events",
                "finite_ttc_events",
                "model_position_reference",
                "validation_dhw_median_abs_error_center",
                "validation_dhw_median_abs_error_rear",
                "validation_dhw_median_abs_error_bbox_topleft",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda x: str(x["recording_id"])):
            writer.writerow(row)


def _write_event_csv(rows: list[dict[str, int | float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recording_id",
                "cutter_id",
                "follower_id",
                "from_lane",
                "to_lane",
                "relation_start_frame",
                "relation_end_frame",
                "relation_start_time",
                "relation_rows",
                "dhw_min",
                "thw_min",
                "ttc_min",
                "dhw_min_raw_window",
                "thw_min_raw_window",
                "ttc_min_raw_window",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _per_recording_counts_table_lines(rows: list[dict[str, int | float | str]]) -> list[str]:
    sorted_rows = sorted(rows, key=lambda x: str(x["recording_id"]))
    body_rows: list[list[object]]
    if sorted_rows:
        body_rows = [
            [
                row["recording_id"],
                row["vehicles"],
                row["lane_changes"],
                row["cutins"],
                row["events_with_metrics"],
                _fmt_value(float(row["ttc_min"])),
                row["model_position_reference"],
            ]
            for row in sorted_rows
        ]
    else:
        body_rows = [["n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]]

    return markdown_table(
        headers=["Recording", "Vehicles", "Lane changes", "Cut-ins", "Metric events", "TTC(min)", "Model"],
        rows=body_rows,
        align=["left", "right", "right", "right", "right", "right", "left"],
    ).splitlines()


def _top_events_table_lines(rows: list[dict[str, int | float | str]], *, top_n: int = 20) -> list[str]:
    if not rows:
        return ["No events with computed metrics."]
    body_rows = [
        [
            row["recording_id"],
            row["cutter_id"],
            row["follower_id"],
            f"{row['from_lane']}->{row['to_lane']}",
            f"{float(row['relation_start_time']):.3f}",
            _fmt_value(float(row["dhw_min"])),
            _fmt_value(float(row["thw_min"])),
            _fmt_value(float(row["ttc_min"])),
        ]
        for row in rows[:top_n]
    ]
    return markdown_table(
        headers=["Recording", "Cutter", "Follower", "Lane", "t_rel_start (s)", "DHW(min)", "THW(min)", "TTC(min)"],
        rows=body_rows,
        align=["left", "right", "right", "left", "right", "right", "right", "right"],
    ).splitlines()


def _build_details_markdown(
    *,
    dataset_root: Path,
    indicator_options: IndicatorOptions,
    pre_frames: int,
    post_frames: int,
    validation_sample_n: int,
    validation_random_state: int,
    recording_ids: list[str],
    per_recording_rows: list[dict[str, int | float | str]],
    failed_recordings: list[tuple[str, str]],
    validation_failed_recordings: list[tuple[str, str]],
    model_choice_counts: Counter[str],
    total_lane_changes: int,
    total_cutins: int,
    event_summaries: list[dict[str, int | float | str]],
    dhw_mins: list[float],
    thw_mins_finite: list[float],
    ttc_mins_finite: list[float],
    timestamp: str,
) -> str:
    lines = [
        f"# {step_display_name(STEP_NUMBER)} Surrogate Risk Detailed Statistics",
        "",
        f"- Generated at: {timestamp}",
        f"- Dataset root: `{dataset_root}`",
        "",
        "## Configuration",
        f"- Indicator `min_speed`: {indicator_options.min_speed}",
        f"- Indicator `closing_speed_epsilon`: {indicator_options.closing_speed_epsilon}",
        f"- Event window `pre_frames`: {pre_frames}",
        f"- Event window `post_frames`: {post_frames}",
        "- Event minima are computed on relation frames only (`relation_start_frame..relation_end_frame`).",
        "- Thesis-facing minima are clipped to non-negative for DHW/THW.",
        "- Raw window minima are still exported as diagnostic columns (`*_raw_window`).",
        f"- Validation `sample_n`: {validation_sample_n}",
        f"- Validation `random_state`: {validation_random_state}",
    ]

    lines.extend(
        [
            "",
            "## Dataset-wide summary",
            f"- Recordings discovered: {len(recording_ids)}",
            f"- Recordings processed: {len(per_recording_rows)}",
            f"- Recordings failed: {len(failed_recordings)}",
            f"- Validation skipped: {len(validation_failed_recordings)}",
            f"- Total lane changes: {total_lane_changes}",
            f"- Total cut-ins: {total_cutins}",
            f"- Events with computed metrics: {len(event_summaries)}",
            f"- Chosen model `center`: {model_choice_counts.get('center', 0)}",
            f"- Chosen model `rear`: {model_choice_counts.get('rear', 0)}",
            f"- Chosen model `bbox_topleft`: {model_choice_counts.get('bbox_topleft', 0)}",
        ]
    )
    if total_cutins:
        lines.append(f"- Metrics coverage over cut-ins: {len(event_summaries) / total_cutins:.2%}")

    lines.extend(["", "### Risk metric summaries (event-level minima)"])
    lines.extend(_summary_lines("DHW(min) [m]", _numeric_summary(dhw_mins), digits=3))
    lines.extend(
        _summary_lines(
            "THW(min) [s] (finite only)",
            _numeric_summary(thw_mins_finite),
            digits=3,
        )
    )
    lines.extend(
        _summary_lines(
            "TTC(min) [s] (finite only)",
            _numeric_summary(ttc_mins_finite),
            digits=3,
        )
    )

    lines.extend(["", "## Top events by TTC(min)"])
    lines.extend(_top_events_table_lines(event_summaries, top_n=20))

    lines.extend(["", "## Per-recording summary"])
    lines.extend(_per_recording_counts_table_lines(per_recording_rows))

    lines.extend(["", "## Failed recordings"])
    if failed_recordings:
        for rid, err in failed_recordings:
            lines.append(f"- `{rid}`: {err}")
    else:
        lines.append("- None")

    lines.extend(["", "## Validation skipped recordings"])
    if validation_failed_recordings:
        for rid, err in validation_failed_recordings:
            lines.append(f"- `{rid}`: {err}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> None:
    root = dataset_root_path()
    report_dir = step_output_dir(STEP_NUMBER, kind="reports")

    lane_change_options = LaneChangeOptions()
    cutin_options = CutInOptions()
    indicator_options = IndicatorOptions(
        min_speed=thesis_float("indicators.min_speed", 0.1, min_value=0.0),
        closing_speed_epsilon=thesis_float(
            "indicators.closing_speed_epsilon",
            1e-6,
            min_value=0.0,
        ),
    )
    pre_frames = thesis_int("step04.pre_frames", 50, min_value=0)
    post_frames = thesis_int("step04.post_frames", 75, min_value=0)
    validation_sample_n = thesis_int("step04.validation_sample_n", 20000, min_value=1)
    validation_random_state = thesis_int("step04.validation_random_state", 7)

    print(f"{step_display_name(STEP_NUMBER)}: Risk Metrics Report")
    all_recording_ids = _all_recording_ids(root)
    total_lane_changes = 0
    total_cutins = 0
    failed_recordings: list[tuple[str, str]] = []
    validation_failed_recordings: list[tuple[str, str]] = []
    per_recording_rows: list[dict[str, int | float | str]] = []
    event_summaries: list[dict[str, int | float | str]] = []
    model_choice_counts: Counter[str] = Counter()
    dhw_mins: list[float] = []
    thw_mins_finite: list[float] = []
    ttc_mins_finite: list[float] = []

    for _, _, rid in iter_with_progress(
        all_recording_ids,
        label=f"{step_display_name(STEP_NUMBER)} recordings",
        item_name="recording",
    ):
        try:
            rec = load_highd_recording(root, rid)
            df = build_tracking_table(rec)

            candidate_models = [
                LongitudinalModel(position_reference="center"),
                LongitudinalModel(position_reference="rear"),
                LongitudinalModel(position_reference="bbox_topleft"),
            ]
            validation_reports: dict[str, dict[str, float]] = {}
            model = candidate_models[0]
            try:
                for candidate in candidate_models:
                    validation_reports[candidate.position_reference] = validate_against_dataset_preceding(
                        df,
                        sample_n=validation_sample_n,
                        random_state=validation_random_state,
                        model=candidate,
                        options=indicator_options,
                    )
                model = min(
                    candidate_models,
                    key=lambda m: validation_reports[m.position_reference]["dhw_median_abs_error"],
                )
            except Exception as exc:
                validation_failed_recordings.append((rid, str(exc)))

            lane_changes = detect_lane_changes(df, options=lane_change_options)
            cutins = detect_cutins(df, lane_changes, options=cutin_options)
            total_lane_changes += len(lane_changes)
            total_cutins += len(cutins)
            model_choice_counts.update([model.position_reference])

            sign_map = infer_direction_sign_map(df)
            indexed = df.set_index(["id", "frame"], drop=False)

            recording_event_rows: list[dict[str, int | float | str]] = []
            for ev in cutins:
                center_frame = int(ev.relation_start_frame)
                frames = range(max(1, center_frame - pre_frames), center_frame + post_frames + 1)

                ts = compute_pair_timeseries(
                    indexed,
                    leader_id=int(ev.cutter_id),
                    follower_id=int(ev.follower_id),
                    frames=frames,
                    sign_map=sign_map,
                    model=model,
                    options=indicator_options,
                )
                if ts.empty:
                    continue

                mins = _event_minima(
                    ts,
                    relation_start_frame=int(ev.relation_start_frame),
                    relation_end_frame=int(ev.relation_end_frame),
                )
                dhw_min = float(mins["dhw_min"])
                thw_min = float(mins["thw_min"])
                ttc_min = float(mins["ttc_min"])

                row = {
                    "recording_id": rid,
                    "cutter_id": int(ev.cutter_id),
                    "follower_id": int(ev.follower_id),
                    "from_lane": int(ev.from_lane),
                    "to_lane": int(ev.to_lane),
                    "relation_start_frame": int(ev.relation_start_frame),
                    "relation_end_frame": int(ev.relation_end_frame),
                    "relation_start_time": float(ev.relation_start_time),
                    "relation_rows": int(mins["relation_rows"]),
                    "dhw_min": dhw_min,
                    "thw_min": thw_min,
                    "ttc_min": ttc_min,
                    "dhw_min_raw_window": float(mins["dhw_min_raw_window"]),
                    "thw_min_raw_window": float(mins["thw_min_raw_window"]),
                    "ttc_min_raw_window": float(mins["ttc_min_raw_window"]),
                }
                recording_event_rows.append(row)
                event_summaries.append(row)
                dhw_mins.append(dhw_min)
                if math.isfinite(thw_min):
                    thw_mins_finite.append(thw_min)
                if math.isfinite(ttc_min):
                    ttc_mins_finite.append(ttc_min)

            per_recording_rows.append(
                {
                    "recording_id": rid,
                    "vehicles": int(df["id"].nunique()),
                    "lane_changes": len(lane_changes),
                    "cutins": len(cutins),
                    "events_with_metrics": len(recording_event_rows),
                    "dhw_min": min((float(r["dhw_min"]) for r in recording_event_rows), default=float("nan")),
                    "thw_min": min((float(r["thw_min"]) for r in recording_event_rows), default=float("inf")),
                    "ttc_min": min((float(r["ttc_min"]) for r in recording_event_rows), default=float("inf")),
                    "dhw_min_raw_window": min(
                        (float(r["dhw_min_raw_window"]) for r in recording_event_rows),
                        default=float("nan"),
                    ),
                    "thw_min_raw_window": min(
                        (float(r["thw_min_raw_window"]) for r in recording_event_rows),
                        default=float("inf"),
                    ),
                    "ttc_min_raw_window": min(
                        (float(r["ttc_min_raw_window"]) for r in recording_event_rows),
                        default=float("inf"),
                    ),
                    "finite_thw_events": sum(math.isfinite(float(r["thw_min"])) for r in recording_event_rows),
                    "finite_ttc_events": sum(math.isfinite(float(r["ttc_min"])) for r in recording_event_rows),
                    "model_position_reference": model.position_reference,
                    "validation_dhw_median_abs_error_center": (
                        float(validation_reports["center"]["dhw_median_abs_error"])
                        if "center" in validation_reports
                        else float("nan")
                    ),
                    "validation_dhw_median_abs_error_rear": (
                        float(validation_reports["rear"]["dhw_median_abs_error"])
                        if "rear" in validation_reports
                        else float("nan")
                    ),
                    "validation_dhw_median_abs_error_bbox_topleft": (
                        float(validation_reports["bbox_topleft"]["dhw_median_abs_error"])
                        if "bbox_topleft" in validation_reports
                        else float("nan")
                    ),
                }
            )
        except Exception as exc:
            failed_recordings.append((rid, str(exc)))

    event_summaries.sort(
        key=lambda r: (
            not math.isfinite(float(r["ttc_min"])),
            float(r["ttc_min"]),
            float(r["thw_min"]),
            float(r["dhw_min"]),
        )
    )

    print("\nOverall stats (all recordings):")
    print("Recordings discovered:", len(all_recording_ids))
    print("Recordings processed:", len(per_recording_rows))
    print("Total lane changes:", total_lane_changes)
    print("Total cut-ins:", total_cutins)
    print("Events with computed metrics:", len(event_summaries))
    if failed_recordings:
        print("Recordings failed:", ", ".join(rid for rid, _ in failed_recordings))

    summary_csv_path = report_dir / "risk_metrics_by_recording.csv"
    _write_per_recording_csv(per_recording_rows, summary_csv_path)

    events_csv_path = report_dir / "risk_metrics_events.csv"
    _write_event_csv(event_summaries, events_csv_path)

    details_md_path = report_dir / "risk_metrics_details.md"
    details_md_path.write_text(
        _build_details_markdown(
            dataset_root=root,
            indicator_options=indicator_options,
            pre_frames=pre_frames,
            post_frames=post_frames,
            validation_sample_n=validation_sample_n,
            validation_random_state=validation_random_state,
            recording_ids=all_recording_ids,
            per_recording_rows=per_recording_rows,
            failed_recordings=failed_recordings,
            validation_failed_recordings=validation_failed_recordings,
            model_choice_counts=model_choice_counts,
            total_lane_changes=total_lane_changes,
            total_cutins=total_cutins,
            event_summaries=event_summaries,
            dhw_mins=dhw_mins,
            thw_mins_finite=thw_mins_finite,
            ttc_mins_finite=ttc_mins_finite,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        ),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(f"  Details markdown: {details_md_path}")
    print(f"  Per-recording CSV: {summary_csv_path}")
    print(f"  Per-event CSV: {events_csv_path}")


if __name__ == "__main__":
    main()
