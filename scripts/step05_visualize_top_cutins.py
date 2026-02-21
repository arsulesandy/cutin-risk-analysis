"""Step 05: visualize only the globally highest-risk cut-ins using Step-04 metrics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.indicators.surrogate_safety import (
    IndicatorOptions,
    LongitudinalModel,
    compute_pair_timeseries,
    infer_direction_sign_map,
)
from cutin_risk.io.markdown import markdown_table
from cutin_risk.paths import dataset_root_path, step_display_name, step_output_dir
from cutin_risk.thesis_config import thesis_float, thesis_int, thesis_str

STEP_NUMBER = 5
STEP04_NUMBER = 4


def _norm_recording_id(value: str | int | float) -> str:
    token = str(value).strip()
    return f"{int(token):02d}" if token.isdigit() else token


def _safe_float(x: float, clip: float | None = None) -> float:
    if not math.isfinite(float(x)):
        return float("nan")
    value = float(x)
    return float(min(value, clip)) if clip is not None else value


def _fmt_value(value: float, *, digits: int = 3) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def _plot_event_metrics(
    *,
    t_rel: np.ndarray,
    dhw: np.ndarray,
    thw: np.ndarray,
    ttc: np.ndarray,
    title: str,
    out_path: Path,
    ttc_clip: float,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))

    axes[0].plot(t_rel, dhw, color="#1f77b4")
    axes[0].axvline(0.0, linestyle="--", color="black", linewidth=1.0)
    axes[0].set_ylabel("DHW (m)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t_rel, thw, color="#2ca02c")
    axes[1].axvline(0.0, linestyle="--", color="black", linewidth=1.0)
    axes[1].set_ylabel("THW (s)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t_rel, ttc, color="#d62728")
    axes[2].axvline(0.0, linestyle="--", color="black", linewidth=1.0)
    axes[2].set_ylabel(f"TTC (s, clip={ttc_clip:.0f})")
    axes[2].set_xlabel("time relative to relation start (s)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


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
                "selected_top_events",
                "ttc_min_recording",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda x: str(x["recording_id"])):
            writer.writerow(row)


def _write_events_csv(rows: list[dict[str, int | float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_rank",
                "selected_for_plot",
                "figure_path",
                "recording_id",
                "cutter_id",
                "follower_id",
                "from_lane",
                "to_lane",
                "relation_start_time",
                "dhw_min",
                "thw_min",
                "ttc_min",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _top_events_table_lines(rows: list[dict[str, int | float | str]], *, top_n: int = 20) -> list[str]:
    if not rows:
        return ["No events with computed metrics."]
    body = [
        [
            int(row["global_rank"]),
            str(row["recording_id"]),
            int(row["cutter_id"]),
            int(row["follower_id"]),
            f"{int(row['from_lane'])}->{int(row['to_lane'])}",
            _fmt_value(float(row["relation_start_time"]), digits=2),
            _fmt_value(float(row["dhw_min"]), digits=2),
            _fmt_value(float(row["thw_min"]), digits=2),
            _fmt_value(float(row["ttc_min"]), digits=2),
        ]
        for row in rows[:top_n]
    ]
    return markdown_table(
        headers=[
            "Rank",
            "Recording",
            "Cutter",
            "Follower",
            "Lane",
            "t_rel_start (s)",
            "DHW(min)",
            "THW(min)",
            "TTC(min)",
        ],
        rows=body,
        align=["right", "left", "right", "right", "left", "right", "right", "right", "right"],
    ).splitlines()


def _build_details_markdown(
    *,
    dataset_root: Path,
    events_source_path: Path,
    by_recording_source_path: Path,
    position_reference: str,
    min_speed: float,
    closing_speed_epsilon: float,
    pre_seconds: float,
    post_seconds: float,
    ttc_clip: float,
    top_k: int,
    per_recording_rows: list[dict[str, int | float | str]],
    events_rows: list[dict[str, int | float | str]],
    plot_failures: list[str],
    timestamp: str,
) -> str:
    lines = [
        f"# {step_display_name(STEP_NUMBER)} Top Cut-in Visualization Report",
        "",
        f"- Generated at: {timestamp}",
        f"- Dataset root: `{dataset_root}`",
        f"- Step 04 per-event source: `{events_source_path}`",
        f"- Step 04 per-recording source: `{by_recording_source_path}`",
        "",
        "## Configuration",
        f"- Indicator model `position_reference`: {position_reference}",
        f"- Indicator `min_speed`: {min_speed}",
        f"- Indicator `closing_speed_epsilon`: {closing_speed_epsilon}",
        f"- Event window `pre_seconds`: {pre_seconds}",
        f"- Event window `post_seconds`: {post_seconds}",
        f"- Plot `ttc_clip`: {ttc_clip}",
        f"- Global top-k visualized events: {top_k}",
        "",
        "## Dataset-wide summary",
        f"- Recordings discovered (from Step 04): {len(per_recording_rows)}",
        f"- Total lane changes (from Step 04): {sum(int(r['lane_changes']) for r in per_recording_rows)}",
        f"- Total cut-ins (from Step 04): {sum(int(r['cutins']) for r in per_recording_rows)}",
        f"- Events with computed metrics (from Step 04): {len(events_rows)}",
    ]
    total_cutins = sum(int(r["cutins"]) for r in per_recording_rows)
    if total_cutins:
        lines.append(f"- Metrics coverage over cut-ins: {len(events_rows) / total_cutins:.2%}")

    lines.extend(["", "## Top events by TTC(min)"])
    lines.extend(_top_events_table_lines(events_rows[:max(top_k, 20)], top_n=max(top_k, 20)))

    lines.extend(["", "## Plot reconstruction issues"])
    if plot_failures:
        lines.extend([f"- {msg}" for msg in plot_failures])
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def _infer_relation_start_frame(df: pd.DataFrame, *, cutter_id: int, relation_start_time: float) -> int | None:
    rows = df.loc[df["id"] == cutter_id, ["frame", "time"]]
    if rows.empty:
        return None
    idx = (rows["time"].astype(float) - float(relation_start_time)).abs().idxmin()
    return int(rows.loc[idx, "frame"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: use Step-04 metrics (all recordings) and plot only globally top-risk events."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(dataset_root_path()),
        help="Directory containing highD recording CSV files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=thesis_int("step05.top_k", 3, min_value=1),
        help="Number of most-critical dataset-wide events to visualize.",
    )
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=thesis_float("step05.pre_seconds", 2.0, min_value=0.0),
        help="Seconds before relation start.",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=thesis_float("step05.post_seconds", 3.0, min_value=0.0),
        help="Seconds after relation start.",
    )
    parser.add_argument(
        "--ttc-clip",
        type=float,
        default=thesis_float("step05.ttc_clip", 60.0, min_value=0.0),
        help="Clip TTC in plots to this value (seconds).",
    )
    parser.add_argument(
        "--step04-events-csv",
        type=str,
        default=str(step_output_dir(STEP04_NUMBER, kind="reports") / "risk_metrics_events.csv"),
        help="Step-04 per-event risk metrics CSV (all recordings).",
    )
    parser.add_argument(
        "--step04-by-recording-csv",
        type=str,
        default=str(step_output_dir(STEP04_NUMBER, kind="reports") / "risk_metrics_by_recording.csv"),
        help="Step-04 per-recording summary CSV.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    report_dir = step_output_dir(STEP_NUMBER, kind="reports")
    figure_dir = step_output_dir(STEP_NUMBER, kind="figures")
    events_source_path = Path(args.step04_events_csv)
    by_recording_source_path = Path(args.step04_by_recording_csv)

    if not events_source_path.exists():
        raise FileNotFoundError(f"Missing Step-04 events CSV: {events_source_path}")
    if not by_recording_source_path.exists():
        raise FileNotFoundError(f"Missing Step-04 per-recording CSV: {by_recording_source_path}")

    events_df = pd.read_csv(events_source_path)
    by_recording_df = pd.read_csv(by_recording_source_path)

    required_event_cols = {
        "recording_id",
        "cutter_id",
        "follower_id",
        "from_lane",
        "to_lane",
        "relation_start_time",
        "dhw_min",
        "thw_min",
        "ttc_min",
    }
    missing_event_cols = required_event_cols.difference(events_df.columns)
    if missing_event_cols:
        raise ValueError(f"Step-04 events CSV is missing columns: {sorted(missing_event_cols)}")

    required_recording_cols = {"recording_id", "vehicles", "lane_changes", "cutins", "events_with_metrics", "ttc_min"}
    missing_recording_cols = required_recording_cols.difference(by_recording_df.columns)
    if missing_recording_cols:
        raise ValueError(f"Step-04 per-recording CSV is missing columns: {sorted(missing_recording_cols)}")

    events_df = events_df.copy()
    events_df["recording_id"] = events_df["recording_id"].map(_norm_recording_id)
    events_df = events_df.sort_values(["ttc_min", "thw_min", "dhw_min"], ascending=[True, True, True]).reset_index(drop=True)

    top_k = min(int(args.top_k), len(events_df))
    position_reference = thesis_str("indicators.position_reference", "rear", allowed={"center", "rear"})
    indicator_options = IndicatorOptions(
        min_speed=thesis_float("indicators.min_speed", 0.1, min_value=0.0),
        closing_speed_epsilon=thesis_float("indicators.closing_speed_epsilon", 1e-6, min_value=0.0),
    )
    model = LongitudinalModel(position_reference=position_reference)

    print(f"{step_display_name(STEP_NUMBER)}: Top Cut-in Visualization")
    print("Source events:", events_source_path)
    print("Source per-recording:", by_recording_source_path)
    print("Events available:", len(events_df))
    print("Top events visualized:", top_k)

    print("\nTop events by minimum TTC:")
    for rank, row in events_df.head(top_k).iterrows():
        print(
            f"#{rank + 1}: rec={row['recording_id']}, cutter={int(row['cutter_id'])}, follower={int(row['follower_id'])}, "
            f"{int(row['from_lane'])}->{int(row['to_lane'])}, "
            f"t_rel_start={float(row['relation_start_time']):.2f}s, "
            f"ttc_min={_fmt_value(float(row['ttc_min']), digits=2)}s, "
            f"thw_min={_fmt_value(float(row['thw_min']), digits=2)}s, "
            f"dhw_min={_fmt_value(float(row['dhw_min']), digits=2)}m"
        )

    selected_counts = events_df.head(top_k)["recording_id"].value_counts().to_dict()
    per_recording_rows: list[dict[str, int | float | str]] = []
    for _, row in by_recording_df.iterrows():
        rid = _norm_recording_id(row["recording_id"])
        per_recording_rows.append(
            {
                "recording_id": rid,
                "vehicles": int(row["vehicles"]),
                "lane_changes": int(row["lane_changes"]),
                "cutins": int(row["cutins"]),
                "events_with_metrics": int(row["events_with_metrics"]),
                "selected_top_events": int(selected_counts.get(rid, 0)),
                "ttc_min_recording": float(row["ttc_min"]),
            }
        )

    cache: dict[str, dict[str, object]] = {}
    figure_paths_by_rank: dict[int, str] = {}
    plot_failures: list[str] = []

    for rank, row in events_df.head(top_k).iterrows():
        rec_id = str(row["recording_id"])
        if rec_id not in cache:
            rec = load_highd_recording(dataset_root, rec_id)
            df = build_tracking_table(rec)
            cache[rec_id] = {
                "df": df,
                "frame_rate": float(rec.recording_meta.loc[0, "frameRate"]),
                "sign_map": infer_direction_sign_map(df),
                "indexed": df.set_index(["id", "frame"], drop=False),
            }

        cached = cache[rec_id]
        df = cached["df"]
        frame_rate = float(cached["frame_rate"])
        sign_map = cached["sign_map"]
        indexed = cached["indexed"]

        relation_start_time = float(row["relation_start_time"])
        relation_start_frame = _infer_relation_start_frame(
            df,
            cutter_id=int(row["cutter_id"]),
            relation_start_time=relation_start_time,
        )
        if relation_start_frame is None:
            plot_failures.append(
                f"Rank {rank + 1}: no cutter rows for rec={rec_id}, cutter={int(row['cutter_id'])}"
            )
            continue

        pre_frames = int(round(float(args.pre_seconds) * frame_rate))
        post_frames = int(round(float(args.post_seconds) * frame_rate))
        frames = range(max(1, relation_start_frame - pre_frames), relation_start_frame + post_frames + 1)

        ts = compute_pair_timeseries(
            indexed,
            leader_id=int(row["cutter_id"]),
            follower_id=int(row["follower_id"]),
            frames=frames,
            sign_map=sign_map,
            model=model,
            options=indicator_options,
        )
        if ts.empty:
            plot_failures.append(
                f"Rank {rank + 1}: empty pair time series for rec={rec_id}, cutter={int(row['cutter_id'])}, "
                f"follower={int(row['follower_id'])}"
            )
            continue

        t_rel = ts["time"].to_numpy(dtype=float) - relation_start_time
        dhw = ts["dhw"].to_numpy(dtype=float)
        thw = np.array([_safe_float(x) for x in ts["thw"].to_numpy(dtype=float)], dtype=float)
        ttc = np.array([_safe_float(x, clip=float(args.ttc_clip)) for x in ts["ttc"].to_numpy(dtype=float)], dtype=float)

        prefix = (
            f"rank{rank + 1:02d}_rec{rec_id}_cutter{int(row['cutter_id'])}_follower{int(row['follower_id'])}_"
            f"{int(row['from_lane'])}to{int(row['to_lane'])}"
        )
        out_path = figure_dir / f"{prefix}_risk_metrics.png"
        title = (
            f"Top cut-in #{rank + 1} | rec {rec_id}, cutter {int(row['cutter_id'])}, follower {int(row['follower_id'])}, "
            f"{int(row['from_lane'])}->{int(row['to_lane'])} | "
            f"TTC(min)={_fmt_value(float(row['ttc_min']), digits=2)}s, "
            f"THW(min)={_fmt_value(float(row['thw_min']), digits=2)}s, "
            f"DHW(min)={_fmt_value(float(row['dhw_min']), digits=2)}m"
        )
        _plot_event_metrics(
            t_rel=t_rel,
            dhw=dhw,
            thw=thw,
            ttc=ttc,
            title=title,
            out_path=out_path,
            ttc_clip=float(args.ttc_clip),
            show=args.show,
        )
        figure_paths_by_rank[rank + 1] = str(out_path)

    events_output_rows: list[dict[str, int | float | str]] = []
    for rank, row in events_df.iterrows():
        event_rank = rank + 1
        events_output_rows.append(
            {
                "global_rank": event_rank,
                "selected_for_plot": 1 if event_rank <= top_k else 0,
                "figure_path": figure_paths_by_rank.get(event_rank, ""),
                "recording_id": str(row["recording_id"]),
                "cutter_id": int(row["cutter_id"]),
                "follower_id": int(row["follower_id"]),
                "from_lane": int(row["from_lane"]),
                "to_lane": int(row["to_lane"]),
                "relation_start_time": float(row["relation_start_time"]),
                "dhw_min": float(row["dhw_min"]),
                "thw_min": float(row["thw_min"]),
                "ttc_min": float(row["ttc_min"]),
            }
        )

    per_recording_csv_path = report_dir / "top_cutin_visualization_by_recording.csv"
    _write_per_recording_csv(per_recording_rows, per_recording_csv_path)

    events_csv_path = report_dir / "top_cutin_visualization_events.csv"
    _write_events_csv(events_output_rows, events_csv_path)

    details_md_path = report_dir / "top_cutin_visualization_details.md"
    details_md_path.write_text(
        _build_details_markdown(
            dataset_root=dataset_root,
            events_source_path=events_source_path,
            by_recording_source_path=by_recording_source_path,
            position_reference=position_reference,
            min_speed=indicator_options.min_speed,
            closing_speed_epsilon=indicator_options.closing_speed_epsilon,
            pre_seconds=float(args.pre_seconds),
            post_seconds=float(args.post_seconds),
            ttc_clip=float(args.ttc_clip),
            top_k=top_k,
            per_recording_rows=per_recording_rows,
            events_rows=events_output_rows,
            plot_failures=plot_failures,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        ),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(f"  Details markdown: {details_md_path}")
    print(f"  Per-recording CSV: {per_recording_csv_path}")
    print(f"  Per-event CSV: {events_csv_path}")
    print(f"  Figures directory: {figure_dir}")
    if plot_failures:
        print(f"  Plot reconstruction issues: {len(plot_failures)} (see details markdown)")


if __name__ == "__main__":
    main()
