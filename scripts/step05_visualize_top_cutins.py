"""Step 05: visualize selected high-risk cut-ins and export diagnostic plots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
from cutin_risk.indicators.surrogate_safety import (
    LongitudinalModel,
    IndicatorOptions,
    infer_direction_sign_map,
    compute_pair_timeseries,
)
from cutin_risk.paths import dataset_root_path, output_path
from cutin_risk.thesis_config import thesis_float, thesis_int, thesis_str


def _finite_min(values: pd.Series) -> float:
    vals = [float(x) for x in values.tolist() if math.isfinite(float(x))]
    return min(vals) if vals else float("inf")


def _safe_float(x: float, clip: float | None = None) -> float:
    if not math.isfinite(float(x)):
        return float("nan")
    if clip is not None:
        return float(min(float(x), clip))
    return float(x)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_series(
        *,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        out_path: Path,
        vline_at: float = 0.0,
        show: bool = False,
) -> None:
    plt.figure()
    plt.plot(x, y)
    plt.axvline(vline_at, linestyle="--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _vehicle_state_timeseries(indexed: pd.DataFrame, vehicle_id: int, frames: range) -> pd.DataFrame:
    rows = []
    for f in frames:
        try:
            r = indexed.loc[(vehicle_id, f)]
        except KeyError:
            continue
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append(
            {
                "frame": int(f),
                "time": float(r["time"]),
                "laneId": int(r["laneId"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "xVelocity": float(r["xVelocity"]),
                "yVelocity": float(r["yVelocity"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: rank and visualize top cut-in events.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(dataset_root_path()),
        help="Directory that contains <recording>_tracks.csv, <recording>_tracksMeta.csv, <recording>_recordingMeta.csv",
    )
    parser.add_argument(
        "--recording-id",
        type=str,
        default=thesis_str("step05.recording_id", "01"),
        help="Recording id (e.g., 01)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=thesis_int("step05.top_k", 3, min_value=1),
        help="Number of most-critical events to plot",
    )
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=thesis_float("step05.pre_seconds", 2.0, min_value=0.0),
        help="Seconds before relation start",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=thesis_float("step05.post_seconds", 3.0, min_value=0.0),
        help="Seconds after relation start",
    )
    parser.add_argument(
        "--ttc-clip",
        type=float,
        default=thesis_float("step05.ttc_clip", 60.0, min_value=0.0),
        help="Clip TTC in plots to this value (seconds)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(output_path("figures")),
        help="Base output directory for figures",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    rec = load_highd_recording(dataset_root, args.recording_id)
    df = build_tracking_table(rec)

    frame_rate = float(rec.recording_meta.loc[0, "frameRate"])
    pre_frames = int(round(args.pre_seconds * frame_rate))
    post_frames = int(round(args.post_seconds * frame_rate))

    # Step 2: lane changes
    lane_changes = detect_lane_changes(
        df,
        options=LaneChangeOptions(),
    )

    # Step 3: cut-ins
    cutins = detect_cutins(
        df,
        lane_changes,
        options=CutInOptions(),
    )

    print("== Step 5 ==")
    print("Recording:", rec.recording_id)
    print("Lane changes:", len(lane_changes))
    print("Cut-ins:", len(cutins))

    if not cutins:
        print("No cut-ins found; nothing to visualize.")
        return

    # Step 4 model: use the best-performing reference found during validation earlier.
    # Keep this explicit and consistent across runs.
    model = LongitudinalModel(
        position_reference=thesis_str(
            "indicators.position_reference",
            "rear",
            allowed={"center", "rear"},
        ),
    )
    ind_opt = IndicatorOptions(
        min_speed=thesis_float("indicators.min_speed", 0.1, min_value=0.0),
        closing_speed_epsilon=thesis_float(
            "indicators.closing_speed_epsilon",
            1e-6,
            min_value=0.0,
        ),
    )

    sign_map = infer_direction_sign_map(df)
    indexed = df.set_index(["id", "frame"], drop=False)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out_dir) / f"step5_recording_{rec.recording_id}" / run_id
    _ensure_dir(out_base)

    # Compute per-event summary metrics
    event_rows = []

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
            options=ind_opt,
        )
        if ts.empty:
            continue

        dhw_min = float(ts["dhw"].min())
        thw_min = _finite_min(ts["thw"])
        ttc_min = _finite_min(ts["ttc"])

        event_rows.append(
            {
                **asdict(ev),
                "dhw_min": dhw_min,
                "thw_min": thw_min,
                "ttc_min": ttc_min,
            }
        )

    metrics = pd.DataFrame(event_rows)
    if metrics.empty:
        print("Could not compute metrics for any cut-in event.")
        return

    metrics = metrics.sort_values(["ttc_min", "thw_min", "dhw_min"], ascending=[True, True, True]).reset_index(drop=True)

    metrics_path = out_base / "cutin_event_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved per-event metrics: {metrics_path}")

    # Select top-K most critical by TTC(min)
    top_k = min(int(args.top_k), len(metrics))
    top = metrics.head(top_k)

    print("\nTop events by minimum TTC:")
    for i, row in top.iterrows():
        print(
            f"#{i+1}: cutter={int(row['cutter_id'])}, follower={int(row['follower_id'])}, "
            f"{int(row['from_lane'])}->{int(row['to_lane'])}, "
            f"t_rel_start={float(row['relation_start_time']):.2f}s, "
            f"ttc_min={float(row['ttc_min']):.2f}s, thw_min={float(row['thw_min']):.2f}s, dhw_min={float(row['dhw_min']):.2f}m"
        )

    # Plot for each top event
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        cutter_id = int(row.cutter_id)
        follower_id = int(row.follower_id)
        to_lane = int(row.to_lane)

        center_frame = int(row.relation_start_frame)
        frames = range(max(1, center_frame - pre_frames), center_frame + post_frames + 1)

        ts = compute_pair_timeseries(
            indexed,
            leader_id=cutter_id,
            follower_id=follower_id,
            frames=frames,
            sign_map=sign_map,
            model=model,
            options=ind_opt,
        )
        if ts.empty:
            continue

        relation_start_time = float(row.relation_start_time)
        t_rel = ts["time"].to_numpy(dtype=float) - relation_start_time

        # For plotting, replace inf with NaN and optionally clip TTC
        dhw = ts["dhw"].to_numpy(dtype=float)
        thw = np.array([_safe_float(x) for x in ts["thw"].to_numpy(dtype=float)], dtype=float)
        ttc = np.array([_safe_float(x, clip=float(args.ttc_clip)) for x in ts["ttc"].to_numpy(dtype=float)], dtype=float)

        # Also plot laneId for cutter and follower as a sanity check
        cutter_state = _vehicle_state_timeseries(indexed, cutter_id, frames)
        follower_state = _vehicle_state_timeseries(indexed, follower_id, frames)

        # Align those to relation start
        cutter_t_rel = cutter_state["time"].to_numpy(dtype=float) - relation_start_time
        follower_t_rel = follower_state["time"].to_numpy(dtype=float) - relation_start_time

        prefix = f"rank{rank:02d}_cutter{cutter_id}_follower{follower_id}_{row.from_lane}to{row.to_lane}"

        title_base = (
            f"Cut-in candidate #{rank}: cutter {cutter_id} -> lane {to_lane}, follower {follower_id} | "
            f"TTC(min)={float(row.ttc_min):.2f}s, THW(min)={float(row.thw_min):.2f}s, DHW(min)={float(row.dhw_min):.2f}m"
        )

        _plot_series(
            x=t_rel,
            y=dhw,
            title=title_base + " | DHW",
            xlabel="time relative to relation start (s)",
            ylabel="DHW (m)",
            out_path=out_base / f"{prefix}_dhw.png",
            show=args.show,
        )

        _plot_series(
            x=t_rel,
            y=thw,
            title=title_base + " | THW",
            xlabel="time relative to relation start (s)",
            ylabel="THW (s)",
            out_path=out_base / f"{prefix}_thw.png",
            show=args.show,
        )

        _plot_series(
            x=t_rel,
            y=ttc,
            title=title_base + f" | TTC (clipped at {args.ttc_clip:.0f}s)",
            xlabel="time relative to relation start (s)",
            ylabel="TTC (s)",
            out_path=out_base / f"{prefix}_ttc.png",
            show=args.show,
        )

        # Lane sanity plot (two lines)
        plt.figure()
        plt.plot(cutter_t_rel, cutter_state["laneId"].to_numpy(dtype=float), label=f"cutter {cutter_id}")
        plt.plot(follower_t_rel, follower_state["laneId"].to_numpy(dtype=float), label=f"follower {follower_id}")
        plt.axvline(0.0, linestyle="--")
        plt.title(title_base + " | laneId sanity")
        plt.xlabel("time relative to relation start (s)")
        plt.ylabel("laneId")
        plt.legend()
        plt.tight_layout()
        lane_path = out_base / f"{prefix}_laneId.png"
        plt.savefig(lane_path, dpi=150)
        if args.show:
            plt.show()
        plt.close()

    print(f"\nSaved figures to: {out_base}")


if __name__ == "__main__":
    main()
