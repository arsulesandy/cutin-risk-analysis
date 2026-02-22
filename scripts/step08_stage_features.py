"""Step 08: extract stage-wise event features for downstream risk modeling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import pandas as pd

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.reconstruction.lanes import parse_lane_markings, infer_lane_index
from cutin_risk.reconstruction.neighbors import (
    reconstruct_same_lane_neighbors,
    NeighborReconstructionOptions,
)
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
from cutin_risk.indicators.surrogate_safety import (
    LongitudinalModel,
    IndicatorOptions,
    infer_direction_sign_map,
    compute_pair_timeseries,
)
from cutin_risk.io.step_reports import (
    mirror_file_to_step,
    step_figures_dir,
    step_reports_dir,
    write_step_markdown,
)
from cutin_risk.paths import dataset_root_path, output_path
from cutin_risk.thesis_config import thesis_bool, thesis_float, thesis_str


@dataclass(frozen=True)
class Stage:
    name: str
    start_s: float  # inclusive
    end_s: float    # exclusive


def _finite_min(values: pd.Series, *, lower_bound: float | None = None) -> float:
    vals: list[float] = []
    for x in values.tolist():
        fx = float(x)
        if not math.isfinite(fx):
            continue
        if lower_bound is not None:
            fx = max(fx, float(lower_bound))
        vals.append(fx)
    return min(vals) if vals else float("inf")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _remove_legacy_timestamped_outputs(report_dir: Path, fig_dir: Path) -> None:
    for p in report_dir.glob("cutin_stage_features_*.csv"):
        p.unlink(missing_ok=True)
    for p in fig_dir.glob("median_ttc_curve_*.png"):
        p.unlink(missing_ok=True)


def _vehicle_slice(
        indexed: pd.DataFrame,
        vehicle_id: int,
        start_frame: int,
        end_frame: int,
        cols: list[str],
) -> pd.DataFrame:
    try:
        g = indexed.loc[(vehicle_id, slice(start_frame, end_frame)), cols]
    except KeyError:
        return pd.DataFrame(columns=["frame", "time", *cols])

    g = g.reset_index(drop=True)
    # Ensure frame/time exist (they should, since we keep drop=False in indexing)
    return g


def _rename_state(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    keep = ["frame", "time"]
    other_cols = [c for c in df.columns if c not in keep]
    rename_map = {c: f"{prefix}_{c}" for c in other_cols}
    return df.rename(columns=rename_map)


def _stage_summary(joined: pd.DataFrame, stage: Stage) -> dict[str, float]:
    m = (joined["t_rel"] >= stage.start_s) & (joined["t_rel"] < stage.end_s)
    seg = joined.loc[m].copy()
    if seg.empty:
        return {
            f"{stage.name}_rows": 0.0,
            f"{stage.name}_dhw_min": float("nan"),
            f"{stage.name}_thw_min": float("nan"),
            f"{stage.name}_ttc_min": float("nan"),
            f"{stage.name}_cutter_lat_v_abs_max": float("nan"),
            f"{stage.name}_cutter_speed_mean": float("nan"),
            f"{stage.name}_cutter_acc_mean": float("nan"),
            f"{stage.name}_follower_speed_mean": float("nan"),
            f"{stage.name}_follower_acc_mean": float("nan"),
            f"{stage.name}_cutter_dy": float("nan"),
        }

    # Pair metrics
    dhw_min = float(seg["dhw"].clip(lower=0.0).min())
    thw_min = _finite_min(seg["thw"], lower_bound=0.0)
    ttc_min = _finite_min(seg["ttc"])

    # Lateral motion (cutter)
    cutter_lat_v_abs_max = float(np.nanmax(np.abs(seg["cutter_yVelocity"].to_numpy(dtype=float))))

    # Longitudinal motion (use speed column if present)
    cutter_speed_mean = float(np.nanmean(seg["cutter_speed"].to_numpy(dtype=float)))
    follower_speed_mean = float(np.nanmean(seg["follower_speed"].to_numpy(dtype=float)))

    cutter_acc_mean = float(np.nanmean(seg["cutter_xAcceleration"].to_numpy(dtype=float)))
    follower_acc_mean = float(np.nanmean(seg["follower_xAcceleration"].to_numpy(dtype=float)))

    # Lateral displacement (cutter y)
    y_vals = seg["cutter_y"].to_numpy(dtype=float)
    cutter_dy = float(y_vals[-1] - y_vals[0]) if len(y_vals) >= 2 else 0.0

    return {
        f"{stage.name}_rows": float(len(seg)),
        f"{stage.name}_dhw_min": dhw_min,
        f"{stage.name}_thw_min": thw_min,
        f"{stage.name}_ttc_min": ttc_min,
        f"{stage.name}_cutter_lat_v_abs_max": cutter_lat_v_abs_max,
        f"{stage.name}_cutter_speed_mean": cutter_speed_mean,
        f"{stage.name}_cutter_acc_mean": cutter_acc_mean,
        f"{stage.name}_follower_speed_mean": follower_speed_mean,
        f"{stage.name}_follower_acc_mean": follower_acc_mean,
        f"{stage.name}_cutter_dy": cutter_dy,
    }


def _accumulate_by_offset(acc: dict[int, list[float]], offsets: np.ndarray, values: np.ndarray) -> None:
    for off, v in zip(offsets.tolist(), values.tolist()):
        fv = float(v)
        if not math.isfinite(fv):
            continue
        acc.setdefault(int(off), []).append(fv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: stage-based analysis around cut-in events.")
    parser.add_argument("--dataset-root", type=str, default=str(dataset_root_path()))
    parser.add_argument("--recording-id", type=str, default=thesis_str("step08.recording_id", "01"))

    parser.add_argument("--pre-seconds", type=float, default=thesis_float("step08.pre_seconds", 4.0, min_value=0.0))
    parser.add_argument("--post-seconds", type=float, default=thesis_float("step08.post_seconds", 4.0, min_value=0.0))

    parser.add_argument(
        "--make-plot",
        action=argparse.BooleanOptionalAction,
        help="Save median TTC curve aligned at t0",
        default=thesis_bool("step08.make_plot", True),
    )
    parser.add_argument("--out-dir", type=str, default=str(output_path(".")))

    args = parser.parse_args()

    # Load + build tracking table
    rec = load_highd_recording(Path(args.dataset_root), args.recording_id)
    df = build_tracking_table(rec)

    frame_rate = float(rec.recording_meta.loc[0, "frameRate"])
    pre_frames = int(round(args.pre_seconds * frame_rate))
    post_frames = int(round(args.post_seconds * frame_rate))

    # --- Minimal-input path: infer lane index from y + lane markings ---
    markings = parse_lane_markings(rec.recording_meta)
    df = df.join(infer_lane_index(df, markings))  # laneIndex_xy

    # --- Reconstruct neighbors using inferred lanes ---
    df = df.join(
        reconstruct_same_lane_neighbors(
            df,
            options=NeighborReconstructionOptions(
                lane_col="laneIndex_xy",
                out_preceding_col="precedingId_xy_lane",
                out_following_col="followingId_xy_lane",
                ignore_lane_ids=(0,),
                no_neighbor_id=0,
            ),
        )
    )

    # Detect lane changes + cut-ins using inferred lanes and reconstructed neighbors
    lane_changes = detect_lane_changes(df, options=LaneChangeOptions(lane_col="laneIndex_xy"))
    cutins = detect_cutins(
        df,
        lane_changes,
        options=CutInOptions(
            lane_col="laneIndex_xy",
            following_col="followingId_xy_lane",
            preceding_col="precedingId_xy_lane",
        ),
    )

    print("== Step 8: Stage features report ==")
    print("Recording:", rec.recording_id)
    print("Lane changes:", len(lane_changes))
    print("Cut-ins:", len(cutins))
    if not cutins:
        print("No cut-ins found; nothing to analyze.")
        return

    # Prepare indexed view for fast access
    indexed = df.set_index(["id", "frame"], drop=False).sort_index()

    # Pair metrics setup
    sign_map = infer_direction_sign_map(df)
    model = LongitudinalModel(
        position_reference=thesis_str(
            "indicators.position_reference",
            "bbox_topleft",
            allowed={"center", "rear", "bbox_topleft"},
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

    # Stage definitions (3-stage model, plus optional recovery in the full window)
    stages = [
        Stage(
            "intention",
            thesis_float("step08.stages.intention.start_s", -4.0),
            thesis_float("step08.stages.intention.end_s", -2.0),
        ),
        Stage(
            "decision",
            thesis_float("step08.stages.decision.start_s", -2.0),
            thesis_float("step08.stages.decision.end_s", 0.0),
        ),
        Stage(
            "execution",
            thesis_float("step08.stages.execution.start_s", 0.0),
            thesis_float("step08.stages.execution.end_s", 2.0),
        ),
        Stage(
            "recovery",
            thesis_float("step08.stages.recovery.start_s", 2.0),
            thesis_float("step08.stages.recovery.end_s", 4.0),
        ),
    ]

    out_base = Path(args.out_dir)
    report_dir = out_base / "reports" / f"step8_recording_{rec.recording_id}"
    fig_dir = out_base / "figures" / f"step8_recording_{rec.recording_id}"
    _ensure_dir(report_dir)
    _remove_legacy_timestamped_outputs(report_dir, fig_dir)

    rows: list[dict[str, float | int | str]] = []

    # For optional “median TTC curve” plot
    ttc_by_offset: dict[int, list[float]] = {}

    for ev in cutins:
        t0_frame = int(ev.relation_start_frame)
        start_frame = max(1, t0_frame - pre_frames)
        end_frame = t0_frame + post_frames
        frames = range(start_frame, end_frame + 1)

        # Pair metrics (cutter is leader, follower behind)
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

        # Cutter + follower state slices
        state_cols = [
            "frame",
            "time",
            "x",
            "y",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "speed",
        ]
        cutter_state = _vehicle_slice(indexed, int(ev.cutter_id), start_frame, end_frame, state_cols)
        follower_state = _vehicle_slice(indexed, int(ev.follower_id), start_frame, end_frame, state_cols)

        if cutter_state.empty or follower_state.empty:
            continue

        cutter_state = _rename_state(cutter_state, "cutter")
        follower_state = _rename_state(follower_state, "follower")

        # Merge everything on frame (time is derivable from frame, so frame is enough)
        joined = ts.merge(cutter_state, on="frame", how="left").merge(follower_state, on="frame", how="left")

        # Relative time around the cut-in moment
        t0_time = float(ev.relation_start_time)
        joined["t_rel"] = joined["time"].astype(float) - t0_time

        # Some event-level minima over full window
        dhw_min_total = float(joined["dhw"].clip(lower=0.0).min())
        thw_min_total = _finite_min(joined["thw"], lower_bound=0.0)
        ttc_min_total = _finite_min(joined["ttc"])

        out = {
            "recording_id": str(rec.recording_id),
            "cutter_id": int(ev.cutter_id),
            "follower_id": int(ev.follower_id),
            "from_lane": int(ev.from_lane),
            "to_lane": int(ev.to_lane),
            "t0_frame": int(t0_frame),
            "t0_time": float(t0_time),
            "dhw_min_total": float(dhw_min_total),
            "thw_min_total": float(thw_min_total),
            "ttc_min_total": float(ttc_min_total),
        }

        # Stage summaries
        for st in stages:
            out.update(_stage_summary(joined, st))

        rows.append(out)

        if args.make_plot:
            offsets = (joined["frame"].to_numpy(dtype=int) - t0_frame).astype(int)
            ttc_vals = joined["ttc"].to_numpy(dtype=float)
            _accumulate_by_offset(ttc_by_offset, offsets, ttc_vals)

    features = pd.DataFrame(rows)
    if features.empty:
        print("No usable events for stage analysis (missing data slices).")
        return

    out_csv = report_dir / "cutin_stage_features.csv"
    features.to_csv(out_csv, index=False)
    canonical_subdir = f"recording_{rec.recording_id}"
    canonical_report_dir = step_reports_dir(8, subdir=canonical_subdir)
    canonical_fig_dir = step_figures_dir(8, subdir=canonical_subdir, create=False)
    for p in canonical_report_dir.glob("cutin_stage_features_*.csv"):
        p.unlink(missing_ok=True)
    if canonical_fig_dir.exists():
        for p in canonical_fig_dir.glob("median_ttc_curve_*.png"):
            p.unlink(missing_ok=True)
    canonical_csv = mirror_file_to_step(out_csv, 8, subdir=canonical_subdir)
    print(f"\nSaved stage features: {out_csv}")
    print(f"Mirrored stage features: {canonical_csv}")

    # Quick summary for meeting
    finite_ttc = features["ttc_min_total"].replace([np.inf, -np.inf], np.nan).dropna()
    print("\nTTC(min) total summary (finite only):")
    print("  events:", int(len(features)))
    print("  min:", float(finite_ttc.min()))
    print("  median:", float(finite_ttc.median()))
    print("  p25:", float(finite_ttc.quantile(0.25)))
    print("  p75:", float(finite_ttc.quantile(0.75)))

    # Optional median TTC curve plot
    out_png: Path | None = None
    if args.make_plot and ttc_by_offset:
        try:
            import matplotlib.pyplot as plt  # lazy import: only needed for plotting mode
        except Exception as exc:
            print(f"[WARN] Unable to generate plot: {exc}")
        else:
            offsets_sorted = np.array(sorted(ttc_by_offset.keys()), dtype=int)
            med = np.array([float(np.median(ttc_by_offset[o])) for o in offsets_sorted], dtype=float)
            t_rel_s = offsets_sorted / frame_rate

            plt.figure()
            plt.plot(t_rel_s, med)
            plt.axvline(0.0, linestyle="--")
            plt.title(f"Median TTC aligned at cut-in (recording {rec.recording_id})")
            plt.xlabel("time relative to cut-in start (s)")
            plt.ylabel("TTC (s)")
            plt.tight_layout()
            fig_dir.mkdir(parents=True, exist_ok=True)
            out_png = fig_dir / "median_ttc_curve.png"
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"Saved plot: {out_png}")
            canonical_png = mirror_file_to_step(out_png, 8, kind="figures", subdir=canonical_subdir)
            print(f"Mirrored plot: {canonical_png}")

    details_md = write_step_markdown(
        8,
        "stage_features_details.md",
        [
            "# Step 08 Stage Features Report",
            "",
            f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
            f"- Recording: `{rec.recording_id}`",
            f"- Dataset root: `{Path(args.dataset_root).resolve()}`",
            f"- Lane changes: `{len(lane_changes)}`",
            f"- Cut-ins: `{len(cutins)}`",
            f"- Stage events exported: `{len(features)}`",
            f"- Stage features CSV: `{canonical_csv}`",
            f"- Median TTC figure: `{out_png.name if out_png is not None else 'not generated'}`",
        ],
        subdir=canonical_subdir,
    )
    print(f"Saved summary markdown: {details_md}")


if __name__ == "__main__":
    main()
