from __future__ import annotations

from pathlib import Path
import math
import statistics as stats

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
from cutin_risk.indicators.surrogate_safety import (
    LongitudinalModel,
    IndicatorOptions,
    infer_direction_sign_map,
    compute_pair_timeseries,
    validate_against_dataset_preceding,
)


def main() -> None:
    root = Path("/Users/sandeep/IdeaProjects/cutin-risk-analysis/data/raw/highD-dataset-v1.0/data")

    rec = load_highd_recording(root, "01")
    df = build_tracking_table(rec)

    # --- (Optional) validate indicator model against dataset-provided preceding metrics ---
    # Try "center" first; if error looks bad, try "rear" and compare.
    model_center = LongitudinalModel(position_reference="center")
    model_rear = LongitudinalModel(position_reference="rear")
    ind_opt = IndicatorOptions()

    print("== Step 4: Indicator sanity check against dataset (precedingId pair) ==")
    try:
        report_center = validate_against_dataset_preceding(df, model=model_center, options=ind_opt)
        report_rear = validate_against_dataset_preceding(df, model=model_rear, options=ind_opt)
        print("Model=center:", report_center)
        print("Model=rear:  ", report_rear)
        # Choose the better DHW median error model as default
        model = model_center if report_center["dhw_median_abs_error"] <= report_rear["dhw_median_abs_error"] else model_rear
        print("Chosen position_reference:", model.position_reference)
    except Exception as e:
        # If optional columns were dropped, validation won't run; that's fine.
        print("Validation skipped:", str(e))
        model = model_center

    # --- detect events ---
    lane_changes = detect_lane_changes(
        df,
        options=LaneChangeOptions(min_stable_before_frames=25, min_stable_after_frames=25),
    )
    cutins = detect_cutins(
        df,
        lane_changes,
        options=CutInOptions(search_window_frames=50, min_relation_frames=10),
    )

    print("\n== Step 4: Risk metrics on cut-in pairs ==")
    print("Recording:", rec.recording_id)
    print("Lane changes:", len(lane_changes))
    print("Cut-ins:", len(cutins))

    sign_map = infer_direction_sign_map(df)
    indexed = df.set_index(["id", "frame"], drop=False)

    # window around the insertion moment (relation_start_frame)
    pre_frames = 50   # 2 seconds
    post_frames = 75  # 3 seconds

    event_summaries = []

    for ev in cutins:
        center_f = int(ev.relation_start_frame)
        frames = range(max(1, center_f - pre_frames), center_f + post_frames + 1)

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

        # Event-level minima (ignore negative gaps for TTC/THW)
        dhw_min = float(ts["dhw"].min())

        finite_ttc = [x for x in ts["ttc"].tolist() if math.isfinite(x)]
        finite_thw = [x for x in ts["thw"].tolist() if math.isfinite(x)]

        ttc_min = min(finite_ttc) if finite_ttc else float("inf")
        thw_min = min(finite_thw) if finite_thw else float("inf")

        event_summaries.append(
            {
                "cutter_id": int(ev.cutter_id),
                "follower_id": int(ev.follower_id),
                "from_lane": int(ev.from_lane),
                "to_lane": int(ev.to_lane),
                "relation_start_time": float(ev.relation_start_time),
                "dhw_min": dhw_min,
                "thw_min": thw_min,
                "ttc_min": ttc_min,
            }
        )

    print("Events with computed metrics:", len(event_summaries))
    if not event_summaries:
        return

    # Sort by min TTC (risky first)
    event_summaries.sort(key=lambda r: r["ttc_min"])

    print("\nAll events by minimum TTC (smaller = more critical):")
    for r in event_summaries:
        print(r)

    ttc_vals = [r["ttc_min"] for r in event_summaries if math.isfinite(r["ttc_min"])]
    if ttc_vals:
        print("\nTTC(min) summary (finite only):")
        print("  min:", min(ttc_vals))
        print("  median:", stats.median(ttc_vals))
        print("  p25:", stats.quantiles(ttc_vals, n=4)[0])
        print("  p75:", stats.quantiles(ttc_vals, n=4)[2])


if __name__ == "__main__":
    main()
