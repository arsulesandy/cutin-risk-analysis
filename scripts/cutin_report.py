from __future__ import annotations

from pathlib import Path
from collections import Counter
import statistics as stats

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.detection.cutin import detect_cutins, CutInOptions


def main() -> None:
    root = Path("/Users/sandeep/IdeaProjects/cutin-risk-analysis/data/raw/highD-dataset-v1.0/data")

    rec = load_highd_recording(root, "01")
    df = build_tracking_table(rec)

    lane_changes = detect_lane_changes(
        df,
        options=LaneChangeOptions(min_stable_before_frames=25, min_stable_after_frames=25),
    )

    cutins = detect_cutins(
        df,
        lane_changes,
        options=CutInOptions(search_window_frames=50, min_relation_frames=10),
    )

    print("== Cut-in Report ==")
    print("Recording:", rec.recording_id)
    print("Vehicles:", df["id"].nunique())
    print("Lane changes:", len(lane_changes))
    print("Cut-ins:", len(cutins))
    if lane_changes:
        print(f"Cut-in ratio: {len(cutins) / len(lane_changes):.2%}")

    transition_counts = Counter((e.from_lane, e.to_lane) for e in cutins)
    print("\nTop cut-in transitions:")
    for (a, b), c in transition_counts.most_common(10):
        print(f"  {a} -> {b}: {c}")

    durations = [(e.relation_end_frame - e.relation_start_frame + 1) for e in cutins]
    if durations:
        print("\nFollower-relation duration (frames):")
        print("  min:", min(durations))
        print("  median:", int(stats.median(durations)))
        print("  max:", max(durations))

    print("\nFirst 10 cut-ins:")
    for e in cutins[:10]:
        print(e)


if __name__ == "__main__":
    main()
