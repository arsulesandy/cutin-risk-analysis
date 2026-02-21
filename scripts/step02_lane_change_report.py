from __future__ import annotations

from collections import Counter

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.paths import dataset_root_path


def main() -> None:
    root = dataset_root_path()
    rec = load_highd_recording(root, "01")
    df = build_tracking_table(rec)

    events = detect_lane_changes(
        df,
        options=LaneChangeOptions(min_stable_before_frames=25, min_stable_after_frames=25),
    )


    print("Step 02: Lane Change Report")
    print("Recording:", rec.recording_id)
    print("Vehicles:", df["id"].nunique())
    print("Lane change events:", len(events))

    counts = Counter((e.from_lane, e.to_lane) for e in events)
    print("\nTop transitions:")
    for (a, b), c in counts.most_common(10):
        print(f"  {a} -> {b}: {c}")

    print("\nAll lane change events:")
    for e in events:
        print(e)


if __name__ == "__main__":
    main()
