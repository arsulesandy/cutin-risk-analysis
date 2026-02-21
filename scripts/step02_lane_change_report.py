"""Step 02: lane-change detection report with overall dataset totals."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.paths import dataset_root_path


def _all_recording_ids(root: Path) -> list[str]:
    pattern = f"*_{RECORDING_META_SUFFIX}.csv"
    ids = {
        (rid.zfill(2) if rid.isdigit() else rid)
        for p in root.glob(pattern)
        for rid in [p.stem.split("_", 1)[0]]
        if rid
    }
    return sorted(ids)


def main() -> None:
    root = dataset_root_path()
    lane_change_options = LaneChangeOptions(min_stable_before_frames=25, min_stable_after_frames=25)

    rec = load_highd_recording(root, "01")
    df = build_tracking_table(rec)

    events = detect_lane_changes(
        df,
        options=lane_change_options,
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

    all_recording_ids = _all_recording_ids(root) or [rec.recording_id]
    total_lane_changes = 0
    failed_recordings: list[str] = []
    for rid in all_recording_ids:
        try:
            rec_all = load_highd_recording(root, rid)
            df_all = build_tracking_table(rec_all)
            all_events = detect_lane_changes(df_all, options=lane_change_options)
            total_lane_changes += len(all_events)
        except Exception:
            failed_recordings.append(rid)

    print("\nOverall stats (all recordings):")
    print("Recordings discovered:", len(all_recording_ids))
    print("Recordings processed:", len(all_recording_ids) - len(failed_recordings))
    print("Total lane change events:", total_lane_changes)
    if failed_recordings:
        print("Recordings failed:", ", ".join(failed_recordings))


if __name__ == "__main__":
    main()
