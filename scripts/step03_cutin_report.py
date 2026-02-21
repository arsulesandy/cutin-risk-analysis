"""Step 03: cut-in detection report built on lane-change events."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics as stats

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.detection.lane_change import detect_lane_changes, LaneChangeOptions
from cutin_risk.detection.cutin import detect_cutins, CutInOptions
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
    cutin_options = CutInOptions(search_window_frames=50, min_relation_frames=10)

    rec = load_highd_recording(root, "01")
    df = build_tracking_table(rec)

    lane_changes = detect_lane_changes(
        df,
        options=lane_change_options,
    )

    cutins = detect_cutins(
        df,
        lane_changes,
        options=cutin_options,
    )

    print("Step 03: Cut-in Report")
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

    print("\nAll lane cut-ins:")
    for e in cutins:
        print(e)

    all_recording_ids = _all_recording_ids(root) or [rec.recording_id]
    total_lane_changes = 0
    total_cutins = 0
    failed_recordings: list[str] = []
    for rid in all_recording_ids:
        try:
            rec_all = load_highd_recording(root, rid)
            df_all = build_tracking_table(rec_all)

            all_lane_changes = detect_lane_changes(df_all, options=lane_change_options)
            all_cutins = detect_cutins(df_all, all_lane_changes, options=cutin_options)

            total_lane_changes += len(all_lane_changes)
            total_cutins += len(all_cutins)
        except Exception:
            failed_recordings.append(rid)

    print("\nOverall stats (all recordings):")
    print("Recordings discovered:", len(all_recording_ids))
    print("Recordings processed:", len(all_recording_ids) - len(failed_recordings))
    print("Total lane changes:", total_lane_changes)
    print("Total cut-ins:", total_cutins)
    if total_lane_changes:
        print(f"Overall cut-in ratio: {total_cutins / total_lane_changes:.2%}")
    if failed_recordings:
        print("Recordings failed:", ", ".join(failed_recordings))


if __name__ == "__main__":
    main()
