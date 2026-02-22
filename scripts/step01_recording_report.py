"""Step 01: per-recording schema and quality report for normalized highD tables."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
import logging

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.schema import RECORDING_META_SUFFIX
from cutin_risk.datasets.highd.transforms import build_tracking_table, BuildOptions
from cutin_risk.io.progress import iter_with_progress
from cutin_risk.paths import dataset_root_path, step_output_dir
from cutin_risk.preprocessing.quality_checks import (
    compute_basic_stats,
    check_duplicates_id_frame,
    check_time_monotonicity,
    sample_neighbor_id_integrity,
)
from cutin_risk.thesis_config import thesis_int

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

SEPARATOR = "-" * 80


def _all_recording_ids(root: Path) -> list[str]:
    pattern = f"*_{RECORDING_META_SUFFIX}.csv"
    ids = {
        (rid.zfill(2) if rid.isdigit() else rid)
        for p in root.glob(pattern)
        for rid in [p.stem.split("_", 1)[0]]
        if rid
    }
    return sorted(ids)


def process_recording(root: Path, rec_id: str) -> bool:
    try:
        rec = load_highd_recording(root, rec_id)
        df = build_tracking_table(
            rec, options=BuildOptions(keep_optional_tracks_columns=True)
        )

        stats = compute_basic_stats(df)
        logger.info("Step 01: Report for recording %s", rec.recording_id)
        logger.info("Rows: %d", stats.rows)
        logger.info("Vehicles: %d", stats.vehicles)
        logger.info("Time range: %.2fs .. %.2fs", stats.time_min, stats.time_max)
        logger.info("Lane IDs: %s", stats.lane_ids)
        logger.info("Optional columns present: %s", rec.tracks_schema_optional_present)

        logger.info("Duplicate (id, frame) rows: %d", check_duplicates_id_frame(df))
        logger.info(
            "Vehicles with non-monotonic time: %d",
            len(check_time_monotonicity(df)),
        )

        neigh = sample_neighbor_id_integrity(
            df,
            sample_n=thesis_int("step01.neighbor_integrity_sample_n", 3000, min_value=1),
        )
        if len(neigh) == 0:
            logger.info("Neighbor integrity: no neighbor columns found.")
        else:
            bad = neigh[~neigh["ok"]]
            logger.info(
                "Neighbor id integrity: %d/%d OK",
                int(neigh["ok"].sum()),
                len(neigh),
            )
            logger.info("Neighbor id integrity failures: %d", len(bad))
            if len(bad) > 0:
                logger.info("Examples:\n%s", bad.head(10).to_string(index=False))

        return True

    except Exception as exc:
        logger.exception("[%s] error while processing recording: %s", rec_id, exc)
        return False


def main() -> None:
    root = dataset_root_path()
    report_dir = step_output_dir(1, kind="reports")

    failed: List[str] = []
    succeeded = 0

    recordings = _all_recording_ids(root)
    if not recordings:
        raise FileNotFoundError(
            f"No recording metadata files found under {root} matching *_recordingMeta.csv"
        )

    for _, _, rec_id in iter_with_progress(
        recordings,
        label="Step 01 recordings",
        item_name="recording",
        emit=logger.info,
    ):

        logger.info(SEPARATOR)
        logger.info("")

        if not process_recording(root, rec_id):
            failed.append(rec_id)
        else:
            succeeded += 1

        logger.info("")

    logger.info(SEPARATOR)
    if failed:
        logger.warning("Recordings that failed to process: %s", failed)
    else:
        logger.info("All recordings processed successfully.")

    summary_path = report_dir / "recording_quality_summary.md"
    summary_lines = [
        "# Step 01 Recording Quality Summary",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset root: `{root}`",
        f"- Recordings attempted: {len(recordings)}",
        f"- Recordings processed successfully: {succeeded}",
        f"- Recordings failed: {len(failed)}",
        "",
        "## Failed recordings",
    ]
    if failed:
        summary_lines.extend([f"- `{rid}`" for rid in failed])
    else:
        summary_lines.append("- None")

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logger.info("Summary markdown written to %s", summary_path)


if __name__ == "__main__":
    main()
