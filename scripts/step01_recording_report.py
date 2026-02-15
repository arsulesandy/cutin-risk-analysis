from __future__ import annotations

from pathlib import Path
from typing import List
import logging

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table, BuildOptions
from cutin_risk.preprocessing.quality_checks import (
    compute_basic_stats,
    check_duplicates_id_frame,
    check_time_monotonicity,
    sample_neighbor_id_integrity,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

SEPARATOR = "-" * 80


def process_recording(root: Path, rec_id: str) -> bool:
    try:
        rec = load_highd_recording(root, rec_id)
        df = build_tracking_table(
            rec, options=BuildOptions(keep_optional_tracks_columns=True)
        )

        stats = compute_basic_stats(df)
        logger.info("== Report for recording %s ==", rec.recording_id)
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

        neigh = sample_neighbor_id_integrity(df, sample_n=3000)
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
    root = Path(
        "/Users/sandeep/IdeaProjects/cutin-risk-analysis/data/raw/highD-dataset-v1.0/data"
    )

    failed: List[str] = []

    for i in range(1, 61):
        rec_id = f"{i:02d}"

        logger.info(SEPARATOR)
        logger.info("")

        if not process_recording(root, rec_id):
            failed.append(rec_id)

        logger.info("")

    logger.info(SEPARATOR)
    if failed:
        logger.warning("Recordings that failed to process: %s", failed)
    else:
        logger.info("All recordings processed successfully.")


if __name__ == "__main__":
    main()
