"""Quality-check helpers used during dataset sanity reporting."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


NEIGHBOR_COLS = [
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftAlongsideId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightAlongsideId",
    "rightFollowingId",
]


@dataclass(frozen=True)
class BasicStats:
    """Compact summary of one normalized recording table."""
    rows: int
    vehicles: int
    time_min: float
    time_max: float
    lane_ids: tuple[int, ...]
    missing_class_rows: int


def compute_basic_stats(df: pd.DataFrame) -> BasicStats:
    """Compute recording-level statistics used in step-level reports."""
    lane_ids = tuple(sorted(int(x) for x in df["laneId"].dropna().unique().tolist()))
    return BasicStats(
        rows=int(len(df)),
        vehicles=int(df["id"].nunique()),
        time_min=float(df["time"].min()),
        time_max=float(df["time"].max()),
        lane_ids=lane_ids,
        missing_class_rows=int(df["class"].isna().sum()) if "class" in df.columns else int(len(df)),
    )


def check_duplicates_id_frame(df: pd.DataFrame) -> int:
    """Count duplicate `(id, frame)` keys."""
    return int(df.duplicated(["id", "frame"]).sum())


def check_time_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of vehicle ids where time decreases within that vehicle's trajectory.
    """
    bad = []
    for vid, g in df.groupby("id", sort=False):
        t = g["time"].to_numpy()
        if (t[1:] < t[:-1]).any():
            bad.append(int(vid))
    return pd.DataFrame({"vehicle_id": bad})


def sample_neighbor_id_integrity(df: pd.DataFrame, sample_n: int = 2000, random_state: int = 7) -> pd.DataFrame:
    """
    Sample rows and verify that neighbor ids are either:
      - a sentinel meaning "no neighbor" (commonly -1 or 0)
      - or exist as a vehicle id in this recording
    """
    no_neighbor_ids = {-1, 0}

    present_ids = set(int(x) for x in df["id"].unique().tolist())
    sample = df.sample(n=min(sample_n, len(df)), random_state=random_state).copy()

    rows = []
    for _, row in sample.iterrows():
        ego = int(row["id"])
        frame = int(row["frame"])
        for col in NEIGHBOR_COLS:
            if col not in df.columns:
                continue
            n_id = int(row[col])
            ok = (n_id in no_neighbor_ids) or (n_id in present_ids)
            rows.append(
                {
                    "frame": frame,
                    "ego_id": ego,
                    "neighbor_col": col,
                    "neighbor_id": n_id,
                    "ok": ok,
                }
            )
    return pd.DataFrame(rows)
