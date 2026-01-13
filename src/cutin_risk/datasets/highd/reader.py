from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import (
    RECORDING_META_SUFFIX,
    TRACKS_META_SUFFIX,
    TRACKS_SUFFIX,
    REQUIRED_RECORDING_META_COLUMNS,
    REQUIRED_TRACKS_COLUMNS,
    REQUIRED_TRACKS_META_COLUMNS,
    OPTIONAL_TRACKS_COLUMNS,
    build_schema_report,
    require_schema,
)


@dataclass(frozen=True)
class HighDRecording:
    """
    Container for one highD recording.
    """
    recording_id: str
    recording_meta: pd.DataFrame
    tracks: pd.DataFrame
    tracks_meta: pd.DataFrame
    tracks_schema_optional_present: tuple[str, ...]


def _normalize_recording_id(recording_id: str) -> str:
    """
    Normalize recording identifiers to the canonical highD naming format.
    """
    rid = str(recording_id).strip()
    if rid.isdigit():
        return rid.zfill(2)
    return rid


def _file_path(root: Path, recording_id: str, suffix: str) -> Path:
    return root / f"{recording_id}_{suffix}.csv"


def load_highd_recording(dataset_root: str | Path, recording_id: str) -> HighDRecording:
    """
    Load one highD recording from a directory.

    Responsibilities:
      - file discovery for a given recording id
      - CSV loading
      - strict validation of required schema contracts

    Non-responsibilities:
      - time normalization
      - joining tracks with tracksMeta
      - filtering or cleaning
      - feature engineering
    """
    root = Path(dataset_root)
    rid = _normalize_recording_id(recording_id)

    tracks_path = _file_path(root, rid, TRACKS_SUFFIX)
    tracks_meta_path = _file_path(root, rid, TRACKS_META_SUFFIX)
    recording_meta_path = _file_path(root, rid, RECORDING_META_SUFFIX)

    for p in (tracks_path, tracks_meta_path, recording_meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    # Keep dtype inference default for now.
    # If needed later, an explicit dtype map can be introduced for performance and consistency.
    tracks = pd.read_csv(tracks_path)
    tracks_meta = pd.read_csv(tracks_meta_path)
    recording_meta = pd.read_csv(recording_meta_path)

    # Strict required schema validation
    require_schema(tracks, name=tracks_path.name, required=REQUIRED_TRACKS_COLUMNS)
    require_schema(tracks_meta, name=tracks_meta_path.name, required=REQUIRED_TRACKS_META_COLUMNS)
    require_schema(recording_meta, name=recording_meta_path.name, required=REQUIRED_RECORDING_META_COLUMNS)

    # Optional schema report for tracks.csv (informational only)
    tracks_report = build_schema_report(
        tracks,
        name=tracks_path.name,
        required=REQUIRED_TRACKS_COLUMNS,
        optional=OPTIONAL_TRACKS_COLUMNS,
    )

    return HighDRecording(
        recording_id=rid,
        recording_meta=recording_meta,
        tracks=tracks,
        tracks_meta=tracks_meta,
        tracks_schema_optional_present=tracks_report.optional_present,
    )
