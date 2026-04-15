"""CSV reader for one exiD recording."""

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
    require_schema,
)


@dataclass(frozen=True)
class ExiDRecording:
    """In-memory container with raw exiD tables for one recording."""

    recording_id: str
    recording_meta: pd.DataFrame
    tracks: pd.DataFrame
    tracks_meta: pd.DataFrame


def _normalize_recording_id(recording_id: str) -> str:
    rid = str(recording_id).strip()
    return rid.zfill(2) if rid.isdigit() else rid


def _file_path(root: Path, recording_id: str, suffix: str) -> Path:
    return root / f"{recording_id}_{suffix}.csv"


def load_exid_recording(dataset_root: str | Path, recording_id: str) -> ExiDRecording:
    """Load one exiD recording and validate the required raw schema."""

    root = Path(dataset_root)
    rid = _normalize_recording_id(recording_id)

    tracks_path = _file_path(root, rid, TRACKS_SUFFIX)
    tracks_meta_path = _file_path(root, rid, TRACKS_META_SUFFIX)
    recording_meta_path = _file_path(root, rid, RECORDING_META_SUFFIX)

    for path in (tracks_path, tracks_meta_path, recording_meta_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

    tracks = pd.read_csv(tracks_path, low_memory=False)
    tracks_meta = pd.read_csv(tracks_meta_path)
    recording_meta = pd.read_csv(recording_meta_path)

    require_schema(tracks, name=tracks_path.name, required=REQUIRED_TRACKS_COLUMNS)
    require_schema(tracks_meta, name=tracks_meta_path.name, required=REQUIRED_TRACKS_META_COLUMNS)
    require_schema(recording_meta, name=recording_meta_path.name, required=REQUIRED_RECORDING_META_COLUMNS)

    return ExiDRecording(
        recording_id=rid,
        recording_meta=recording_meta,
        tracks=tracks,
        tracks_meta=tracks_meta,
    )
