"""Artifact-directory helpers for report and model outputs."""

from __future__ import annotations
from pathlib import Path


def ensure_artifact_dir(path: str | Path) -> Path:
    """Create (or reuse) an artifact directory and return its resolved path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
