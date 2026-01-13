from __future__ import annotations
from pathlib import Path


def ensure_artifact_dir(path: str | Path) -> Path:
    """Create the artifact directory if it does not exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
