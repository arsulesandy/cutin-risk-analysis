"""Path resolution helpers used by scripts and library entry points.

Configuration precedence for each named path is:
1. Environment variable `CUTIN_<NAME>`
2. JSON configuration file value (`configs/paths.local.json` preferred)
3. Built-in default

All relative paths are resolved against the project root.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


def _normalize_recording_id(recording_id: str) -> str:
    """Normalize highD recording ids to two digits where possible."""
    rid = str(recording_id).strip()
    return f"{int(rid):02d}" if rid.isdigit() else rid


@lru_cache(maxsize=1)
def project_root() -> Path:
    """
    Resolve repository root by walking up from this file until pyproject.toml is found.
    Falls back to current working directory if not found.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def _config_file_path() -> Path:
    """Resolve the active path configuration file."""
    env_path = os.environ.get("CUTIN_PATHS_FILE")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.is_absolute() else project_root() / p

    local_cfg = project_root() / "configs" / "paths.local.json"
    if local_cfg.exists():
        return local_cfg

    return project_root() / "configs" / "paths.json"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    """Load JSON config once per process.

    Returns an empty dict when no config file exists or is empty.
    """
    cfg_path = _config_file_path()
    if not cfg_path.exists():
        return {}

    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {cfg_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config file {cfg_path} must contain a JSON object.")

    return data


def _value_from_config(name: str) -> str | None:
    """Read a path-like value from the loaded config structure."""
    cfg = _load_config()

    paths_block = cfg.get("paths")
    if isinstance(paths_block, dict):
        value = paths_block.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    value = cfg.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _resolve_to_project(path_like: str | Path) -> Path:
    """Resolve path to absolute location relative to project root when needed."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (project_root() / p).resolve()


def configured_path(name: str, default: str | Path) -> Path:
    """Resolve a named configured path with env > config > default precedence."""
    env_name = f"CUTIN_{name.upper()}"
    env_value = os.environ.get(env_name)
    if env_value and env_value.strip():
        return _resolve_to_project(env_value.strip())

    cfg_value = _value_from_config(name)
    if cfg_value is not None:
        return _resolve_to_project(cfg_value)

    return _resolve_to_project(default)


def dataset_root_path() -> Path:
    """Return the configured root directory containing highD CSV files."""
    return configured_path("dataset_root", "data/raw/highD-dataset-v1.0/data")


def outputs_root_path() -> Path:
    """Return the configured root directory for generated outputs."""
    return configured_path("outputs_root", "outputs")


def output_path(relative_path: str | Path) -> Path:
    """Resolve a path under `outputs_root_path`."""
    return (outputs_root_path() / Path(relative_path)).resolve()


def step_display_name(step: int | str) -> str:
    """Normalize step identifiers into canonical display form, e.g. ``Step 02``."""
    token = str(step).strip()
    if not token:
        raise ValueError("step must not be empty")

    if token.lower().startswith("step"):
        token = token[4:].strip()

    compact = token.replace(" ", "").replace("_", "")
    match = re.fullmatch(r"(\d+)([A-Za-z]?)", compact)
    if match is None:
        raise ValueError(f"Unsupported step identifier: {step!r}")

    number = int(match.group(1))
    suffix = match.group(2).upper()
    return f"Step {number:02d}{suffix}"


def step_output_dir(step: int | str, *, kind: str = "reports", create: bool = True) -> Path:
    """
    Resolve a step-specific output directory under ``outputs/reports`` or ``outputs/figures``.

    Examples
    --------
    - ``step_output_dir(2, kind="reports")`` -> ``.../outputs/reports/Step 02``
    - ``step_output_dir("03", kind="figures")`` -> ``.../outputs/figures/Step 03``
    """
    kind_norm = kind.strip().lower()
    if kind_norm not in {"reports", "figures"}:
        raise ValueError(f"kind must be 'reports' or 'figures', got: {kind!r}")

    target = output_path(Path(kind_norm) / step_display_name(step))
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def step14_codes_csv_path() -> Path:
    """Return path to Step-14 binary SFC code table."""
    return configured_path(
        "step14_codes_csv",
        output_path("reports/Step 14/sfc_binary_codes_long_hilbert.csv"),
    )


def highd_tracks_csv(recording_id: str = "01") -> Path:
    """Return `<recording>_tracks.csv` path for a highD recording."""
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}_tracks.csv"


def highd_tracks_meta_csv(recording_id: str = "01") -> Path:
    """Return `<recording>_tracksMeta.csv` path for a highD recording."""
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}_tracksMeta.csv"


def highd_recording_meta_csv(recording_id: str = "01") -> Path:
    """Return `<recording>_recordingMeta.csv` path for a highD recording."""
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}_recordingMeta.csv"


def highd_pickle_path(recording_id: str = "01") -> Path:
    """Return `<recording>.pickle` path under dataset root."""
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}.pickle"


def highd_background_image(recording_id: str = "01") -> Path:
    """Return `<recording>_highway.png` background image path."""
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}_highway.png"
