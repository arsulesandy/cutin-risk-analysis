from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def _normalize_recording_id(recording_id: str) -> str:
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
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (project_root() / p).resolve()


def configured_path(name: str, default: str | Path) -> Path:
    env_name = f"CUTIN_{name.upper()}"
    env_value = os.environ.get(env_name)
    if env_value and env_value.strip():
        return _resolve_to_project(env_value.strip())

    cfg_value = _value_from_config(name)
    if cfg_value is not None:
        return _resolve_to_project(cfg_value)

    return _resolve_to_project(default)


def dataset_root_path() -> Path:
    return configured_path("dataset_root", "data/raw/highD-dataset-v1.0/data")


def outputs_root_path() -> Path:
    return configured_path("outputs_root", "outputs")


def output_path(relative_path: str | Path) -> Path:
    return (outputs_root_path() / Path(relative_path)).resolve()


def step14_codes_csv_path() -> Path:
    return configured_path(
        "step14_codes_csv",
        output_path("reports/step14_sfc_binary/sfc_binary_codes_long_hilbert.csv"),
    )


def highd_tracks_csv(recording_id: str = "01") -> Path:
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}_tracks.csv"


def highd_tracks_meta_csv(recording_id: str = "01") -> Path:
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}_tracksMeta.csv"


def highd_recording_meta_csv(recording_id: str = "01") -> Path:
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}_recordingMeta.csv"


def highd_pickle_path(recording_id: str = "01") -> Path:
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}.pickle"


def highd_background_image(recording_id: str = "01") -> Path:
    rid = _normalize_recording_id(recording_id)
    return dataset_root_path() / f"{rid}_highway.png"
