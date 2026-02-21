"""Detection configuration helpers for thesis-sensitive defaults.

Precedence:
1. Environment variable `CUTIN_DETECTION_CONFIG_FILE`
2. `configs/detection.local.json`
3. `configs/detection.json`
4. Built-in defaults in code
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

from cutin_risk.paths import project_root

_DEFAULTS = {
    "lane_change": {
        "min_stable_before_frames": 25,
        "min_stable_after_frames": 25,
        "ignore_lane_ids": [0],
    },
    "cutin": {
        "search_window_frames": 50,
        "start_offset_frames": 0,
        "max_relation_delay_frames": 15,
        "min_relation_frames": 10,
        "require_new_follower": True,
        "precheck_frames": 25,
        "no_neighbor_ids": [0, -1],
        "require_lane_match": True,
        "require_preceding_consistency": True,
    },
}


def _config_file_path() -> Path:
    env_path = os.environ.get("CUTIN_DETECTION_CONFIG_FILE")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.is_absolute() else project_root() / p

    local_cfg = project_root() / "configs" / "detection.local.json"
    if local_cfg.exists():
        return local_cfg

    return project_root() / "configs" / "detection.json"


@lru_cache(maxsize=1)
def _load_detection_config() -> dict[str, Any]:
    cfg_path = _config_file_path()
    if not cfg_path.exists():
        return {}

    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in detection config file {cfg_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Detection config file {cfg_path} must contain a JSON object.")

    return data


def _section(name: str) -> dict[str, Any]:
    cfg = _load_detection_config()

    detection_block = cfg.get("detection")
    if isinstance(detection_block, dict):
        nested = detection_block.get(name)
        if isinstance(nested, dict):
            return nested

    direct = cfg.get(name)
    if isinstance(direct, dict):
        return direct

    return {}


def _as_non_negative_int(
    *,
    section: str,
    key: str,
    default: int,
) -> int:
    block = _section(section)
    raw = block.get(key, default)

    if isinstance(raw, bool):
        raise ValueError(f"detection.{section}.{key} must be an integer, not boolean")

    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        token = raw.strip()
        if not token:
            value = default
        else:
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"detection.{section}.{key} must be an integer value") from exc
    else:
        raise ValueError(f"detection.{section}.{key} must be an integer value")

    if value < 0:
        raise ValueError(f"detection.{section}.{key} must be >= 0")

    return value


def _as_bool(
    *,
    section: str,
    key: str,
    default: bool,
) -> bool:
    block = _section(section)
    raw = block.get(key, default)

    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False

    raise ValueError(f"detection.{section}.{key} must be a boolean value")


def _as_int_tuple(
    *,
    section: str,
    key: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    block = _section(section)
    raw = block.get(key, list(default))

    if isinstance(raw, str):
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        try:
            out = tuple(int(t) for t in tokens)
        except ValueError as exc:
            raise ValueError(f"detection.{section}.{key} must be a list of integers") from exc
        return out if out else default

    if isinstance(raw, (list, tuple)):
        out: list[int] = []
        for x in raw:
            if isinstance(x, bool):
                raise ValueError(f"detection.{section}.{key} must be a list of integers")
            if isinstance(x, int):
                out.append(x)
                continue
            if isinstance(x, str):
                token = x.strip()
                if not token:
                    continue
                try:
                    out.append(int(token))
                except ValueError as exc:
                    raise ValueError(f"detection.{section}.{key} must be a list of integers") from exc
                continue
            raise ValueError(f"detection.{section}.{key} must be a list of integers")
        return tuple(out) if out else default

    raise ValueError(f"detection.{section}.{key} must be a list of integers")


@lru_cache(maxsize=1)
def lane_change_default_min_stable_before_frames() -> int:
    return _as_non_negative_int(
        section="lane_change",
        key="min_stable_before_frames",
        default=_DEFAULTS["lane_change"]["min_stable_before_frames"],
    )


@lru_cache(maxsize=1)
def lane_change_default_min_stable_after_frames() -> int:
    return _as_non_negative_int(
        section="lane_change",
        key="min_stable_after_frames",
        default=_DEFAULTS["lane_change"]["min_stable_after_frames"],
    )


@lru_cache(maxsize=1)
def lane_change_default_ignore_lane_ids() -> tuple[int, ...]:
    return _as_int_tuple(
        section="lane_change",
        key="ignore_lane_ids",
        default=tuple(_DEFAULTS["lane_change"]["ignore_lane_ids"]),
    )


@lru_cache(maxsize=1)
def cutin_default_search_window_frames() -> int:
    return _as_non_negative_int(
        section="cutin",
        key="search_window_frames",
        default=_DEFAULTS["cutin"]["search_window_frames"],
    )


@lru_cache(maxsize=1)
def cutin_default_start_offset_frames() -> int:
    return _as_non_negative_int(
        section="cutin",
        key="start_offset_frames",
        default=_DEFAULTS["cutin"]["start_offset_frames"],
    )


@lru_cache(maxsize=1)
def cutin_default_max_relation_delay_frames() -> int:
    return _as_non_negative_int(
        section="cutin",
        key="max_relation_delay_frames",
        default=_DEFAULTS["cutin"]["max_relation_delay_frames"],
    )


@lru_cache(maxsize=1)
def cutin_default_min_relation_frames() -> int:
    return _as_non_negative_int(
        section="cutin",
        key="min_relation_frames",
        default=_DEFAULTS["cutin"]["min_relation_frames"],
    )


@lru_cache(maxsize=1)
def cutin_default_require_new_follower() -> bool:
    return _as_bool(
        section="cutin",
        key="require_new_follower",
        default=bool(_DEFAULTS["cutin"]["require_new_follower"]),
    )


@lru_cache(maxsize=1)
def cutin_default_precheck_frames() -> int:
    return _as_non_negative_int(
        section="cutin",
        key="precheck_frames",
        default=_DEFAULTS["cutin"]["precheck_frames"],
    )


@lru_cache(maxsize=1)
def cutin_default_no_neighbor_ids() -> tuple[int, ...]:
    return _as_int_tuple(
        section="cutin",
        key="no_neighbor_ids",
        default=tuple(_DEFAULTS["cutin"]["no_neighbor_ids"]),
    )


@lru_cache(maxsize=1)
def cutin_default_require_lane_match() -> bool:
    return _as_bool(
        section="cutin",
        key="require_lane_match",
        default=bool(_DEFAULTS["cutin"]["require_lane_match"]),
    )


@lru_cache(maxsize=1)
def cutin_default_require_preceding_consistency() -> bool:
    return _as_bool(
        section="cutin",
        key="require_preceding_consistency",
        default=bool(_DEFAULTS["cutin"]["require_preceding_consistency"]),
    )
