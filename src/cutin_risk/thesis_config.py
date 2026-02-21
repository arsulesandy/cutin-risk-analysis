"""Central configuration loader for thesis-impacting defaults.

Precedence:
1. Environment variable `CUTIN_THESIS_CONFIG_FILE`
2. `configs/thesis.local.json`
3. `configs/thesis.json`
4. Call-site default value
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

from cutin_risk.paths import project_root

_MISSING = object()


def _config_file_path() -> Path:
    env_path = os.environ.get("CUTIN_THESIS_CONFIG_FILE")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.is_absolute() else project_root() / p

    local_cfg = project_root() / "configs" / "thesis.local.json"
    if local_cfg.exists():
        return local_cfg

    return project_root() / "configs" / "thesis.json"


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
        raise ValueError(f"Invalid JSON in thesis config file {cfg_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Thesis config file {cfg_path} must contain a JSON object.")
    return data


def _raw(path: str) -> Any:
    if not path.strip():
        raise ValueError("Config path must not be empty.")

    cur: Any = _load_config()
    for part in path.split("."):
        key = part.strip()
        if not key:
            raise ValueError(f"Invalid config path segment in {path!r}")
        if not isinstance(cur, dict) or key not in cur:
            return _MISSING
        cur = cur[key]
    return cur


def thesis_value(path: str, default: Any) -> Any:
    value = _raw(path)
    return default if value is _MISSING else value


def thesis_str(path: str, default: str, *, allowed: set[str] | None = None) -> str:
    raw = _raw(path)
    if raw is _MISSING:
        value = default
    elif isinstance(raw, str):
        value = raw.strip()
        if not value:
            value = default
    else:
        value = str(raw).strip()
        if not value:
            value = default

    if allowed is not None and value not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}, got {value!r}")
    return value


def thesis_bool(path: str, default: bool) -> bool:
    raw = _raw(path)
    if raw is _MISSING:
        return bool(default)

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

    raise ValueError(f"{path} must be a boolean value.")


def thesis_int(path: str, default: int, *, min_value: int | None = None) -> int:
    raw = _raw(path)
    if raw is _MISSING:
        value = int(default)
    elif isinstance(raw, bool):
        raise ValueError(f"{path} must be an integer, not boolean")
    elif isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError(f"{path} must be an integer value")
        value = int(raw)
    elif isinstance(raw, str):
        token = raw.strip()
        if not token:
            value = int(default)
        else:
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"{path} must be an integer value") from exc
    else:
        raise ValueError(f"{path} must be an integer value")

    if min_value is not None and value < min_value:
        raise ValueError(f"{path} must be >= {min_value}")
    return value


def thesis_optional_int(path: str, default: int | None = None, *, min_value: int | None = None) -> int | None:
    raw = _raw(path)
    if raw is _MISSING:
        value = default
    elif raw is None:
        value = None
    else:
        value = thesis_int(path, 0, min_value=min_value)
    return value


def thesis_float(path: str, default: float, *, min_value: float | None = None) -> float:
    raw = _raw(path)
    if raw is _MISSING:
        value = float(default)
    elif isinstance(raw, bool):
        raise ValueError(f"{path} must be a float, not boolean")
    elif isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        token = raw.strip()
        if not token:
            value = float(default)
        else:
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(f"{path} must be a float value") from exc
    else:
        raise ValueError(f"{path} must be a float value")

    if min_value is not None and value < min_value:
        raise ValueError(f"{path} must be >= {min_value}")
    return value


def thesis_optional_float(
    path: str,
    default: float | None = None,
    *,
    min_value: float | None = None,
) -> float | None:
    raw = _raw(path)
    if raw is _MISSING:
        value = default
    elif raw is None:
        value = None
    else:
        value = thesis_float(path, 0.0, min_value=min_value)
    return value
