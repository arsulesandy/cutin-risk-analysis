"""Helpers for canonical Step NN report/figure artifact layout."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable

from cutin_risk.paths import step_output_dir


def step_reports_dir(step: int | str, *, subdir: str | None = None, create: bool = True) -> Path:
    """Return canonical report directory for a step, optionally under a subdirectory."""
    base = step_output_dir(step, kind="reports", create=create)
    if not subdir:
        return base
    target = base / subdir
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def step_figures_dir(step: int | str, *, subdir: str | None = None, create: bool = True) -> Path:
    """Return canonical figure directory for a step, optionally under a subdirectory."""
    base = step_output_dir(step, kind="figures", create=create)
    if not subdir:
        return base
    target = base / subdir
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def mirror_file_to_step(
    src: str | Path,
    step: int | str,
    *,
    kind: str = "reports",
    subdir: str | None = None,
    dst_name: str | None = None,
) -> Path:
    """
    Copy an existing file into canonical Step NN outputs and return destination path.

    Existing destination files are replaced.
    """
    source = Path(src)
    if not source.exists():
        raise FileNotFoundError(f"Cannot mirror missing file: {source}")

    if kind == "reports":
        dst_dir = step_reports_dir(step, subdir=subdir, create=True)
    elif kind == "figures":
        dst_dir = step_figures_dir(step, subdir=subdir, create=True)
    else:
        raise ValueError(f"Unsupported artifact kind: {kind!r}")

    target = dst_dir / (dst_name or source.name)
    if source.resolve() == target.resolve():
        return target
    shutil.copy2(source, target)
    return target


def write_step_markdown(
    step: int | str,
    filename: str,
    lines: Iterable[str],
    *,
    subdir: str | None = None,
) -> Path:
    """Write markdown lines into canonical Step NN report directory."""
    if not filename.lower().endswith(".md"):
        raise ValueError("filename must end with .md")
    dst = step_reports_dir(step, subdir=subdir, create=True) / filename
    text = "\n".join(lines).rstrip() + "\n"
    dst.write_text(text, encoding="utf-8")
    return dst

