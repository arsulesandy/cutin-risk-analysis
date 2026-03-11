"""Copy thesis figures referenced by latex/main.tex from outputs/ into latex/."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


INCLUDE_RE = re.compile(r"\\(?:safe)?includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
SUPPORTED_SUFFIXES = {".png", ".pdf", ".jpg", ".jpeg"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_referenced_assets(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    ordered: list[str] = []
    seen: set[str] = set()
    for match in INCLUDE_RE.finditer(text):
        raw_name = match.group(1).strip()
        name = Path(raw_name).name
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def resolve_source(outputs_dir: Path, filename: str) -> Path | None:
    matches = sorted(path for path in outputs_dir.rglob(filename) if path.is_file())
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    report_matches = [path for path in matches if "outputs/reports/" in path.as_posix()]
    if len(report_matches) == 1:
        return report_matches[0]

    raise ValueError(
        f"Ambiguous asset '{filename}'. Multiple candidates found:\n"
        + "\n".join(f"- {path}" for path in matches)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy thesis figure assets from outputs/ into latex/.")
    parser.add_argument("--latex-dir", type=Path, default=project_root() / "latex")
    parser.add_argument("--outputs-dir", type=Path, default=project_root() / "outputs")
    parser.add_argument("--tex-file", type=Path, default=None, help="Defaults to <latex-dir>/main.tex")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    latex_dir = args.latex_dir.resolve()
    outputs_dir = args.outputs_dir.resolve()
    tex_file = args.tex_file.resolve() if args.tex_file else latex_dir / "main.tex"

    if not tex_file.exists():
        raise FileNotFoundError(f"Missing LaTeX source: {tex_file}")
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Missing outputs directory: {outputs_dir}")

    copied: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    for filename in parse_referenced_assets(tex_file):
        source = resolve_source(outputs_dir, filename)
        if source is None:
            skipped.append(filename)
            continue
        target = latex_dir / filename
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied.append((source, target))

    mode = "Would copy" if args.dry_run else "Copied"
    print(f"{mode} {len(copied)} asset(s) into {latex_dir}")
    for source, target in copied:
        print(f"- {source} -> {target}")

    if skipped:
        print(f"Skipped {len(skipped)} referenced asset(s) with no source under outputs/:")
        for filename in skipped:
            print(f"- {filename}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
