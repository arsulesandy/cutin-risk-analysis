"""Run the retained thesis workflow using thesis-level macro-phases."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cutin_risk.paths import output_path, step14_codes_csv_path
from cutin_risk.thesis_config import thesis_float, thesis_int, thesis_str


@dataclass(frozen=True)
class CommandSpec:
    label: str
    argv: list[str]


@dataclass(frozen=True)
class PhaseSpec:
    title: str
    description: str
    commands: tuple[CommandSpec, ...]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_phase_specs(python_bin: str) -> OrderedDict[str, PhaseSpec]:
    recordings_batch = thesis_str("pipeline.recordings_batch", "all")
    thw_risk = thesis_float("pipeline.thw_risk", 0.7, min_value=0.0)
    ci_bootstrap = thesis_int("pipeline.ci_bootstrap", 3000, min_value=1)
    ci_seed = thesis_int("pipeline.ci_seed", 7)
    step14_codes_csv = str(step14_codes_csv_path())

    return OrderedDict(
        [
            (
                "preliminary-processing",
                PhaseSpec(
                    title="Preliminary processing",
                    description="Basic dataset harmonisation, recording checks, and lane-change summary.",
                    commands=(
                        CommandSpec("Recording report", [python_bin, "scripts/step01_recording_report.py"]),
                        CommandSpec("Lane-change report", [python_bin, "scripts/step02_lane_change_report.py"]),
                    ),
                ),
            ),
            (
                "interaction-mining",
                PhaseSpec(
                    title="Interaction mining",
                    description="Cut-in extraction and dataset-wide event statistics.",
                    commands=(
                        CommandSpec("Cut-in report", [python_bin, "scripts/step03_cutin_report.py"]),
                    ),
                ),
            ),
            (
                "core-reconstruction",
                PhaseSpec(
                    title="Core reconstruction",
                    description="Neighbour reconstruction and XY-lane reconstruction with agreement checks.",
                    commands=(
                        CommandSpec("Neighbour reconstruction", [python_bin, "scripts/step06_neighbor_reconstruction_report.py"]),
                        CommandSpec("XY-lane pipeline", [python_bin, "scripts/step07_xy_lane_pipeline_report.py"]),
                    ),
                ),
            ),
            (
                "safety-quantification",
                PhaseSpec(
                    title="Safety quantification",
                    description="Stage feature extraction, risk summaries, and pair-consistent indicators.",
                    commands=(
                        CommandSpec(
                            "Batch stage features",
                            [python_bin, "scripts/step09_batch_stage_features.py", "--recordings", recordings_batch],
                        ),
                        CommandSpec(
                            "Risk report",
                            [python_bin, "scripts/step10_risk_report.py", "--thw-risk", str(thw_risk)],
                        ),
                    ),
                ),
            ),
            (
                "context-signatures",
                PhaseSpec(
                    title="Context signatures",
                    description="Binary SFC encoding, canonical mirroring, and archetype analysis.",
                    commands=(
                        CommandSpec("SFC binary encode", [python_bin, "scripts/step14_sfc_binary_encode.py"]),
                        CommandSpec("SFC binary report", [python_bin, "scripts/step14_sfc_binary_report.py"]),
                        CommandSpec(
                            "SFC mirror normalize",
                            [python_bin, "scripts/step15a_sfc_mirror_normalize.py", "--codes-csv", step14_codes_csv],
                        ),
                        CommandSpec("SFC archetypes", [python_bin, "scripts/step15d_sfc_archetypes.py"]),
                    ),
                ),
            ),
            (
                "reference-label-evaluation",
                PhaseSpec(
                    title="Reference-label evaluation",
                    description="Bootstrap confidence intervals for the reconstruction agreement metrics.",
                    commands=(
                        CommandSpec(
                            "Metrics CI",
                            [
                                python_bin,
                                "scripts/step18_metrics_ci.py",
                                "--n-bootstrap",
                                str(ci_bootstrap),
                                "--seed",
                                str(ci_seed),
                            ],
                        ),
                    ),
                ),
            ),
        ]
    )


def append_command_log(path: Path | None, argv: list[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(" ".join(argv) + "\n")


def run_quiet(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def write_manifest(final_dir: Path, phases: list[str], commands: list[list[str]]) -> Path:
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "root_dir": str(project_root()),
        "python_version": platform.python_version(),
        "git_commit": run_quiet(["git", "rev-parse", "HEAD"]),
        "git_branch": run_quiet(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "phases": phases,
        "commands": commands,
    }
    out_path = final_dir / "phase_run_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the thesis workflow using thesis-level macro-phases.")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["all"],
        help="Phase keys to run, or 'all'. Use --list-phases to inspect available keys.",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--command-log", type=Path, default=None)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--manifest-dir", type=Path, default=output_path("reports/final"))
    parser.add_argument("--list-phases", action="store_true")
    args = parser.parse_args()

    phase_specs = build_phase_specs(args.python_bin)
    available = list(phase_specs.keys())

    if args.list_phases:
        print("Available thesis phases:")
        for key, spec in phase_specs.items():
            print(f"- {key}: {spec.title}")
            print(f"  {spec.description}")
        return 0

    if args.phases == ["all"]:
        selected = available
    else:
        invalid = [phase for phase in args.phases if phase not in phase_specs]
        if invalid:
            raise ValueError(f"Unknown phase(s): {', '.join(invalid)}")
        selected = args.phases

    executed_commands: list[list[str]] = []
    print("== Thesis Phase Runner ==")
    print(f"Root: {project_root()}")
    print(f"Python: {args.python_bin}")
    print(f"Phases: {', '.join(selected)}")

    for phase_key in selected:
        spec = phase_specs[phase_key]
        print()
        print(f"== Phase: {spec.title} ({phase_key}) ==")
        print(spec.description)
        for command in spec.commands:
            argv = command.argv
            append_command_log(args.command_log, argv)
            executed_commands.append(argv)
            print(f"[RUN] {command.label}: {' '.join(argv)}")
            subprocess.run(argv, check=True, cwd=project_root(), env=os.environ.copy())

    if args.write_manifest:
        manifest_path = write_manifest(args.manifest_dir.resolve(), selected, executed_commands)
        print()
        print(f"Wrote manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
