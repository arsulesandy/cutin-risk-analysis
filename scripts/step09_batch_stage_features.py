"""Step 09: batch runner that executes Step 08 across many recordings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path

import pandas as pd
from cutin_risk.io.progress import iter_with_progress
from cutin_risk.io.step_reports import mirror_file_to_step, step_reports_dir, write_step_markdown
from cutin_risk.paths import dataset_root_path, output_path, project_root
from cutin_risk.thesis_config import thesis_str


def parse_recordings_arg(value: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return [f"{i:02d}" for i in range(1, 61)]
    parts = [p.strip() for p in value.split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            start = int(a)
            end = int(b)
            out.extend([f"{i:02d}" for i in range(start, end + 1)])
        else:
            out.append(f"{int(p):02d}" if p.isdigit() else p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 9: Run stage_features across many recordings.")
    parser.add_argument(
        "--recordings",
        type=str,
        default=thesis_str("step09.recordings", "1-10"),
        help="Examples: '01', '01,02,03', '1-10', or 'all'",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(dataset_root_path()),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(output_path("reports/step9_batch")),
    )
    parser.add_argument(
        "--make-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also generate plots per recording (slower, more files).",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = parse_recordings_arg(args.recordings)
    step8_script = project_root() / "scripts" / "step08_stage_features.py"

    # We call your existing working Step 8 script, because it is already validated.
    # It will write one CSV per recording in outputs/reports/step8_recording_XX/.
    # Then we merge them into one batch file here.
    produced_csvs: list[Path] = []

    for _, _, rid in iter_with_progress(
        recordings,
        label="Step 09 recordings",
        item_name="recording",
    ):
        cmd = [
            sys.executable,
            str(step8_script),
            "--dataset-root",
            args.dataset_root,
            "--recording-id",
            rid,
        ]
        # Always forward the plot toggle so Step 08 does not silently fall back
        # to its own config default.
        cmd.append("--make-plot" if args.make_plot else "--no-make-plot")

        print(f"\n=== Running stage_features for recording {rid} ===")
        ret = subprocess.run(cmd, text=True)
        if ret.returncode != 0:
            print(f"[WARN] stage_features failed for recording {rid}, continuing…")
            continue

        # Prefer deterministic Step 08 filenames, fallback to older timestamped files if present.
        candidate_dirs = [
            step_reports_dir(8, subdir=f"recording_{rid}", create=False),
            output_path(f"reports/step8_recording_{rid}"),
        ]
        chosen: Path | None = None
        for d in candidate_dirs:
            deterministic = d / "cutin_stage_features.csv"
            if deterministic.exists():
                chosen = deterministic
                break
        if chosen is None:
            csvs: list[Path] = []
            for d in candidate_dirs:
                csvs.extend(list(d.glob("cutin_stage_features_*.csv")))
            csvs = sorted(csvs, key=lambda p: p.stat().st_mtime)
            if csvs:
                chosen = csvs[-1]
        if chosen is None:
            print(f"[WARN] No CSV produced for recording {rid} (unexpected).")
            continue

        produced_csvs.append(chosen)
        print(f"Collected: {chosen}")

    if not produced_csvs:
        print("\nNo outputs collected. Nothing to merge.")
        return

    print("\n=== Merging outputs ===")
    all_frames = []
    for p in produced_csvs:
        df = pd.read_csv(p)
        df["recording_id"] = df["recording_id"].astype(str)
        all_frames.append(df)

    merged = pd.concat(all_frames, ignore_index=True)
    out_csv = out_dir / "cutin_stage_features_merged.csv"
    merged.to_csv(out_csv, index=False)

    # Small per-recording summary table
    summary = (
        merged.groupby("recording_id")
        .agg(
            cutins=("cutter_id", "count"),
            ttc_finite=("ttc_min_total",
                        lambda s: pd.to_numeric(s, errors="coerce").replace([float("inf")], pd.NA).notna().sum()),
            exec_thw_median=("execution_thw_min", "median"),
        )
        .reset_index()
        .sort_values("recording_id")
    )
    out_summary = out_dir / "recording_summary.csv"
    summary.to_csv(out_summary, index=False)

    canonical_csv = mirror_file_to_step(out_csv, 9)
    canonical_summary = mirror_file_to_step(out_summary, 9)
    details_md = write_step_markdown(
        9,
        "stage_batch_details.md",
        [
            "# Step 09 Batch Stage Features Report",
            "",
            f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Recordings requested: `{args.recordings}`",
            f"- Recordings processed: `{len(produced_csvs)}`",
            f"- Total merged events: `{len(merged)}`",
            f"- Merged CSV: `{canonical_csv}`",
            f"- Recording summary CSV: `{canonical_summary}`",
        ],
    )

    print(f"\nSaved merged event table: {out_csv}")
    print(f"Saved per-recording summary: {out_summary}")
    print(f"Total events merged: {len(merged)}")
    print(f"Saved summary markdown: {details_md}")


if __name__ == "__main__":
    main()
