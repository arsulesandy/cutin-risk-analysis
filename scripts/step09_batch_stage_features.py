from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

import subprocess
import sys


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
        default="01",
        help="Examples: '01', '01,02,03', '1-10', or 'all'",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/raw/highD-dataset-v1.0/data",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/reports/step9_batch",
    )
    parser.add_argument(
        "--make-plot",
        action="store_true",
        help="Also generate plots per recording (slower, more files).",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = parse_recordings_arg(args.recordings)

    # We call your existing working Step 8 script, because it is already validated.
    # It will write one CSV per recording in outputs/reports/step8_recording_XX/.
    # Then we merge them into one batch file here.
    produced_csvs: list[Path] = []

    for rid in recordings:
        cmd = [
            sys.executable,
            "scripts/step08_stage_features.py",
            "--dataset-root",
            args.dataset_root,
            "--recording-id",
            rid,
        ]
        if args.make_plot:
            cmd.append("--make-plot")

        print(f"\n=== Running stage_features for recording {rid} ===")
        ret = subprocess.run(cmd, text=True)
        if ret.returncode != 0:
            print(f"[WARN] stage_features failed for recording {rid}, continuing…")
            continue

        # Find the newest CSV that stage_features produced for this recording
        step8_dir = Path("outputs") / "reports" / f"step8_recording_{rid}"
        csvs = sorted(step8_dir.glob("cutin_stage_features_*.csv"))
        if not csvs:
            print(f"[WARN] No CSV produced for recording {rid} (unexpected).")
            continue

        produced_csvs.append(csvs[-1])
        print(f"Collected: {csvs[-1]}")

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
            ttc_finite=("ttc_min_total", lambda s: pd.to_numeric(s, errors="coerce").replace([float("inf")], pd.NA).notna().sum()),
            exec_thw_median=("execution_thw_min", "median"),
        )
        .reset_index()
        .sort_values("recording_id")
    )
    out_summary = out_dir / "recording_summary.csv"
    summary.to_csv(out_summary, index=False)

    print(f"\nSaved merged event table: {out_csv}")
    print(f"Saved per-recording summary: {out_summary}")
    print(f"Total events merged: {len(merged)}")


if __name__ == "__main__":
    main()
