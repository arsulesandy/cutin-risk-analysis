"""Step 10: summarize and sanity-check risk labels from stage feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from cutin_risk.paths import output_path


def finite_count(s: pd.Series) -> int:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return int(np.isfinite(x).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10: risk labeling + summary from merged stage features.")
    parser.add_argument(
        "--merged-csv",
        type=str,
        default=str(output_path("reports/step9_batch/cutin_stage_features_merged.csv")),
    )
    parser.add_argument("--thw-risk", type=float, default=0.7, help="Risk threshold for execution_thw_min.")
    parser.add_argument("--thw-very-risk", type=float, default=0.5, help="Very-risk threshold for execution_thw_min.")
    parser.add_argument("--out-dir", type=str, default=str(output_path("reports/step10_risk")))

    args = parser.parse_args()

    merged_csv = Path(args.merged_csv)
    if not merged_csv.exists():
        raise FileNotFoundError(f"Missing input: {merged_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_csv)

    # Ensure numeric where needed
    for c in [
        "execution_thw_min",
        "execution_dhw_min",
        "execution_ttc_min",
        "decision_cutter_lat_v_abs_max",
        "execution_cutter_lat_v_abs_max",
        "execution_cutter_speed_mean",
        "execution_follower_speed_mean",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["risk_thw"] = df["execution_thw_min"] < float(args.thw_risk)
    df["very_risk_thw"] = df["execution_thw_min"] < float(args.thw_very_risk)

    df["exec_speed_delta"] = df["execution_cutter_speed_mean"] - df["execution_follower_speed_mean"]

    # Per-recording summary
    summary = (
        df.groupby("recording_id")
        .agg(
            cutins=("cutter_id", "count"),
            exec_thw_median=("execution_thw_min", "median"),
            exec_dhw_median=("execution_dhw_min", "median"),
            exec_ttc_finite=("execution_ttc_min", finite_count),
            risky=("risk_thw", "sum"),
            very_risky=("very_risk_thw", "sum"),
            decision_lat_v_median=("decision_cutter_lat_v_abs_max", "median"),
            exec_lat_v_median=("execution_cutter_lat_v_abs_max", "median"),
        )
        .reset_index()
        .sort_values("recording_id")
    )
    summary["risky_pct"] = (summary["risky"] / summary["cutins"] * 100).round(1)
    summary["very_risky_pct"] = (summary["very_risky"] / summary["cutins"] * 100).round(1)

    out_summary = out_dir / "risk_summary_by_recording.csv"
    summary.to_csv(out_summary, index=False)

    # Risk vs non-risk comparison (pooled)
    pooled = (
        df.groupby("risk_thw")
        .agg(
            events=("cutter_id", "count"),
            exec_thw_median=("execution_thw_min", "median"),
            exec_dhw_median=("execution_dhw_min", "median"),
            decision_lat_v_median=("decision_cutter_lat_v_abs_max", "median"),
            exec_lat_v_median=("execution_cutter_lat_v_abs_max", "median"),
            exec_speed_delta_median=("exec_speed_delta", "median"),
        )
        .reset_index()
    )
    out_pooled = out_dir / "risk_vs_nonrisk_pooled.csv"
    pooled.to_csv(out_pooled, index=False)

    print("== Step 10: Risk report ==")
    print("Input:", merged_csv)
    print("Saved:", out_summary)
    print("Saved:", out_pooled)
    print("\nPer-recording summary:")
    print(summary.to_string(index=False))
    print("\nRisk vs non-risk (pooled):")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
