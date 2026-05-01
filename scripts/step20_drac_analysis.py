#!/usr/bin/env python3
"""
Step 20 — DRAC (Deceleration Rate to Avoid Crash) Analysis
===========================================================

Computes DRAC for each cut-in event using existing stage features.
DRAC = (v_follower - v_leader)^2 / (2 * DHW)

This extends the indicator layer beyond DHW/THW/TTC to include
a deceleration-based measure that captures braking demand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs" / "reports"
STAGE_FEATURES = OUTPUTS / "Step 09" / "cutin_stage_features_merged.csv"
OUT_DIR = OUTPUTS / "Step 20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_drac_from_stage_features(sf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute execution-stage DRAC estimate from stage feature summaries.

    Since we have stage-level aggregates (not per-frame data), we compute
    an approximate DRAC using:
      - execution_dhw_min as the critical gap
      - execution_follower_speed_mean and execution_cutter_speed_mean

    For a more precise DRAC, per-frame computation would be needed.
    Here we compute two variants:
      1) DRAC from mean speeds and min DHW (worst-case estimate)
      2) DRAC flag: whether the implied deceleration exceeds literature thresholds
    """
    sf = sf.copy()

    # Closing speed estimate from stage means
    closing_speed = sf["execution_follower_speed_mean"] - sf["execution_cutter_speed_mean"]

    # DRAC = closing_speed^2 / (2 * DHW_min) — only when closing
    dhw = sf["execution_dhw_min"].clip(lower=0.01)
    drac = np.where(
        closing_speed > 1e-6,
        (closing_speed ** 2) / (2.0 * dhw),
        np.inf
    )

    sf["execution_drac"] = drac
    sf["execution_drac_finite"] = np.isfinite(drac)

    # Literature thresholds
    # 3.35 m/s² — Archer (2005), "hard braking"
    # 2.0 m/s² — moderate braking
    sf["drac_exceeds_3.35"] = np.isfinite(drac) & (drac > 3.35)
    sf["drac_exceeds_2.0"] = np.isfinite(drac) & (drac > 2.0)

    return sf


def main():
    print("=" * 70)
    print("Step 20: DRAC Analysis")
    print("=" * 70)

    sf = pd.read_csv(STAGE_FEATURES)
    print(f"Loaded {len(sf)} events")

    # Add risk label for comparison
    sf["risk_thw"] = (sf["execution_thw_min"] < 0.7).astype(int)

    # Compute DRAC
    sf = compute_drac_from_stage_features(sf)

    # --- DRAC Summary ---
    finite_drac = sf[sf["execution_drac_finite"]]
    print(f"\nDRAC finite (closing) events: {len(finite_drac)} / {len(sf)} "
          f"({len(finite_drac)/len(sf):.1%})")

    if len(finite_drac) > 0:
        print(f"\nExecution-stage DRAC distribution (finite only, n={len(finite_drac)}):")
        stats = finite_drac["execution_drac"].describe(
            percentiles=[0.25, 0.5, 0.75, 0.90, 0.95]
        )
        print(stats.to_string())

    # --- DRAC vs THW Risk ---
    print("\n--- DRAC vs THW Risk Cross-tabulation ---")
    for thr_name, thr in [("3.35 m/s² (hard braking)", 3.35), ("2.0 m/s² (moderate)", 2.0)]:
        drac_flag = sf["execution_drac_finite"] & (sf["execution_drac"] > thr)
        ct = pd.crosstab(
            sf["risk_thw"].map({0: "THW-safe", 1: "THW-risky"}),
            drac_flag.map({True: f"DRAC>{thr}", False: f"DRAC<={thr}"}),
            margins=True,
        )
        print(f"\nThreshold: {thr_name}")
        print(ct.to_string())

    # --- Overlap analysis ---
    print("\n--- Indicator Overlap ---")
    thw_risky = sf["risk_thw"] == 1
    ttc_finite = sf["execution_ttc_min"].notna() & np.isfinite(sf["execution_ttc_min"])
    ttc_low = ttc_finite & (sf["execution_ttc_min"] < 5.0)
    drac_high = sf["execution_drac_finite"] & (sf["execution_drac"] > 3.35)

    print(f"THW < 0.7s:     {thw_risky.sum():>5} ({thw_risky.mean():.1%})")
    print(f"TTC < 5.0s:     {ttc_low.sum():>5} ({ttc_low.mean():.1%})")
    print(f"DRAC > 3.35:    {drac_high.sum():>5} ({drac_high.mean():.1%})")
    print(f"THW & DRAC:     {(thw_risky & drac_high).sum():>5}")
    print(f"THW & TTC:      {(thw_risky & ttc_low).sum():>5}")
    print(f"DRAC & TTC:     {(drac_high & ttc_low).sum():>5}")
    print(f"All three:      {(thw_risky & drac_high & ttc_low).sum():>5}")

    # --- Risk group comparison ---
    print("\n--- DRAC by THW Risk Group ---")
    for group_name, mask in [("THW-safe", ~thw_risky), ("THW-risky", thw_risky)]:
        subset = sf[mask & sf["execution_drac_finite"]]
        if len(subset) > 0:
            print(f"\n  {group_name} (n={len(subset)} with finite DRAC):")
            print(f"    DRAC median: {subset['execution_drac'].median():.4f} m/s²")
            print(f"    DRAC mean:   {subset['execution_drac'].mean():.4f} m/s²")
            print(f"    DRAC > 3.35: {(subset['execution_drac'] > 3.35).sum()} "
                  f"({(subset['execution_drac'] > 3.35).mean():.1%})")

    # --- DRAC threshold sensitivity ---
    print("\n--- DRAC Threshold Sensitivity ---")
    for thr in [1.0, 2.0, 3.0, 3.35, 4.0, 5.0]:
        flagged = sf["execution_drac_finite"] & (sf["execution_drac"] > thr)
        print(f"  DRAC > {thr:.2f} m/s²: {flagged.sum():>5} events ({flagged.mean():.2%})")

    # --- Save ---
    sf.to_csv(OUT_DIR / "stage_features_with_drac.csv", index=False)

    # Summary table for thesis
    summary_rows = []
    for thr in [1.0, 2.0, 3.0, 3.35, 5.0]:
        flagged = sf["execution_drac_finite"] & (sf["execution_drac"] > thr)
        summary_rows.append({
            "drac_threshold_ms2": thr,
            "flagged_events": int(flagged.sum()),
            "flagged_ratio": round(flagged.mean(), 4),
            "overlap_thw_risky": int((flagged & thw_risky).sum()),
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "drac_threshold_sensitivity.csv", index=False)

    print(f"\nOutputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
