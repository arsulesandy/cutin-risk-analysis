from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return Metrics(tp, fp, fn, tn, precision, recall, f1)


def best_thresholds(
        lat: np.ndarray,
        spd: np.ndarray,
        y: np.ndarray,
        lat_grid: np.ndarray,
        spd_grid: np.ndarray,
) -> tuple[float, float, Metrics]:
    best = None  # (f1, recall, precision, lat_thr, spd_thr, metrics)

    for lat_thr in lat_grid:
        lat_mask = lat >= lat_thr
        for spd_thr in spd_grid:
            pred = lat_mask & (spd >= spd_thr)
            m = compute_metrics(y, pred)

            key = (m.f1, m.recall, m.precision)
            if best is None or key > (best[0], best[1], best[2]):
                best = (m.f1, m.recall, m.precision, float(lat_thr), float(spd_thr), m)

    assert best is not None
    return best[3], best[4], best[5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11: Early warning rule search (grid).")
    parser.add_argument(
        "--merged-csv",
        type=str,
        default="outputs/reports/step9_batch/cutin_stage_features_merged.csv",
    )
    parser.add_argument("--thw-risk", type=float, default=0.7)
    parser.add_argument("--lat-min", type=float, default=0.6)
    parser.add_argument("--lat-max", type=float, default=1.6)
    parser.add_argument("--lat-step", type=float, default=0.05)
    parser.add_argument("--spd-min", type=float, default=0.0)
    parser.add_argument("--spd-max", type=float, default=10.0)
    parser.add_argument("--spd-step", type=float, default=0.25)
    parser.add_argument("--out-dir", type=str, default="outputs/reports/step11_warning")
    args = parser.parse_args()

    df = pd.read_csv(args.merged_csv)

    # Label = risky by your current thesis definition
    df["risk_thw"] = pd.to_numeric(df["execution_thw_min"], errors="coerce") < float(args.thw_risk)

    # Pre-cut-in (decision stage) signals
    df["decision_lat_v"] = pd.to_numeric(df["decision_cutter_lat_v_abs_max"], errors="coerce")
    df["decision_speed_delta"] = (
            pd.to_numeric(df["decision_cutter_speed_mean"], errors="coerce")
            - pd.to_numeric(df["decision_follower_speed_mean"], errors="coerce")
    )

    keep = df.dropna(subset=["recording_id", "risk_thw", "decision_lat_v", "decision_speed_delta"]).copy()

    if keep.empty:
        raise ValueError("No usable rows after dropping NaNs. Check column names in merged CSV.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_grid = np.arange(args.lat_min, args.lat_max + 1e-9, args.lat_step, dtype=float)
    spd_grid = np.arange(args.spd_min, args.spd_max + 1e-9, args.spd_step, dtype=float)

    # --- Global best thresholds (all data) ---
    lat = keep["decision_lat_v"].to_numpy(dtype=float)
    spd = keep["decision_speed_delta"].to_numpy(dtype=float)
    y = keep["risk_thw"].to_numpy(dtype=bool)

    best_lat, best_spd, best_m = best_thresholds(lat, spd, y, lat_grid, spd_grid)

    print("== Step 11: Early warning rule search ==")
    print(f"Label: execution_thw_min < {args.thw_risk:.2f}s")
    print("\nGlobal best rule:")
    print(f"  decision_lat_v >= {best_lat:.2f} AND decision_speed_delta >= {best_spd:.2f}")
    print(f"  precision={best_m.precision:.3f} recall={best_m.recall:.3f} f1={best_m.f1:.3f} "
          f"(tp={best_m.tp} fp={best_m.fp} fn={best_m.fn} tn={best_m.tn})")

    # --- Leave-one-recording-out evaluation ---
    records = sorted(keep["recording_id"].astype(str).unique().tolist())
    loo_rows = []

    for rid in records:
        train = keep[keep["recording_id"].astype(str) != rid]
        test = keep[keep["recording_id"].astype(str) == rid]

        lat_tr = train["decision_lat_v"].to_numpy(dtype=float)
        spd_tr = train["decision_speed_delta"].to_numpy(dtype=float)
        y_tr = train["risk_thw"].to_numpy(dtype=bool)

        lat_te = test["decision_lat_v"].to_numpy(dtype=float)
        spd_te = test["decision_speed_delta"].to_numpy(dtype=float)
        y_te = test["risk_thw"].to_numpy(dtype=bool)

        lat_thr, spd_thr, _ = best_thresholds(lat_tr, spd_tr, y_tr, lat_grid, spd_grid)
        pred_te = (lat_te >= lat_thr) & (spd_te >= spd_thr)
        m_te = compute_metrics(y_te, pred_te)

        loo_rows.append(
            {
                "heldout_recording": rid,
                "lat_thr": lat_thr,
                "spd_thr": spd_thr,
                "precision": m_te.precision,
                "recall": m_te.recall,
                "f1": m_te.f1,
                "tp": m_te.tp,
                "fp": m_te.fp,
                "fn": m_te.fn,
                "tn": m_te.tn,
                "n_test": int(len(y_te)),
            }
        )

    loo = pd.DataFrame(loo_rows)
    out_csv = out_dir / "leave_one_recording_out.csv"
    loo.to_csv(out_csv, index=False)

    print("\nLeave-one-recording-out results:")
    print(loo.to_string(index=False))

    print("\nMacro averages:")
    print(
        {
            "precision": float(loo["precision"].mean()),
            "recall": float(loo["recall"].mean()),
            "f1": float(loo["f1"].mean()),
        }
    )
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
