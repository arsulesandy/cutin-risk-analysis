"""Step 18: bootstrap confidence intervals for per-fold and micro metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.paths import output_path


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom else 0.0


def _micro_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _bootstrap_macro(
    df: pd.DataFrame,
    metrics: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, tuple[float, float]]:
    n = len(df)
    out: dict[str, tuple[float, float]] = {}
    idx = np.arange(n)
    for metric in metrics:
        vals = np.empty(n_boot, dtype=float)
        arr = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        for i in range(n_boot):
            s = rng.choice(idx, size=n, replace=True)
            vals[i] = float(np.nanmean(arr[s]))
        out[metric] = (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
    return out


def _bootstrap_micro(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, tuple[float, float]]:
    n = len(df)
    idx = np.arange(n)
    p_vals = np.empty(n_boot, dtype=float)
    r_vals = np.empty(n_boot, dtype=float)
    f_vals = np.empty(n_boot, dtype=float)

    tp_arr = pd.to_numeric(df["tp"], errors="coerce").to_numpy(dtype=float)
    fp_arr = pd.to_numeric(df["fp"], errors="coerce").to_numpy(dtype=float)
    fn_arr = pd.to_numeric(df["fn"], errors="coerce").to_numpy(dtype=float)
    tn_arr = pd.to_numeric(df["tn"], errors="coerce").to_numpy(dtype=float)

    for i in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        m = _micro_from_counts(
            tp=float(np.nansum(tp_arr[s])),
            fp=float(np.nansum(fp_arr[s])),
            fn=float(np.nansum(fn_arr[s])),
            tn=float(np.nansum(tn_arr[s])),
        )
        p_vals[i] = m["precision"]
        r_vals[i] = m["recall"]
        f_vals[i] = m["f1"]

    return {
        "precision": (float(np.nanpercentile(p_vals, 2.5)), float(np.nanpercentile(p_vals, 97.5))),
        "recall": (float(np.nanpercentile(r_vals, 2.5)), float(np.nanpercentile(r_vals, 97.5))),
        "f1": (float(np.nanpercentile(f_vals, 2.5)), float(np.nanpercentile(f_vals, 97.5))),
    }


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = df.columns.tolist()
    rows = df.astype(str).values.tolist()
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


@dataclass(frozen=True)
class ExperimentTarget:
    experiment: str
    csv_path: Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 18: Bootstrap confidence intervals for LOO metrics.")
    ap.add_argument("--n-bootstrap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=str, default=str(output_path("reports/final")))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ExperimentTarget("step11_rule_search", output_path("reports/step11_warning/leave_one_recording_out.csv")),
        ExperimentTarget("step12_logreg", output_path("reports/step12_warning_logreg/leave_one_recording_out_logreg.csv")),
        ExperimentTarget("step13a_tuned_logreg", output_path("reports/step13a_warning_logreg/leave_one_recording_out_step13a.csv")),
        ExperimentTarget("step15c_binary", output_path("reports/step15c_pred_binary/leave_one_recording_out.csv")),
        ExperimentTarget(
            "step15c_weighted_distance",
            output_path("reports/step15c_pred_weighted_distance/leave_one_recording_out.csv"),
        ),
        ExperimentTarget("step15c_weighted_ttc", output_path("reports/step15c_pred_weighted_ttc/leave_one_recording_out.csv")),
        ExperimentTarget("step16_lanechange", output_path("reports/step16_sfc_predict/lanechange_loocv.csv")),
        ExperimentTarget("step16_cutin", output_path("reports/step16_sfc_predict/cutin_loocv.csv")),
    ]

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(args.seed))
    metrics = ["precision", "recall", "f1"]

    for t in targets:
        if not t.csv_path.exists():
            continue
        df = pd.read_csv(t.csv_path)
        need = {"precision", "recall", "tp", "fp", "fn", "tn"}
        if not (need <= set(df.columns)):
            continue

        # clean
        d = df.copy()
        for c in ["precision", "recall", "tp", "fp", "fn", "tn"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        if "f1" in d.columns:
            d["f1"] = pd.to_numeric(d["f1"], errors="coerce")
        else:
            # step13a stores fbeta only; derive f1 directly from confusion counts for consistency.
            d["f1"] = d.apply(
                lambda r: _f1(_safe_div(r["tp"], r["tp"] + r["fp"]), _safe_div(r["tp"], r["tp"] + r["fn"])),
                axis=1,
            )
        d = d.dropna(subset=["precision", "recall", "f1", "tp", "fp", "fn", "tn"]).copy()
        if d.empty:
            continue

        macro_point = {m: float(d[m].mean()) for m in metrics}
        macro_ci = _bootstrap_macro(d, metrics, rng, int(args.n_bootstrap))

        micro_point = _micro_from_counts(
            tp=float(d["tp"].sum()),
            fp=float(d["fp"].sum()),
            fn=float(d["fn"].sum()),
            tn=float(d["tn"].sum()),
        )
        micro_ci = _bootstrap_micro(d, rng, int(args.n_bootstrap))

        for m in metrics:
            rows.append(
                {
                    "experiment": t.experiment,
                    "source_csv": str(t.csv_path),
                    "n_folds": int(len(d)),
                    "estimate_type": "macro",
                    "metric": m,
                    "point": macro_point[m],
                    "ci_low": macro_ci[m][0],
                    "ci_high": macro_ci[m][1],
                    "n_bootstrap": int(args.n_bootstrap),
                }
            )
            rows.append(
                {
                    "experiment": t.experiment,
                    "source_csv": str(t.csv_path),
                    "n_folds": int(len(d)),
                    "estimate_type": "micro",
                    "metric": m,
                    "point": micro_point[m],
                    "ci_low": micro_ci[m][0],
                    "ci_high": micro_ci[m][1],
                    "n_bootstrap": int(args.n_bootstrap),
                }
            )

    if not rows:
        raise ValueError("No usable experiment CSVs were found for CI computation.")

    out_df = pd.DataFrame(rows).sort_values(["experiment", "estimate_type", "metric"]).reset_index(drop=True)
    out_csv = out_dir / "metrics_with_ci.csv"
    out_df.to_csv(out_csv, index=False)

    now = datetime.now(timezone.utc).isoformat()
    summary = out_df.copy()
    summary["point"] = summary["point"].map(lambda x: f"{x:.4f}")
    summary["ci_low"] = summary["ci_low"].map(lambda x: f"{x:.4f}")
    summary["ci_high"] = summary["ci_high"].map(lambda x: f"{x:.4f}")
    summary["ci95"] = summary["ci_low"] + " .. " + summary["ci_high"]
    summary = summary[["experiment", "estimate_type", "metric", "point", "ci95", "n_folds"]]

    out_md = out_dir / "metrics_with_ci.md"
    md = [
        "# Metrics With 95% CI",
        "",
        f"Generated: `{now}`",
        "",
        "Method:",
        "- Leave-one-recording-out fold metrics are bootstrapped by resampling folds with replacement.",
        "- `macro`: mean of fold metrics.",
        "- `micro`: metrics from pooled confusion counts.",
        "",
        _to_markdown_table(summary),
        "",
        f"Full CSV: `{out_csv}`",
    ]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("== Step 18: Metrics CI ==")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
