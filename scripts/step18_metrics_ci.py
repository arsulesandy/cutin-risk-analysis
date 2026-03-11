"""Step 18: bootstrap confidence intervals for thesis-facing Step 07 metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.io.markdown import markdown_table
from cutin_risk.io.step_reports import mirror_file_to_step
from cutin_risk.paths import output_path
from cutin_risk.thesis_config import thesis_int


def _bootstrap_mean_ci(
    df: pd.DataFrame,
    metrics: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, tuple[float, float]]:
    n = len(df)
    idx = np.arange(n)
    out: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        vals = np.empty(n_boot, dtype=float)
        arr = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        for i in range(n_boot):
            s = rng.choice(idx, size=n, replace=True)
            vals[i] = float(np.nanmean(arr[s]))
        out[metric] = (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
    return out


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns.tolist()]
    rows = df.astype(str).values.tolist()
    right_align_cols = {"point", "ci95", "ci_half_width", "n_recordings"}
    align = ["right" if h in right_align_cols else "left" for h in headers]
    return markdown_table(headers=headers, rows=rows, align=align)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 18: Bootstrap confidence intervals for Step 07 recording means.")
    ap.add_argument("--n-bootstrap", type=int, default=thesis_int("step18.n_bootstrap", 3000, min_value=1))
    ap.add_argument("--seed", type=int, default=thesis_int("step18.seed", 7))
    ap.add_argument(
        "--step07-csv",
        type=str,
        default=str(output_path("reports/Step 07/xy_lane_pipeline_metrics_by_recording.csv")),
    )
    ap.add_argument("--out-dir", type=str, default=str(output_path("reports/final")))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_csv = Path(args.step07_csv)
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing Step 07 metrics CSV: {source_csv}")

    df = pd.read_csv(source_csv)
    metrics = [
        "preceding_overall_accuracy",
        "following_overall_accuracy",
        "cutin_precision",
        "cutin_recall",
        "cutin_f1",
    ]
    missing = [metric for metric in metrics if metric not in df.columns]
    if missing:
        raise ValueError(f"Step 07 CSV is missing required columns: {', '.join(missing)}")

    d = df[["recording_id", *metrics]].copy()
    for metric in metrics:
        d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=metrics).copy()
    if d.empty:
        raise ValueError("No usable Step 07 recording rows were found for CI computation.")

    rng = np.random.default_rng(int(args.seed))
    point = {metric: float(d[metric].mean()) for metric in metrics}
    ci = _bootstrap_mean_ci(d, metrics, rng, int(args.n_bootstrap))

    rows: list[dict[str, object]] = []
    for metric in metrics:
        ci_low, ci_high = ci[metric]
        rows.append(
            {
                "metric": metric,
                "point": point[metric],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_half_width": max(point[metric] - ci_low, ci_high - point[metric]),
                "n_recordings": int(len(d)),
                "n_bootstrap": int(args.n_bootstrap),
                "source_csv": str(source_csv),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)
    out_csv = out_dir / "metrics_with_ci.csv"
    out_df.to_csv(out_csv, index=False)

    now = datetime.now(timezone.utc).isoformat()
    summary = out_df.copy()
    summary["point"] = summary["point"].map(lambda x: f"{x:.4f}")
    summary["ci_low"] = summary["ci_low"].map(lambda x: f"{x:.4f}")
    summary["ci_high"] = summary["ci_high"].map(lambda x: f"{x:.4f}")
    summary["ci_half_width"] = summary["ci_half_width"].map(lambda x: f"{x:.2e}")
    summary["ci95"] = summary["ci_low"] + " .. " + summary["ci_high"]
    summary = summary[["metric", "point", "ci95", "ci_half_width", "n_recordings"]]

    out_md = out_dir / "metrics_with_ci.md"
    md = [
        "# Step 07 Metrics With 95% CI",
        "",
        f"Generated: `{now}`",
        "",
        "Method:",
        "- Step 07 per-recording metrics are bootstrapped by resampling recordings with replacement.",
        "- Point estimates are arithmetic means over recordings.",
        "- `ci_half_width` is the larger of the upper and lower absolute deviations from the point estimate.",
        "",
        _to_markdown_table(summary),
        "",
        f"Source CSV: `{source_csv}`",
        f"Full CSV: `{out_csv}`",
    ]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    canonical_csv = mirror_file_to_step(out_csv, 18)
    canonical_md = mirror_file_to_step(out_md, 18)

    print("== Step 18: Step 07 metrics CI ==")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Mirrored: {canonical_csv}")
    print(f"Mirrored: {canonical_md}")


if __name__ == "__main__":
    main()
