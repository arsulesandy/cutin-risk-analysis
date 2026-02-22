"""Step 18: bootstrap confidence intervals for per-fold and micro metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.io.markdown import markdown_table
from cutin_risk.io.step_reports import mirror_file_to_step
from cutin_risk.paths import output_path
from cutin_risk.thesis_config import thesis_int


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


def _baseline_metrics_from_existing_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Use baseline confusion counts already produced by upstream scripts when available.
    """
    raw_cols = ["baseline_tp", "baseline_fp", "baseline_fn", "baseline_tn"]
    if not (set(raw_cols) <= set(df.columns)):
        return None

    base = df[raw_cols].copy()
    for c in raw_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    if base[raw_cols].isna().any().any():
        return None

    rows: list[dict[str, float]] = []
    for _, r in base.iterrows():
        rows.append(
            _micro_from_counts(
                tp=float(r["baseline_tp"]),
                fp=float(r["baseline_fp"]),
                fn=float(r["baseline_fn"]),
                tn=float(r["baseline_tn"]),
            )
        )
    return pd.DataFrame(rows)


def _majority_baseline_from_train_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a train-majority baseline per fold from confusion counts.

    Assumes leave-one-fold-out partitioning where each fold appears once as test.
    For each fold:
      - infer test positives = tp + fn and test negatives = tn + fp
      - infer train class counts by subtracting held-out counts from totals
      - predict the training-majority class on the held-out fold
    """
    pos_test = (pd.to_numeric(df["tp"], errors="coerce") + pd.to_numeric(df["fn"], errors="coerce")).to_numpy(
        dtype=float
    )
    neg_test = (pd.to_numeric(df["tn"], errors="coerce") + pd.to_numeric(df["fp"], errors="coerce")).to_numpy(
        dtype=float
    )
    pos_total = float(np.nansum(pos_test))
    neg_total = float(np.nansum(neg_test))

    rows: list[dict[str, float]] = []
    for pos, neg in zip(pos_test.tolist(), neg_test.tolist()):
        train_pos = pos_total - float(pos)
        train_neg = neg_total - float(neg)
        predict_positive = train_pos >= train_neg
        if predict_positive:
            tp_b, fp_b, fn_b, tn_b = float(pos), float(neg), 0.0, 0.0
        else:
            tp_b, fp_b, fn_b, tn_b = 0.0, 0.0, float(pos), float(neg)
        rows.append(_micro_from_counts(tp_b, fp_b, fn_b, tn_b))
    return pd.DataFrame(rows)


def _majority_baseline_fold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    existing = _baseline_metrics_from_existing_columns(df)
    if existing is not None:
        return existing
    return _majority_baseline_from_train_counts(df)


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns.tolist()]
    rows = df.astype(str).values.tolist()
    right_align_cols = {"point", "ci95", "n_folds"}
    align = [
        "right" if (h in right_align_cols or h.endswith("_f1") or h.startswith("delta_")) else "left"
        for h in headers
    ]
    return markdown_table(headers=headers, rows=rows, align=align)


@dataclass(frozen=True)
class ExperimentTarget:
    experiment: str
    csv_path: Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 18: Bootstrap confidence intervals for LOO metrics.")
    ap.add_argument("--n-bootstrap", type=int, default=thesis_int("step18.n_bootstrap", 3000, min_value=1))
    ap.add_argument("--seed", type=int, default=thesis_int("step18.seed", 7))
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
    comparison_rows: list[dict[str, object]] = []
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

        d_base = _majority_baseline_fold_metrics(d)
        macro_point_base = {m: float(d_base[m].mean()) for m in metrics}
        macro_ci_base = _bootstrap_macro(d_base, metrics, rng, int(args.n_bootstrap))
        micro_point_base = _micro_from_counts(
            tp=float(d_base["tp"].sum()),
            fp=float(d_base["fp"].sum()),
            fn=float(d_base["fn"].sum()),
            tn=float(d_base["tn"].sum()),
        )
        micro_ci_base = _bootstrap_micro(d_base, rng, int(args.n_bootstrap))

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
            rows.append(
                {
                    "experiment": f"{t.experiment}__majority_baseline",
                    "source_csv": str(t.csv_path),
                    "n_folds": int(len(d)),
                    "estimate_type": "macro",
                    "metric": m,
                    "point": macro_point_base[m],
                    "ci_low": macro_ci_base[m][0],
                    "ci_high": macro_ci_base[m][1],
                    "n_bootstrap": int(args.n_bootstrap),
                }
            )
            rows.append(
                {
                    "experiment": f"{t.experiment}__majority_baseline",
                    "source_csv": str(t.csv_path),
                    "n_folds": int(len(d)),
                    "estimate_type": "micro",
                    "metric": m,
                    "point": micro_point_base[m],
                    "ci_low": micro_ci_base[m][0],
                    "ci_high": micro_ci_base[m][1],
                    "n_bootstrap": int(args.n_bootstrap),
                }
            )

        comparison_rows.append(
            {
                "experiment": t.experiment,
                "macro_f1_model": macro_point["f1"],
                "macro_f1_majority_baseline": macro_point_base["f1"],
                "delta_macro_f1": macro_point["f1"] - macro_point_base["f1"],
                "micro_f1_model": micro_point["f1"],
                "micro_f1_majority_baseline": micro_point_base["f1"],
                "delta_micro_f1": micro_point["f1"] - micro_point_base["f1"],
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

    comp_df = pd.DataFrame(comparison_rows).sort_values("experiment").reset_index(drop=True)
    comp_fmt = comp_df.copy()
    for c in [
        "macro_f1_model",
        "macro_f1_majority_baseline",
        "delta_macro_f1",
        "micro_f1_model",
        "micro_f1_majority_baseline",
        "delta_micro_f1",
    ]:
        comp_fmt[c] = comp_fmt[c].map(lambda x: f"{x:.4f}")

    non_improving = comp_df[
        (comp_df["delta_macro_f1"] <= 0.0) | (comp_df["delta_micro_f1"] <= 0.0)
    ].copy()
    non_improving_fmt = non_improving.copy()
    for c in [
        "macro_f1_model",
        "macro_f1_majority_baseline",
        "delta_macro_f1",
        "micro_f1_model",
        "micro_f1_majority_baseline",
        "delta_micro_f1",
    ]:
        if c in non_improving_fmt.columns:
            non_improving_fmt[c] = non_improving_fmt[c].map(lambda x: f"{x:.4f}")

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
        "- `__majority_baseline`: per fold, predict the training-fold majority class.",
        "- If baseline confusion columns exist in source CSV, those are used directly.",
        "",
        _to_markdown_table(summary),
        "",
        "## Model vs Majority Baseline (F1)",
        "",
        _to_markdown_table(comp_fmt),
    ]
    if not non_improving_fmt.empty:
        md.extend(
            [
                "",
                "## Non-improving Experiments vs Baseline",
                "",
                "These experiments do not beat the majority baseline on F1 and should be treated as negative findings.",
                "",
                _to_markdown_table(non_improving_fmt),
            ]
        )
    md.extend(
        [
            "",
            f"Full CSV: `{out_csv}`",
        ]
    )
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    canonical_csv = mirror_file_to_step(out_csv, 18)
    canonical_md = mirror_file_to_step(out_md, 18)

    print("== Step 18: Metrics CI ==")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Mirrored: {canonical_csv}")
    print(f"Mirrored: {canonical_md}")


if __name__ == "__main__":
    main()
