"""Step 13A: tuned logistic-regression early warning model and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from cutin_risk.io.progress import iter_with_progress
from cutin_risk.io.step_reports import mirror_file_to_step, write_step_markdown
from cutin_risk.paths import output_path
from cutin_risk.thesis_config import (
    thesis_bool,
    thesis_float,
    thesis_int,
    thesis_optional_float,
    thesis_str,
)


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def fbeta(self, beta: float) -> float:
        p = self.precision
        r = self.recall
        if p == 0.0 and r == 0.0:
            return 0.0
        b2 = beta * beta
        denom = (b2 * p + r)
        return (1.0 + b2) * p * r / denom if denom else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Confusion:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def pick_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        beta: float,
        min_recall: float | None,
) -> float:
    """
    Choose a probability threshold using TRAIN data.

    - If min_recall is provided, maximize precision subject to recall >= min_recall.
    - Otherwise, maximize F_beta (beta < 1 favors precision, beta > 1 favors recall).
    """
    thresholds = np.linspace(
        thesis_float("step13a.threshold_grid_min", 0.05),
        thesis_float("step13a.threshold_grid_max", 0.95),
        thesis_int("step13a.threshold_grid_steps", 91, min_value=2),
        dtype=float,
    )

    best_thr = 0.5
    best_key = None

    for thr in thresholds:
        pred = y_prob >= thr
        c = confusion(y_true, pred)
        p = c.precision
        r = c.recall

        if min_recall is not None:
            if r < float(min_recall):
                continue
            key = (p, r)  # prioritize precision, then recall
        else:
            key = (c.fbeta(beta), r, p)  # stable tie-breakers

        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)

    # If recall constraint was too strict and nothing matched, fall back to F_beta.
    if best_key is None and min_recall is not None:
        return pick_threshold(y_true, y_prob, beta=beta, min_recall=None)

    return best_thr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 13A: tuned logistic regression early-warning (precision-aware thresholding)."
    )
    parser.add_argument(
        "--merged-csv",
        type=str,
        default=str(output_path("reports/step9_batch/cutin_stage_features_merged.csv")),
    )
    parser.add_argument("--thw-risk", type=float, default=thesis_float("risk_label.thw_risk", 0.7, min_value=0.0))

    parser.add_argument(
        "--feature-set",
        choices=["core", "full"],
        default=thesis_str("step13a.feature_set", "full", allowed={"core", "full"}),
        help="core=2 features (lat_v, speed_delta). full=adds acc_delta and dy_abs.",
    )
    parser.add_argument(
        "--with-interaction",
        action=argparse.BooleanOptionalAction,
        default=thesis_bool("step13a.with_interaction", False),
        help="Add decision_lat_v * decision_speed_delta as an extra feature.",
    )

    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default=thesis_str("step13a.class_weight", "none", allowed={"none", "balanced"}),
        help="Use class_weight=None (default) or class_weight='balanced'.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=thesis_float("step13a.c", 1.0, min_value=1e-12),
        help="Inverse regularization strength.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=thesis_float("step13a.beta", 0.5, min_value=1e-12),
        help="Optimize F_beta on train when min_recall is not set (beta<1 favors precision).",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=thesis_optional_float("step13a.min_recall", None, min_value=0.0),
        help="If set, choose threshold to maximize precision while keeping recall >= min_recall (train set).",
    )

    parser.add_argument("--out-dir", type=str, default=str(output_path("reports/step13a_warning_logreg")))
    args = parser.parse_args()

    # Lazy import for clear error message if sklearn is missing.
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn") from e

    merged_csv = Path(args.merged_csv)
    if not merged_csv.exists():
        raise FileNotFoundError(f"Missing input: {merged_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_csv)

    # Risk label (same definition as Step 10/11/12)
    df["risk_thw"] = pd.to_numeric(df["execution_thw_min"], errors="coerce") < float(args.thw_risk)

    # Decision-stage features (pre cut-in)
    df["decision_lat_v"] = pd.to_numeric(df["decision_cutter_lat_v_abs_max"], errors="coerce")
    df["decision_speed_delta"] = (
            pd.to_numeric(df["decision_cutter_speed_mean"], errors="coerce")
            - pd.to_numeric(df["decision_follower_speed_mean"], errors="coerce")
    )

    df["decision_acc_delta"] = (
            pd.to_numeric(df["decision_cutter_acc_mean"], errors="coerce")
            - pd.to_numeric(df["decision_follower_acc_mean"], errors="coerce")
    )
    df["decision_dy_abs"] = pd.to_numeric(df["decision_cutter_dy"], errors="coerce").abs()

    feature_cols = ["decision_lat_v", "decision_speed_delta"]
    if args.feature_set == "full":
        feature_cols += ["decision_acc_delta", "decision_dy_abs"]

    if args.with_interaction:
        df["decision_lat_v_x_speed_delta"] = df["decision_lat_v"] * df["decision_speed_delta"]
        feature_cols.append("decision_lat_v_x_speed_delta")

    keep = df.dropna(subset=["recording_id", "risk_thw", *feature_cols]).copy()
    keep["recording_id"] = keep["recording_id"].astype(str)

    if keep.empty:
        raise ValueError("No usable rows after dropping NaNs. Check your merged CSV and column names.")

    class_weight = None if args.class_weight == "none" else "balanced"
    logreg_solver = thesis_str(
        "step13a.logreg_solver",
        "lbfgs",
        allowed={"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"},
    )
    logreg_max_iter = thesis_int("step13a.logreg_max_iter", 4000, min_value=1)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=logreg_max_iter,
                    class_weight=class_weight,
                    C=float(args.C),
                    solver=logreg_solver,
                ),
            ),
        ]
    )

    records = sorted(keep["recording_id"].unique().tolist())
    fold_rows = []

    micro_tp = micro_fp = micro_fn = micro_tn = 0

    print("== Step 13A: Tuned logistic regression early warning ==")
    print(f"Label: execution_thw_min < {args.thw_risk:.2f}s")
    print("Features:", ", ".join(feature_cols))
    print(f"class_weight: {args.class_weight} | C: {args.C} | beta: {args.beta} | min_recall: {args.min_recall}")
    print()

    for _, _, rid in iter_with_progress(
        records,
        label="Step 13A LOO folds",
        item_name="heldout_recording",
    ):
        train = keep[keep["recording_id"] != rid]
        test = keep[keep["recording_id"] == rid]

        X_tr = train[feature_cols].to_numpy(dtype=float)
        y_tr = train["risk_thw"].to_numpy(dtype=bool)

        X_te = test[feature_cols].to_numpy(dtype=float)
        y_te = test["risk_thw"].to_numpy(dtype=bool)

        model.fit(X_tr, y_tr)

        prob_tr = model.predict_proba(X_tr)[:, 1]
        thr = pick_threshold(y_tr, prob_tr, beta=float(args.beta), min_recall=args.min_recall)

        prob_te = model.predict_proba(X_te)[:, 1]
        pred_te = prob_te >= thr

        c = confusion(y_te, pred_te)

        micro_tp += c.tp
        micro_fp += c.fp
        micro_fn += c.fn
        micro_tn += c.tn

        fold_rows.append(
            {
                "heldout_recording": rid,
                "threshold": float(thr),
                "precision": float(c.precision),
                "recall": float(c.recall),
                "fbeta": float(c.fbeta(float(args.beta))),
                "tp": c.tp,
                "fp": c.fp,
                "fn": c.fn,
                "tn": c.tn,
                "n_test": int(len(y_te)),
            }
        )

    loo = pd.DataFrame(fold_rows)
    out_csv = out_dir / "leave_one_recording_out_step13a.csv"
    loo.to_csv(out_csv, index=False)

    macro_precision = float(loo["precision"].mean())
    macro_recall = float(loo["recall"].mean())
    macro_fbeta = float(loo["fbeta"].mean())

    # Micro (pooled) metrics
    micro = Confusion(tp=micro_tp, fp=micro_fp, fn=micro_fn, tn=micro_tn)
    micro_precision = micro.precision
    micro_recall = micro.recall
    micro_fbeta = micro.fbeta(float(args.beta))

    print("Leave-one-recording-out results:")
    print(loo.to_string(index=False))

    print("\nMacro averages:")
    print({"precision": macro_precision, "recall": macro_recall, f"f{args.beta}": macro_fbeta})

    print("\nMicro (pooled) totals:")
    print(
        {
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "tn": micro_tn,
            "precision": micro_precision,
            "recall": micro_recall,
            f"f{args.beta}": micro_fbeta,
        }
    )

    # Fit on all data and export coefficients (interpretability)
    X_all = keep[feature_cols].to_numpy(dtype=float)
    y_all = keep["risk_thw"].to_numpy(dtype=bool)
    model.fit(X_all, y_all)

    clf = model.named_steps["clf"]
    coefs = pd.DataFrame({"feature": feature_cols, "coef": clf.coef_[0]}).sort_values("coef", ascending=False)
    coef_csv = out_dir / "logreg_coefficients_step13a.csv"
    coefs.to_csv(coef_csv, index=False)

    canonical_loo = mirror_file_to_step(out_csv, "13A")
    canonical_coef = mirror_file_to_step(coef_csv, "13A")
    details_md = write_step_markdown(
        "13A",
        "tuned_logreg_details.md",
        [
            "# Step 13A Tuned Logistic Regression",
            "",
            f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Input merged CSV: `{merged_csv}`",
            f"- THW risk threshold: `{float(args.thw_risk):.3f}`",
            f"- Feature set: `{args.feature_set}`",
            f"- Interaction enabled: `{bool(args.with_interaction)}`",
            f"- Features: `{', '.join(feature_cols)}`",
            f"- Class weight: `{args.class_weight}`",
            f"- C: `{float(args.C):.6f}`",
            f"- Beta: `{float(args.beta):.6f}`",
            f"- Min recall constraint: `{args.min_recall}`",
            f"- LOO folds: `{len(loo)}`",
            f"- Macro precision: `{macro_precision:.6f}`",
            f"- Macro recall: `{macro_recall:.6f}`",
            f"- Macro F-beta: `{macro_fbeta:.6f}`",
            f"- Micro precision: `{micro_precision:.6f}`",
            f"- Micro recall: `{micro_recall:.6f}`",
            f"- Micro F-beta: `{micro_fbeta:.6f}`",
            f"- LOO CSV: `{canonical_loo}`",
            f"- Coefficients CSV: `{canonical_coef}`",
        ],
    )

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {coef_csv}")
    print(f"Saved: {details_md}")


if __name__ == "__main__":
    main()
