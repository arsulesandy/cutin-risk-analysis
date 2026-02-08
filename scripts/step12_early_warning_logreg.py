from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldResult:
    heldout_recording: str
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_test: int


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int, float, float, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def _best_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Choose threshold on TRAIN set that maximizes F1.
    # We scan a fixed grid for stability/reproducibility.
    thresholds = np.linspace(0.05, 0.95, 91, dtype=float)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = y_prob >= thr
        *_counts, _p, _r, f1 = _compute_metrics(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 12: Early warning with logistic regression (leave-one-recording-out).")
    parser.add_argument(
        "--merged-csv",
        type=str,
        default="outputs/reports/step9_batch/cutin_stage_features_merged.csv",
    )
    parser.add_argument("--thw-risk", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default="outputs/reports/step12_warning_logreg")
    args = parser.parse_args()

    # Lazy import so the script fails with a clear message if sklearn isn't installed.
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for this step. Install it with: pip install scikit-learn"
        ) from e

    merged_csv = Path(args.merged_csv)
    if not merged_csv.exists():
        raise FileNotFoundError(f"Missing input: {merged_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_csv)

    # Risk label (same as Step 10/11)
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

    feature_cols = ["decision_lat_v", "decision_speed_delta", "decision_acc_delta", "decision_dy_abs"]

    keep = df.dropna(subset=["recording_id", "risk_thw", *feature_cols]).copy()
    keep["recording_id"] = keep["recording_id"].astype(str)

    if keep.empty:
        raise ValueError("No usable rows after dropping NaNs. Check feature column names.")

    X_all = keep[feature_cols].to_numpy(dtype=float)
    y_all = keep["risk_thw"].to_numpy(dtype=bool)

    # Model: standardized features + logistic regression (balanced classes)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight=None, solver="lbfgs")),
        ]
    )

    records = sorted(keep["recording_id"].unique().tolist())
    fold_results: list[FoldResult] = []

    print("== Step 12: Logistic regression early warning ==")
    print(f"Label: execution_thw_min < {args.thw_risk:.2f}s")
    print("Features:", ", ".join(feature_cols))
    print()

    for rid in records:
        train = keep[keep["recording_id"] != rid]
        test = keep[keep["recording_id"] == rid]

        X_tr = train[feature_cols].to_numpy(dtype=float)
        y_tr = train["risk_thw"].to_numpy(dtype=bool)

        X_te = test[feature_cols].to_numpy(dtype=float)
        y_te = test["risk_thw"].to_numpy(dtype=bool)

        model.fit(X_tr, y_tr)

        # Tune threshold on training set (maximize F1 on train)
        prob_tr = model.predict_proba(X_tr)[:, 1]
        thr = _best_threshold_for_f1(y_tr, prob_tr)

        # Evaluate on test set with that threshold
        prob_te = model.predict_proba(X_te)[:, 1]
        pred_te = prob_te >= thr

        tp, fp, fn, tn, precision, recall, f1 = _compute_metrics(y_te, pred_te)

        fold_results.append(
            FoldResult(
                heldout_recording=rid,
                threshold=float(thr),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                n_test=int(len(y_te)),
            )
        )

    loo = pd.DataFrame([r.__dict__ for r in fold_results])
    out_csv = out_dir / "leave_one_recording_out_logreg.csv"
    loo.to_csv(out_csv, index=False)

    print("Leave-one-recording-out results:")
    print(loo.to_string(index=False))

    print("\nMacro averages:")
    print(
        {
            "precision": float(loo["precision"].mean()),
            "recall": float(loo["recall"].mean()),
            "f1": float(loo["f1"].mean()),
        }
    )

    # Fit on all data and export coefficients for interpretability
    model.fit(X_all, y_all)
    clf = model.named_steps["clf"]
    coefs = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": clf.coef_[0],
        }
    ).sort_values("coef", ascending=False)

    coef_csv = out_dir / "logreg_coefficients.csv"
    coefs.to_csv(coef_csv, index=False)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {coef_csv}")


if __name__ == "__main__":
    main()
