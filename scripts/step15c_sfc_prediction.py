"""Step 15C: train/evaluate prediction models from binary or weighted SFC features."""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from cutin_risk.paths import output_path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * (precision * recall) / denom


def find_threshold(y_true: np.ndarray, prob: np.ndarray, *, beta: float, min_recall: float | None) -> float:
    best_thr = 0.5
    best_score = -1.0

    for thr in np.linspace(0.05, 0.95, 91):
        pred = prob >= thr
        m = compute_metrics(y_true, pred)
        if min_recall is not None and m["recall"] < float(min_recall):
            continue
        score = fbeta(m["precision"], m["recall"], beta)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr


def build_features_from_binary_long(df: pd.DataFrame, *, stage: str, min_frames: int) -> pd.DataFrame:
    """
    Binary-long input: one row per frame with a 16-bit code.
    Produces per-event features from the chosen stage.
    """
    need = {"recording_id", "risk_thw", "stage", "code"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"binary-long CSV missing columns: {sorted(missing)}")

    if "event_id" not in df.columns:
        if not {"recording_id", "cutter_id", "t0_frame"} <= set(df.columns):
            raise ValueError("binary-long CSV needs event_id or (recording_id,cutter_id,t0_frame)")
        df = df.copy()
        df["event_id"] = df.groupby(["recording_id", "cutter_id", "t0_frame"], sort=False).ngroup() + 1

    d = df[df["stage"] == stage].copy()
    d["code"] = pd.to_numeric(d["code"], errors="coerce")
    d = d.dropna(subset=["code"]).copy()
    d["code"] = d["code"].astype(int)

    rows = []
    for (event_id, rec_id), g in d.groupby(["event_id", "recording_id"], sort=False):
        codes = g["code"].to_numpy(dtype=np.uint16)
        if len(codes) < min_frames:
            continue

        # bit means (16 features)
        bits = ((codes[:, None] >> np.arange(16)) & 1).astype(np.float32)
        bit_mean = bits.mean(axis=0)

        # sequence dynamics
        transitions = float(np.mean(codes[1:] != codes[:-1])) if len(codes) > 1 else 0.0
        unique_codes = int(len(np.unique(codes)))

        # entropy of codes
        vals, counts = np.unique(codes, return_counts=True)
        p = counts / counts.sum()
        ent = float(-(p * np.log2(p)).sum()) if len(p) else 0.0

        row = {
            "event_id": int(event_id),
            "recording_id": str(rec_id),
            "risk_thw": bool(g["risk_thw"].iloc[0]),
            "n_frames": int(len(codes)),
            "transitions": transitions,
            "unique_codes": unique_codes,
            "entropy": ent,
        }
        for k in range(16):
            row[f"bit{k:02d}_mean"] = float(bit_mean[k])
        rows.append(row)

    return pd.DataFrame(rows)


def build_features_from_weighted_wide(df: pd.DataFrame, *, stage: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Weighted-wide input: one row per event with columns like:
      decision_sfc_v00_mean ... decision_sfc_v15_mean
    """
    need = {"recording_id", "risk_thw"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"weighted-wide CSV missing columns: {sorted(missing)}")

    prefix = f"{stage}_sfc_v"
    feat_cols = [c for c in df.columns if c.startswith(prefix) and c.endswith("_mean")]
    if not feat_cols:
        raise ValueError(f"No columns found with prefix '{prefix}' and suffix '_mean'")

    out = df[["recording_id", "risk_thw"] + feat_cols].copy()
    return out, feat_cols


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 15C: Predict execution-stage risk using decision-stage SFC signature.")
    ap.add_argument("--input-type", choices=["binary-long", "weighted-wide"], required=True)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--stage", default="decision")

    ap.add_argument("--min-frames", type=int, default=10)  # binary-long only

    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none")

    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--min-recall", type=float, default=None)

    ap.add_argument("--threshold-strategy", choices=["fixed", "train_opt"], default="train_opt")
    ap.add_argument("--fixed-threshold", type=float, default=0.5)

    ap.add_argument("--out-dir", default=str(output_path("reports/step15c_sfc_prediction")))
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input: {input_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if args.input_type == "binary-long":
        feats = build_features_from_binary_long(df, stage=args.stage, min_frames=int(args.min_frames))
        feature_cols = [c for c in feats.columns if c.startswith("bit") or c in ("transitions", "unique_codes", "entropy")]
    else:
        feats, feature_cols = build_features_from_weighted_wide(df, stage=args.stage)

    if feats.empty:
        raise ValueError("No usable events after feature extraction/filtering.")

    y_all = feats["risk_thw"].to_numpy(dtype=bool)
    recs = feats["recording_id"].astype(str).to_numpy()

    class_weight = None if args.class_weight == "none" else "balanced"

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=float(args.C), max_iter=2000, class_weight=class_weight)),
        ]
    )

    rows = []
    micro = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for held in sorted(np.unique(recs)):
        train_mask = recs != held
        test_mask = recs == held

        X_tr = feats.loc[train_mask, feature_cols].to_numpy(dtype=float)
        y_tr = y_all[train_mask]

        X_te = feats.loc[test_mask, feature_cols].to_numpy(dtype=float)
        y_te = y_all[test_mask]

        if len(y_te) == 0:
            continue

        pipe.fit(X_tr, y_tr)

        prob_tr = pipe.predict_proba(X_tr)[:, 1]
        prob_te = pipe.predict_proba(X_te)[:, 1]

        if args.threshold_strategy == "train_opt":
            thr = find_threshold(y_tr, prob_tr, beta=float(args.beta), min_recall=args.min_recall)
        else:
            thr = float(args.fixed_threshold)

        pred = prob_te >= thr
        m = compute_metrics(y_te, pred)

        micro["tp"] += m["tp"]
        micro["fp"] += m["fp"]
        micro["fn"] += m["fn"]
        micro["tn"] += m["tn"]

        rows.append(
            {
                "heldout_recording": held,
                "threshold": float(thr),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "fn": int(m["fn"]),
                "tn": int(m["tn"]),
                "n_test": int(len(y_te)),
            }
        )

    loo = pd.DataFrame(rows)
    out_loo = out_dir / "leave_one_recording_out.csv"
    loo.to_csv(out_loo, index=False)

    micro_metrics = compute_metrics(
        np.array([True] * micro["tp"] + [False] * micro["fp"] + [True] * micro["fn"] + [False] * micro["tn"]),
        np.array([True] * micro["tp"] + [True] * micro["fp"] + [False] * micro["fn"] + [False] * micro["tn"]),
    )

    print("== Step 15C: SFC prediction ==")
    print("Input:", input_csv)
    print("Saved:", out_loo)
    print("\nLOO results:")
    print(loo.to_string(index=False))
    print("\nMacro averages:", {
        "precision": float(loo["precision"].mean()),
        "recall": float(loo["recall"].mean()),
        "f1": float(loo["f1"].mean()),
    })
    print("\nMicro totals:", micro | {
        "precision": float(micro_metrics["precision"]),
        "recall": float(micro_metrics["recall"]),
        "f1": float(micro_metrics["f1"]),
    })


if __name__ == "__main__":
    main()
