#!/usr/bin/env python3
"""
Step 19 — Risk Prediction from Pre-Outcome Features
====================================================

Uses decision-stage features and SFC context signatures to predict
execution-stage THW risk, demonstrating predictive value of the framework.

Evaluation: Leave-One-Recording-Out Cross-Validation (LORO-CV).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs" / "reports"
STAGE_FEATURES = OUTPUTS / "Step 09" / "cutin_stage_features_merged.csv"
SFC_CODES = OUTPUTS / "Step 15A" / "sfc_binary_codes_long_hilbert_mirrored.csv"
OUT_DIR = OUTPUTS / "Step 19"


# ---------------------------------------------------------------------------
# Simple logistic regression (numpy only)
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogReg:
    def __init__(self, lr=0.05, n_iter=300, C=1.0):
        self.lr, self.n_iter, self.C = lr, n_iter, C
        self.w, self.b = None, 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        pw = n / (2.0 * max(y.sum(), 1))
        nw = n / (2.0 * max((1 - y).sum(), 1))
        sw = np.where(y == 1, pw, nw)
        for _ in range(self.n_iter):
            p = sigmoid(X @ self.w + self.b)
            self.w -= self.lr * ((X.T @ ((p - y) * sw)) / n + self.w / self.C)
            self.b -= self.lr * np.mean((p - y) * sw)

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)


# ---------------------------------------------------------------------------
# Fast decision stump ensemble
# ---------------------------------------------------------------------------
class StumpForest:
    def __init__(self, n_trees=50, seed=42):
        self.n_trees, self.seed = n_trees, seed
        self.stumps = []
        self.feature_importances_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.stumps = []
        imp = np.zeros(d)

        for _ in range(self.n_trees):
            idx = rng.choice(n, n, replace=True)
            fi = rng.randint(0, d)
            Xf, yb = X[idx, fi], y[idx]
            # Try 10 quantile-based thresholds
            thrs = np.percentile(Xf, np.linspace(10, 90, 10))
            best_g, best_t = -1, thrs[0]
            base_g = 2 * yb.mean() * (1 - yb.mean())
            for t in thrs:
                left, right = yb[Xf <= t], yb[Xf > t]
                if len(left) < 2 or len(right) < 2:
                    continue
                g = base_g - (len(left) / n * 2 * left.mean() * (1 - left.mean()) +
                              len(right) / n * 2 * right.mean() * (1 - right.mean()))
                if g > best_g:
                    best_g, best_t = g, t

            lp = yb[Xf <= best_t].mean() if (Xf <= best_t).any() else 0.5
            rp = yb[Xf > best_t].mean() if (Xf > best_t).any() else 0.5
            self.stumps.append((fi, best_t, lp, rp))
            imp[fi] += max(best_g, 0)

        self.feature_importances_ = imp / max(imp.sum(), 1e-10)

    def predict_proba(self, X):
        p = np.zeros(len(X))
        for fi, t, lp, rp in self.stumps:
            p += np.where(X[:, fi] <= t, lp, rp)
        return p / len(self.stumps)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y, yp, prob=None):
    tp = ((yp == 1) & (y == 1)).sum()
    fp = ((yp == 1) & (y == 0)).sum()
    fn = ((yp == 0) & (y == 1)).sum()
    tn = ((yp == 0) & (y == 0)).sum()
    pr = tp / max(tp + fp, 1)
    re = tp / max(tp + fn, 1)
    f1 = 2 * pr * re / max(pr + re, 1e-10)
    acc = (tp + tn) / len(y)

    auc = np.nan
    if prob is not None and y.sum() > 0 and (1 - y).sum() > 0:
        o = np.argsort(-prob)
        sy = y[o]
        tpr = np.cumsum(sy) / sy.sum()
        fpr = np.cumsum(1 - sy) / (1 - sy).sum()
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, 'trapezoid') else float(np.trapz(tpr, fpr))

    return dict(accuracy=round(acc, 4), precision=round(pr, 4), recall=round(re, 4),
                f1=round(f1, 4), auc=round(auc, 4) if not np.isnan(auc) else None)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(stage_features_path: Path, sfc_codes_path: Path):
    sf = pd.read_csv(stage_features_path)
    sf["risk_label"] = (sf["execution_thw_min"] < 0.7).astype(int)

    sfc = pd.read_csv(sfc_codes_path)
    sfc_d = sfc[sfc["stage"] == "decision"]

    def bits(c):
        return [(int(c) >> i) & 1 for i in range(16)]

    ev = (sfc_d.groupby(["event_id", "recording_id", "cutter_id", "t0_frame"])["code_mirrored"]
          .apply(lambda cs: np.mean([bits(c) for c in cs], axis=0)).reset_index())
    bc = [f"sfc_{i}" for i in range(16)]
    ev[bc] = pd.DataFrame(ev["code_mirrored"].tolist(), index=ev.index)
    ev.drop(columns=["code_mirrored"], inplace=True)

    sf["recording_id"] = sf["recording_id"].astype(str).str.zfill(2)
    ev["recording_id"] = ev["recording_id"].astype(str).str.zfill(2)
    # Join on the full event identity to avoid duplicating repeated cutters
    # that appear in multiple cut-ins within the same recording.
    m = sf.merge(
        ev,
        on=["recording_id", "cutter_id", "t0_frame"],
        how="inner",
        validate="one_to_one",
    )
    print(f"Events: {len(m)} ({m['risk_label'].sum()} risky, {m['risk_label'].mean():.1%})")
    return m, bc


def feature_groups(bc):
    kin = [
        "decision_dhw_min", "decision_thw_min", "decision_ttc_min",
        "decision_cutter_lat_v_abs_max", "decision_cutter_speed_mean",
        "decision_cutter_acc_mean", "decision_follower_speed_mean",
        "decision_follower_acc_mean", "decision_cutter_dy",
        "intention_dhw_min", "intention_thw_min", "intention_ttc_min",
        "intention_cutter_lat_v_abs_max", "intention_cutter_speed_mean",
        "intention_cutter_acc_mean", "intention_follower_speed_mean",
        "intention_follower_acc_mean", "intention_cutter_dy",
    ]
    return {"All (kin+SFC)": kin + bc, "Kinematic only": kin, "SFC only": bc}


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
        .replace("-", "_")
    )


def summarize_results(res: pd.DataFrame) -> dict[str, float]:
    auc = res["auc"].dropna()
    return {
        "mean_auc": float(auc.mean()),
        "std_auc": float(auc.std(ddof=1)),
        "mean_f1": float(res["f1"].mean()),
        "std_f1": float(res["f1"].std(ddof=1)),
        "mean_precision": float(res["precision"].mean()),
        "std_precision": float(res["precision"].std(ddof=1)),
        "mean_recall": float(res["recall"].mean()),
        "std_recall": float(res["recall"].std(ddof=1)),
        "mean_delta_f1": float(res["delta_f1"].mean()),
        "std_delta_f1": float(res["delta_f1"].std(ddof=1)),
    }


def sanitize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def wilcoxon_vs_chance(
    res: pd.DataFrame,
    *,
    chance: float = 0.5,
    label: str = "SFC only AUC > chance",
) -> dict[str, float | str | bool]:
    auc = res[["rec", "auc"]].dropna()
    test = wilcoxon(auc["auc"] - chance, alternative="greater", zero_method="wilcox")
    return {
        "comparison": label,
        "test": "wilcoxon_signed_rank",
        "metric": "auc",
        "alternative": "greater",
        "n_folds": int(len(auc)),
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "p_value_adj": float(test.pvalue),
        "correction": "none",
        "family": "single_test",
        "significant_0_05": bool(test.pvalue < 0.05),
        "reference": chance,
    }


def wilcoxon_paired(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str,
) -> dict[str, float | str | bool]:
    merged = left[["rec", "auc"]].merge(
        right[["rec", "auc"]],
        on="rec",
        suffixes=("_left", "_right"),
    )
    test = wilcoxon(
        merged["auc_left"],
        merged["auc_right"],
        alternative="two-sided",
        zero_method="wilcox",
    )
    return {
        "comparison": label,
        "test": "wilcoxon_signed_rank",
        "metric": "auc",
        "alternative": "two-sided",
        "n_folds": int(len(merged)),
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "p_value_adj": float(test.pvalue),
        "correction": "none",
        "family": "single_test",
        "significant_0_05": bool(test.pvalue < 0.05),
        "reference": np.nan,
    }


def bonferroni_adjust(
    rows: list[dict[str, float | str | bool]],
    *,
    family: str,
) -> list[dict[str, float | str | bool]]:
    m = max(len(rows), 1)
    out = []
    for row in rows:
        p_value = float(row["p_value"])
        row = dict(row)
        row["p_value_adj"] = min(p_value * m, 1.0)
        row["correction"] = f"Bonferroni ({m} tests)"
        row["family"] = family
        row["significant_0_05"] = bool(row["p_value_adj"] < 0.05)
        out.append(row)
    return out


def _merge_metric_by_fold(
    named_results: dict[str, pd.DataFrame],
    *,
    metric: str = "auc",
) -> pd.DataFrame:
    merged = None
    for name, res in named_results.items():
        col = _slug(name)
        frame = res[["rec", metric]].dropna().rename(columns={metric: col})
        merged = frame if merged is None else merged.merge(frame, on="rec", how="inner")
    if merged is None:
        raise ValueError("No fold-level results available for comparison.")
    return merged.sort_values("rec").reset_index(drop=True)


def friedman_ablation(
    named_results: dict[str, pd.DataFrame],
    *,
    metric: str = "auc",
) -> tuple[dict[str, float | str | bool], pd.DataFrame]:
    merged = _merge_metric_by_fold(named_results, metric=metric)
    columns = [_slug(name) for name in named_results]
    stat, p_value = friedmanchisquare(*(merged[col] for col in columns))
    ranks = merged[columns].rank(axis=1, method="average", ascending=True)
    mean_ranks = {
        f"mean_rank_{_slug(name)}": float(ranks[_slug(name)].mean())
        for name in named_results
    }
    row = {
        "comparison": "All ablation models",
        "test": "friedman",
        "metric": metric,
        "alternative": "two-sided",
        "n_folds": int(len(merged)),
        "statistic": float(stat),
        "p_value": float(p_value),
        "p_value_adj": float(p_value),
        "correction": "none",
        "family": "omnibus",
        "significant_0_05": bool(p_value < 0.05),
        "reference": np.nan,
        **mean_ranks,
    }
    rank_df = pd.DataFrame(
        {
            "feature_set": list(named_results.keys()),
            "mean_rank": [mean_ranks[f"mean_rank_{_slug(name)}"] for name in named_results],
        }
    ).sort_values("mean_rank", ascending=False)
    return row, rank_df


# ---------------------------------------------------------------------------
# LORO-CV
# ---------------------------------------------------------------------------
def loro_cv(df, features, model_type="ensemble"):
    recs = sorted(df["recording_id"].unique())
    rows = []
    imp_sum = np.zeros(len(features))
    n_folds = 0

    for rec in recs:
        te = df[df["recording_id"] == rec]
        tr = df[df["recording_id"] != rec]
        if len(te) < 3:
            continue

        Xtr = sanitize_features(tr, features).values.astype(float)
        ytr = tr["risk_label"].values.astype(float)
        Xte = sanitize_features(te, features).values.astype(float)
        yte = te["risk_label"].values.astype(float)

        if model_type == "logreg":
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
            clf = LogReg(lr=0.05, n_iter=300)
            clf.fit((Xtr - mu) / sd, ytr)
            prob = clf.predict_proba((Xte - mu) / sd)
        else:
            clf = StumpForest(n_trees=50, seed=42)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)
            imp_sum += clf.feature_importances_
            n_folds += 1

        # Find best threshold on train
        if model_type == "logreg":
            train_prob = clf.predict_proba((Xtr - mu) / sd)
        else:
            train_prob = clf.predict_proba(Xtr)

        best_thr, best_f1 = 0.5, 0
        for t in np.arange(0.15, 0.85, 0.05):
            yp = (train_prob >= t).astype(int)
            tp_ = ((yp == 1) & (ytr == 1)).sum()
            fp_ = ((yp == 1) & (ytr == 0)).sum()
            fn_ = ((yp == 0) & (ytr == 1)).sum()
            p_ = tp_ / max(tp_ + fp_, 1)
            r_ = tp_ / max(tp_ + fn_, 1)
            f_ = 2 * p_ * r_ / max(p_ + r_, 1e-10)
            if f_ > best_f1:
                best_f1, best_thr = f_, t

        pred = (prob >= best_thr).astype(int)
        m = metrics(yte, pred, prob)

        # Baseline
        maj = int(ytr.mean() >= 0.5)
        bl_m = metrics(yte, np.full_like(yte, maj))

        rows.append({"rec": rec, "n": len(te), "pos_rate": round(yte.mean(), 3),
                      "thr": round(best_thr, 2), **m,
                      "bl_f1": bl_m["f1"], "delta_f1": round(m["f1"] - bl_m["f1"], 4)})

    fi = None
    if n_folds > 0:
        fi = pd.DataFrame({"feature": features, "importance": imp_sum / n_folds})\
            .sort_values("importance", ascending=False)

    return pd.DataFrame(rows), fi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 19: Risk prediction from pre-outcome features.")
    parser.add_argument(
        "--stage-features",
        type=Path,
        default=STAGE_FEATURES,
        help="Merged stage-feature CSV to use for the prediction study.",
    )
    parser.add_argument(
        "--sfc-codes",
        type=Path,
        default=SFC_CODES,
        help="Mirror-normalized SFC long-code CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where Step 19 outputs should be written.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 19: Risk Prediction from Pre-Outcome Features")
    print("=" * 70)

    df, bc = load_data(args.stage_features, args.sfc_codes)
    fg = feature_groups(bc)

    # --- Run both models on all features ---
    print("\n=== LORO-CV: All features ===")
    for mt in ["logreg", "ensemble"]:
        res, fi = loro_cv(df, fg["All (kin+SFC)"], model_type=mt)
        name = "LogisticRegression" if mt == "logreg" else "StumpEnsemble"
        summary = summarize_results(res)
        print(f"\n  {name}:")
        print(f"    mean F1={summary['mean_f1']:.4f} (±{summary['std_f1']:.4f})")
        print(f"    mean AUC={summary['mean_auc']:.4f} (±{summary['std_auc']:.4f})")
        print(
            f"    mean Prec={summary['mean_precision']:.4f} (±{summary['std_precision']:.4f})"
            f"  Recall={summary['mean_recall']:.4f} (±{summary['std_recall']:.4f})"
        )
        print(f"    mean ΔF1 vs baseline={summary['mean_delta_f1']:.4f}")
        res.to_csv(args.out_dir / f"loro_{mt}.csv", index=False)
        pd.DataFrame([summary]).to_csv(args.out_dir / f"summary_{mt}.csv", index=False)
        if fi is not None:
            fi.to_csv(args.out_dir / f"importance_{mt}.csv", index=False)
            print(f"    Top features: {', '.join(fi.head(5)['feature'].tolist())}")

    # --- Ablation (ensemble only) ---
    print("\n=== Feature Ablation (StumpEnsemble) ===")
    ablation = []
    ablation_folds = {}
    for name, feats in fg.items():
        res, _ = loro_cv(df, feats, model_type="ensemble")
        ablation_folds[name] = res
        res.to_csv(args.out_dir / f"loro_ablation_{_slug(name)}.csv", index=False)
        summary = summarize_results(res)
        row = {
            "feature_set": name,
            "n_features": len(feats),
            "mean_f1": round(summary["mean_f1"], 4),
            "std_f1": round(summary["std_f1"], 4),
            "mean_auc": round(summary["mean_auc"], 4),
            "std_auc": round(summary["std_auc"], 4),
            "mean_prec": round(summary["mean_precision"], 4),
            "std_prec": round(summary["std_precision"], 4),
            "mean_recall": round(summary["mean_recall"], 4),
            "std_recall": round(summary["std_recall"], 4),
            "mean_delta_f1": round(summary["mean_delta_f1"], 4),
            "std_delta_f1": round(summary["std_delta_f1"], 4),
        }
        ablation.append(row)
        print(
            f"  {name}: F1={row['mean_f1']:.4f} (±{row['std_f1']:.4f}), "
            f"AUC={row['mean_auc']:.4f} (±{row['std_auc']:.4f}), "
            f"ΔF1={row['mean_delta_f1']:.4f}"
        )

    abl_df = pd.DataFrame(ablation)
    abl_df.to_csv(args.out_dir / "ablation.csv", index=False)

    omnibus_test, rank_df = friedman_ablation(ablation_folds, metric="auc")
    rank_df.to_csv(args.out_dir / "ablation_mean_ranks.csv", index=False)

    follow_up_tests = [
        wilcoxon_paired(
            ablation_folds["All (kin+SFC)"],
            ablation_folds["Kinematic only"],
            label="All vs Kinematic only",
        ),
        wilcoxon_paired(
            ablation_folds["All (kin+SFC)"],
            ablation_folds["SFC only"],
            label="All vs SFC only",
        ),
        wilcoxon_paired(
            ablation_folds["Kinematic only"],
            ablation_folds["SFC only"],
            label="Kinematic only vs SFC only",
        ),
        wilcoxon_vs_chance(
            ablation_folds["SFC only"],
            chance=0.5,
            label="SFC only AUC > chance",
        ),
    ]
    follow_up_tests = bonferroni_adjust(
        follow_up_tests,
        family="ablation_follow_up",
    )
    tests = [omnibus_test, *follow_up_tests]
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(args.out_dir / "ablation_tests.csv", index=False)

    print("\n=== Ablation Statistical Tests ===")
    for row in tests:
        print(
            f"  {row['comparison']} [{row['test']}]: statistic={row['statistic']:.4f}, "
            f"p={row['p_value']:.6g}, p_adj={row['p_value_adj']:.6g} ({row['alternative']})"
        )
    print("\n  Mean ranks (higher is better):")
    print(rank_df.to_string(index=False))

    # --- Correlations ---
    print("\n=== Top Feature-Risk Correlations ===")
    corrs = []
    for col in fg["All (kin+SFC)"]:
        v = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        if float(v.std(ddof=0)) == 0.0:
            c = 0.0
        else:
            c = float(np.corrcoef(v, df["risk_label"])[0, 1])
            if not np.isfinite(c):
                c = 0.0
        corrs.append({"feature": col, "corr": round(c, 4)})
    corr_df = pd.DataFrame(corrs).sort_values("corr", key=abs, ascending=False)
    corr_df.to_csv(args.out_dir / "correlations.csv", index=False)
    print(corr_df.head(10).to_string(index=False))

    print(f"\nAll outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
