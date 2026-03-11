"""Step 15D: derive interpretable cut-in context archetypes from mirrored binary SFC signatures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from cutin_risk.encoding.sfc_binary import decode_grid_4x4_bits
from cutin_risk.io.step_reports import mirror_file_to_step, step_reports_dir, write_step_markdown
from cutin_risk.paths import output_path
from cutin_risk.thesis_config import thesis_bool, thesis_float, thesis_int, thesis_str

STEP_ID = "15D"

CELL_NAMES = {
    "g00": "left_preceding",
    "g01": "same_preceding",
    "g02": "right_preceding",
    "g10": "left_alongside",
    "g11": "cutter",
    "g12": "right_alongside",
    "g20": "left_following",
    "g21": "same_following",
    "g22": "right_following",
}


def normalize_recording_id(v: object) -> str:
    s = str(v).strip()
    return f"{int(s):02d}" if s.isdigit() else s


def _load_pyplot() -> object:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_stage_events(events_csv: Path, risk_thw: float) -> pd.DataFrame:
    events = pd.read_csv(events_csv).copy()
    required = {
        "recording_id",
        "cutter_id",
        "follower_id",
        "t0_frame",
        "from_lane",
        "to_lane",
        "execution_thw_min",
        "execution_ttc_min",
        "execution_dhw_min",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events CSV missing columns: {sorted(missing)}")

    events["recording_id"] = events["recording_id"].apply(normalize_recording_id)
    for col in ["cutter_id", "follower_id", "t0_frame", "from_lane", "to_lane"]:
        events[col] = pd.to_numeric(events[col], errors="coerce").astype("Int64")
    for col in ["execution_thw_min", "execution_ttc_min", "execution_dhw_min"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")

    events = events.dropna(
        subset=[
            "recording_id",
            "cutter_id",
            "follower_id",
            "t0_frame",
            "from_lane",
            "to_lane",
            "execution_thw_min",
        ]
    ).copy()
    events["risk_thw"] = events["execution_thw_min"] < float(risk_thw)
    key_cols = ["recording_id", "cutter_id", "t0_frame"]
    events = events.sort_values(key_cols, kind="mergesort")
    dup_mask = events.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        raise ValueError(
            "events CSV contains non-unique (recording_id, cutter_id, t0_frame) keys; "
            "cannot build event-level archetypes safely."
        )
    return events


def _decode_binary_stage_features(codes_csv: Path, stage: str) -> pd.DataFrame:
    codes = pd.read_csv(codes_csv).copy()
    required = {"recording_id", "cutter_id", "t0_frame", "stage", "frame", "sfc_order", "code"}
    missing = required - set(codes.columns)
    if missing:
        raise ValueError(f"codes CSV missing columns: {sorted(missing)}")

    codes["recording_id"] = codes["recording_id"].apply(normalize_recording_id)
    codes["stage"] = codes["stage"].astype(str)
    codes["cutter_id"] = pd.to_numeric(codes["cutter_id"], errors="coerce").astype("Int64")
    codes["t0_frame"] = pd.to_numeric(codes["t0_frame"], errors="coerce").astype("Int64")
    codes["frame"] = pd.to_numeric(codes["frame"], errors="coerce").astype("Int64")
    codes["code"] = pd.to_numeric(codes["code"], errors="coerce").astype("Int64")
    codes = codes.dropna(subset=["cutter_id", "t0_frame", "frame", "code"]).copy()
    codes = codes.loc[codes["stage"] == stage].copy()
    if codes.empty:
        raise ValueError(f"No rows found for stage={stage!r} in {codes_csv}")

    cache: dict[tuple[str, int], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for row in codes.itertuples(index=False):
        order = str(row.sfc_order)
        code_value = int(row.code)
        key = (order, code_value)
        g3 = cache.get(key)
        if g3 is None:
            g4 = decode_grid_4x4_bits(code_value, order=order)
            g3 = np.asarray(g4[:3, :3], dtype=float)
            cache[key] = g3
        out = {
            "recording_id": str(row.recording_id),
            "cutter_id": int(row.cutter_id),
            "t0_frame": int(row.t0_frame),
            "frame": int(row.frame),
        }
        for r in range(3):
            for c in range(3):
                out[f"g{r}{c}"] = float(g3[r, c])
        rows.append(out)

    decoded = pd.DataFrame(rows)
    group_cols = ["recording_id", "cutter_id", "t0_frame"]
    agg = decoded.groupby(group_cols, as_index=False).agg(
        stage_frames=("frame", "count"),
        g00=("g00", "mean"),
        g01=("g01", "mean"),
        g02=("g02", "mean"),
        g10=("g10", "mean"),
        g11=("g11", "mean"),
        g12=("g12", "mean"),
        g20=("g20", "mean"),
        g21=("g21", "mean"),
        g22=("g22", "mean"),
    )
    return agg


def _cluster_features_table(events: pd.DataFrame, stage_features: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["recording_id", "cutter_id", "t0_frame"]
    merged = stage_features.merge(events, on=key_cols, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No event-level overlap between mirrored SFC codes and stage-feature table.")
    return merged


def _cluster_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_cols = ["g00", "g01", "g02", "g10", "g12", "g20", "g21", "g22"]
    x = df[feature_cols].to_numpy(dtype=float)
    return x, feature_cols


def _kmeans_labels(x: np.ndarray, k: int, *, random_state: int, n_init: int) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = model.fit_predict(x)
    return model, labels.astype(int)


def _mean_pairwise_ari(label_runs: list[np.ndarray]) -> float:
    if len(label_runs) < 2:
        return float("nan")
    vals: list[float] = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            vals.append(float(adjusted_rand_score(label_runs[i], label_runs[j])))
    return float(np.mean(vals)) if vals else float("nan")


def _evaluate_k_grid(
    x: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    random_state: int,
    n_init: int,
    stability_runs: int,
    min_cluster_fraction: float,
) -> tuple[pd.DataFrame, int]:
    if k_min < 2:
        raise ValueError("k_min must be >= 2")
    if k_max < k_min:
        raise ValueError("k_max must be >= k_min")

    rows: list[dict[str, float | int | bool]] = []
    for k in range(k_min, k_max + 1):
        base_model, base_labels = _kmeans_labels(x, k, random_state=random_state, n_init=n_init)
        counts = np.bincount(base_labels, minlength=k)
        min_frac = float(counts.min() / len(x))
        sil = float(silhouette_score(x, base_labels))

        label_runs = [base_labels]
        for run_idx in range(1, stability_runs):
            _, alt_labels = _kmeans_labels(
                x,
                k,
                random_state=random_state + 997 * run_idx,
                n_init=n_init,
            )
            label_runs.append(alt_labels)

        stability = _mean_pairwise_ari(label_runs)
        rows.append(
            {
                "k": int(k),
                "silhouette": sil,
                "stability_ari": stability,
                "min_cluster_fraction": min_frac,
                "passes_min_fraction": bool(min_frac >= min_cluster_fraction),
            }
        )

    grid = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    eligible = grid.loc[grid["passes_min_fraction"]].copy()
    if eligible.empty:
        eligible = grid.copy()

    selected = eligible.sort_values(
        ["silhouette", "stability_ari", "k"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return grid, int(selected["k"])


def _relabel_by_cluster_size(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    mapping = {int(old): int(idx + 1) for idx, old in enumerate(counts.index.tolist())}
    relabeled = np.array([mapping[int(v)] for v in labels], dtype=int)
    return relabeled, mapping


def _top_cells(row: pd.Series, *, top_n: int = 3) -> str:
    pairs = []
    for key, label in CELL_NAMES.items():
        if key == "g11":
            continue
        pairs.append((label, float(row[key])))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name}={value:.2f}" for name, value in pairs[:top_n])


def _cluster_heatmap(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            [row["g00"], row["g01"], row["g02"]],
            [row["g10"], row["g11"], row["g12"]],
            [row["g20"], row["g21"], row["g22"]],
        ],
        dtype=float,
    )


def _plot_archetype_heatmaps(summary_df: pd.DataFrame, out_path: Path, *, stage: str, selected_k: int) -> None:
    plt = _load_pyplot()
    n = len(summary_df)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols + 0.8, 4.0 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    im = None
    for idx, (_, row) in enumerate(summary_df.iterrows()):
        ax = axes_arr[idx // ncols, idx % ncols]
        g = _cluster_heatmap(row)
        im = ax.imshow(g, vmin=0.0, vmax=1.0, cmap="YlGnBu")
        ax.set_xticks([0, 1, 2], labels=["Left", "Same", "Right"])
        ax.set_yticks([0, 1, 2], labels=["Ahead", "Along", "Behind"])
        ax.set_title(
            f"A{int(row['archetype_id'])}: n={int(row['events'])}, risk={100.0 * float(row['risk_prevalence']):.1f}%",
            fontsize=10,
            weight="bold",
        )
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{g[r, c]:.2f}", ha="center", va="center", fontsize=9)

    for idx in range(n, nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")

    fig.suptitle(f"Step 15D archetypes from {stage} stage mirrored binary SFC context (K={selected_k})", fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.07, top=0.90, wspace=0.40, hspace=0.45)
    if im is not None:
        cbar_ax = fig.add_axes([0.90, 0.20, 0.022, 0.58])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel("Occupancy", rotation=270, labelpad=12)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_risk_prevalence(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [f"A{int(v)}" for v in summary_df["archetype_id"]]
    rates = 100.0 * summary_df["risk_prevalence"].to_numpy(dtype=float)
    ax.bar(labels, rates, color="#355c7d")
    ax.set_ylabel("Risk prevalence (%)")
    ax.set_xlabel("Archetype")
    ax.set_ylim(0.0, max(5.0, rates.max() * 1.15))
    ax.set_title("Execution-stage THW risk prevalence by context archetype", weight="bold")
    for idx, value in enumerate(rates):
        ax.text(idx, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 15D: derive interpretable context archetypes from mirrored binary SFC codes.")
    ap.add_argument(
        "--codes-csv",
        default=str(output_path("reports/Step 15A/sfc_binary_codes_long_hilbert_mirrored.csv")),
        help="Mirrored Step-15A long binary SFC code CSV.",
    )
    ap.add_argument(
        "--events-csv",
        default=str(output_path("reports/Step 09/cutin_stage_features_merged.csv")),
        help="Merged event table containing execution-stage risk fields.",
    )
    ap.add_argument("--out-dir", default=str(step_reports_dir(STEP_ID)))
    ap.add_argument(
        "--stage",
        choices=["intention", "decision", "execution"],
        default=thesis_str("step15d.stage", "decision", allowed={"intention", "decision", "execution"}),
        help="Stage used to construct pre-outcome context archetypes.",
    )
    ap.add_argument("--risk-thw", type=float, default=thesis_float("step15d.risk_thw", 0.7, min_value=0.0))
    ap.add_argument("--k-min", type=int, default=thesis_int("step15d.k_min", 3, min_value=2))
    ap.add_argument("--k-max", type=int, default=thesis_int("step15d.k_max", 8, min_value=2))
    ap.add_argument(
        "--min-cluster-fraction",
        type=float,
        default=thesis_float("step15d.min_cluster_fraction", 0.03, min_value=0.0),
    )
    ap.add_argument("--random-state", type=int, default=thesis_int("step15d.random_state", 7))
    ap.add_argument("--n-init", type=int, default=thesis_int("step15d.n_init", 50, min_value=1))
    ap.add_argument("--stability-runs", type=int, default=thesis_int("step15d.stability_runs", 5, min_value=2))
    ap.add_argument("--make-plot", action="store_true", default=thesis_bool("step15d.make_plot", True))
    ap.add_argument("--no-make-plot", dest="make_plot", action="store_false")
    args = ap.parse_args()

    codes_csv = Path(args.codes_csv)
    events_csv = Path(args.events_csv)
    if not codes_csv.exists():
        raise FileNotFoundError(f"Missing codes CSV: {codes_csv}")
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events CSV: {events_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = _parse_stage_events(events_csv, risk_thw=float(args.risk_thw))
    stage_features = _decode_binary_stage_features(codes_csv, stage=str(args.stage))
    event_df = _cluster_features_table(events, stage_features)

    x, feature_cols = _cluster_feature_matrix(event_df)
    k_grid_df, selected_k = _evaluate_k_grid(
        x,
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        random_state=int(args.random_state),
        n_init=int(args.n_init),
        stability_runs=int(args.stability_runs),
        min_cluster_fraction=float(args.min_cluster_fraction),
    )

    model, raw_labels = _kmeans_labels(
        x,
        selected_k,
        random_state=int(args.random_state),
        n_init=int(args.n_init),
    )
    labels, label_map = _relabel_by_cluster_size(raw_labels)
    centers = model.cluster_centers_

    assignments = event_df.copy()
    assignments["archetype_id"] = labels
    distances = np.linalg.norm(x - centers[raw_labels], axis=1)
    assignments["distance_to_centroid"] = distances

    rep_rows: list[dict[str, object]] = []
    for old_label, new_label in label_map.items():
        mask = raw_labels == old_label
        idx_cluster = np.flatnonzero(mask)
        centroid = centers[old_label]
        cluster_x = x[mask]
        d = np.linalg.norm(cluster_x - centroid, axis=1)
        best_local = int(idx_cluster[int(np.argmin(d))])
        best_row = assignments.iloc[best_local]
        rep_rows.append(
            {
                "archetype_id": int(new_label),
                "recording_id": best_row["recording_id"],
                "cutter_id": int(best_row["cutter_id"]),
                "follower_id": int(best_row["follower_id"]),
                "t0_frame": int(best_row["t0_frame"]),
                "execution_thw_min": float(best_row["execution_thw_min"]),
                "execution_ttc_min": float(best_row["execution_ttc_min"])
                if pd.notna(best_row["execution_ttc_min"])
                else float("nan"),
                "risk_thw": bool(best_row["risk_thw"]),
                "distance_to_centroid": float(best_row["distance_to_centroid"]),
            }
        )
    reps_df = pd.DataFrame(rep_rows).sort_values("archetype_id").reset_index(drop=True)

    occupancy_cols = [f"g{r}{c}" for r in range(3) for c in range(3)]
    summary = (
        assignments.groupby("archetype_id", as_index=False)
        .agg(
            events=("archetype_id", "size"),
            stage_frames_mean=("stage_frames", "mean"),
            risk_events=("risk_thw", "sum"),
            risk_prevalence=("risk_thw", "mean"),
            execution_thw_median=("execution_thw_min", "median"),
            execution_dhw_median=("execution_dhw_min", "median"),
            execution_ttc_median_finite=(
                "execution_ttc_min",
                lambda s: float(pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna().median())
                if pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna().size
                else float("nan"),
            ),
            **{col: (col, "mean") for col in occupancy_cols},
        )
        .sort_values("archetype_id")
        .reset_index(drop=True)
    )
    overall_risk = float(assignments["risk_thw"].mean())
    summary["event_share"] = summary["events"] / float(len(assignments))
    summary["risk_ratio_vs_overall"] = summary["risk_prevalence"] / overall_risk if overall_risk > 0 else np.nan
    summary["dominant_cells"] = summary.apply(_top_cells, axis=1)
    summary = summary.merge(reps_df, on="archetype_id", how="left", validate="one_to_one")

    selected_metrics = k_grid_df.loc[k_grid_df["k"] == selected_k].iloc[0]
    summary["selected_stage"] = str(args.stage)
    summary["selected_k"] = int(selected_k)
    summary["selected_silhouette"] = float(selected_metrics["silhouette"])
    summary["selected_stability_ari"] = float(selected_metrics["stability_ari"])

    assignment_cols = [
        "recording_id",
        "cutter_id",
        "follower_id",
        "t0_frame",
        "from_lane",
        "to_lane",
        "stage_frames",
        "execution_thw_min",
        "execution_dhw_min",
        "execution_ttc_min",
        "risk_thw",
        "archetype_id",
        "distance_to_centroid",
        *occupancy_cols,
    ]
    assignments_out = assignments[assignment_cols].sort_values(
        ["archetype_id", "recording_id", "cutter_id", "t0_frame"],
        kind="mergesort",
    )

    assignments_path = out_dir / "archetype_assignments.csv"
    summary_path = out_dir / "archetype_summary.csv"
    k_grid_path = out_dir / "archetype_k_selection.csv"
    reps_path = out_dir / "archetype_representatives.csv"
    assignments_out.to_csv(assignments_path, index=False)
    summary.to_csv(summary_path, index=False)
    k_grid_df.to_csv(k_grid_path, index=False)
    reps_df.to_csv(reps_path, index=False)

    canonical_assignments = mirror_file_to_step(assignments_path, STEP_ID)
    canonical_summary = mirror_file_to_step(summary_path, STEP_ID)
    canonical_k_grid = mirror_file_to_step(k_grid_path, STEP_ID)
    canonical_reps = mirror_file_to_step(reps_path, STEP_ID)

    heatmap_path = out_dir / "archetype_heatmaps.png"
    prevalence_path = out_dir / "archetype_risk_prevalence.png"
    canonical_heatmap = None
    canonical_prevalence = None
    if bool(args.make_plot):
        _plot_archetype_heatmaps(summary, heatmap_path, stage=str(args.stage), selected_k=int(selected_k))
        _plot_risk_prevalence(summary, prevalence_path)
        canonical_heatmap = mirror_file_to_step(heatmap_path, STEP_ID)
        canonical_prevalence = mirror_file_to_step(prevalence_path, STEP_ID)

    report_lines = [
        "# Step 15D SFC Context Archetypes",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input mirrored codes CSV: `{codes_csv}`",
        f"- Input events CSV: `{events_csv}`",
        f"- Stage clustered: `{args.stage}`",
        f"- Events clustered: `{len(assignments_out)}`",
        f"- Risk THW threshold: `{float(args.risk_thw):.3f}`",
        f"- K search range: `{int(args.k_min)}..{int(args.k_max)}`",
        f"- Selected K: `{int(selected_k)}`",
        f"- Selected silhouette: `{float(selected_metrics['silhouette']):.4f}`",
        f"- Selected stability ARI: `{float(selected_metrics['stability_ari']):.4f}`",
        f"- Minimum cluster fraction target: `{100.0 * float(args.min_cluster_fraction):.1f}%`",
        f"- Overall risky prevalence: `{100.0 * overall_risk:.2f}%`",
        f"- Assignment CSV: `{canonical_assignments}`",
        f"- Summary CSV: `{canonical_summary}`",
        f"- K-selection CSV: `{canonical_k_grid}`",
        f"- Representative-event CSV: `{canonical_reps}`",
    ]
    if canonical_heatmap is not None and canonical_prevalence is not None:
        report_lines.extend(
            [
                f"- Heatmap figure: `{canonical_heatmap}`",
                f"- Risk-prevalence figure: `{canonical_prevalence}`",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Archetype Summary",
            "",
            "| Archetype | Events | Share | Risk % | Risk ratio | Median execution THW | Dominant cells | Representative event |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for _, row in summary.iterrows():
        rep = f"{row['recording_id']}/{int(row['cutter_id'])}/{int(row['t0_frame'])}"
        report_lines.append(
            "| A{aid} | {events} | {share:.2%} | {risk:.2%} | {ratio:.2f} | {thw:.3f} | {cells} | {rep} |".format(
                aid=int(row["archetype_id"]),
                events=int(row["events"]),
                share=float(row["event_share"]),
                risk=float(row["risk_prevalence"]),
                ratio=float(row["risk_ratio_vs_overall"]),
                thw=float(row["execution_thw_median"]),
                cells=str(row["dominant_cells"]),
                rep=rep,
            )
        )

    report_path = write_step_markdown(STEP_ID, "archetype_report.md", report_lines)

    print("== Step 15D: SFC context archetypes ==")
    print(f"Events clustered: {len(assignments_out)}")
    print(f"Stage: {args.stage}")
    print(f"Selected K: {selected_k}")
    print(f"Overall risky prevalence: {100.0 * overall_risk:.2f}%")
    print("Saved:", canonical_assignments)
    print("Saved:", canonical_summary)
    print("Saved:", canonical_k_grid)
    print("Saved:", canonical_reps)
    if canonical_heatmap is not None:
        print("Saved:", canonical_heatmap)
    if canonical_prevalence is not None:
        print("Saved:", canonical_prevalence)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
