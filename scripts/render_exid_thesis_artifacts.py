"""Generate thesis-facing exiD figures and copy them into latex/."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import PercentFormatter
from PIL import Image

from cutin_risk.paths import exid_dataset_root_path, output_path, project_root


OVERVIEW_SOURCE = output_path("figures/Step 22/exid_exploratory_overview.png")
EXAMPLES_SOURCE = output_path("reports/Step 22/exid_example_events.csv")
QUALITATIVE_OUTPUT = output_path("figures/Step 22/exid_example_panels.png")
MERGED_SOURCE = output_path("reports/Step 22/exid_stage_features_merged.csv")
TTC_SUMMARY_OUTPUT = output_path("figures/Step 22/exid_ttc_summary.png")


def _normalize_recording_id(value: str | int | float) -> str:
    token = str(value).strip()
    return f"{int(token):02d}" if token.isdigit() else token


def _get_rotated_bbox(x_center, y_center, length, width, heading):
    centroids = np.column_stack([x_center, y_center])
    half_length = length / 2.0
    half_width = width / 2.0
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)

    lc = half_length * cos_heading
    ls = half_length * sin_heading
    wc = half_width * cos_heading
    ws = half_width * sin_heading

    corners = np.empty((centroids.shape[0], 4, 2))
    corners[:, 0, 0] = lc - ws
    corners[:, 0, 1] = ls + wc
    corners[:, 1, 0] = -lc - ws
    corners[:, 1, 1] = -ls + wc
    corners[:, 2, 0] = -lc + ws
    corners[:, 2, 1] = -ls - wc
    corners[:, 3, 0] = lc + ws
    corners[:, 3, 1] = ls - wc
    return corners + np.expand_dims(centroids, axis=1)


def _load_frame_tracks(dataset_root: Path, recording_id: str, frame: int) -> pd.DataFrame:
    usecols = ["trackId", "frame", "xCenter", "yCenter", "width", "length", "heading"]
    df = pd.read_csv(dataset_root / f"{recording_id}_tracks.csv", usecols=usecols)
    frame_df = df.loc[df["frame"].astype(int) == int(frame)].copy()
    if frame_df.empty:
        raise ValueError(f"No exiD rows for recording {recording_id} at frame {frame}")
    recording_meta = pd.read_csv(dataset_root / f"{recording_id}_recordingMeta.csv").iloc[0]
    ortho_px_to_meter = float(recording_meta["orthoPxToMeter"])

    frame_df["x_px"] = frame_df["xCenter"].astype(float) / ortho_px_to_meter
    frame_df["y_px"] = -frame_df["yCenter"].astype(float) / ortho_px_to_meter
    frame_df["length_px"] = frame_df["length"].astype(float) / ortho_px_to_meter
    frame_df["width_px"] = frame_df["width"].astype(float) / ortho_px_to_meter
    heading_vis = np.deg2rad((-frame_df["heading"].astype(float).to_numpy()) % 360.0)
    frame_df["bbox_poly"] = list(
        _get_rotated_bbox(
            frame_df["x_px"].to_numpy(dtype=float),
            frame_df["y_px"].to_numpy(dtype=float),
            frame_df["length_px"].to_numpy(dtype=float),
            frame_df["width_px"].to_numpy(dtype=float),
            heading_vis,
        )
    )
    return frame_df


def _track_center(frame_df: pd.DataFrame, track_id: int) -> np.ndarray:
    row = frame_df.loc[frame_df["trackId"].astype(int) == int(track_id)]
    if row.empty:
        raise ValueError(f"Track {track_id} missing in selected frame")
    rec = row.iloc[0]
    return np.array([float(rec["x_px"]), float(rec["y_px"])], dtype=float)


def _crop_limits(image_size: tuple[int, int], center: np.ndarray, *, width: float = 1800.0, height: float = 1200.0):
    image_w, image_h = image_size
    half_w = width / 2.0
    half_h = height / 2.0

    x_min = max(0.0, float(center[0] - half_w))
    x_max = min(float(image_w), float(center[0] + half_w))
    y_min = max(0.0, float(center[1] - half_h))
    y_max = min(float(image_h), float(center[1] + half_h))

    if x_max - x_min < width:
        if x_min <= 0.0:
            x_max = min(float(image_w), width)
        else:
            x_min = max(0.0, float(image_w) - width)
    if y_max - y_min < height:
        if y_min <= 0.0:
            y_max = min(float(image_h), height)
        else:
            y_min = max(0.0, float(image_h) - height)

    return x_min, x_max, y_max, y_min


def _render_example_panel(ax, *, dataset_root: Path, example: pd.Series) -> None:
    recording_id = _normalize_recording_id(example["recording_id"])
    frame = int(example["t0_frame"])
    cutter_id = int(example["cutter_id"])
    follower_id = int(example["follower_id"])

    background_path = dataset_root / f"{recording_id}_background.png"
    background = np.asarray(Image.open(background_path).convert("RGB"))
    frame_df = _load_frame_tracks(dataset_root, recording_id, frame)

    cutter_center = _track_center(frame_df, cutter_id)
    follower_center = _track_center(frame_df, follower_id)
    focus_center = (cutter_center + follower_center) / 2.0

    ax.imshow(background, alpha=0.96)
    ax.set_xlim(*_crop_limits((background.shape[1], background.shape[0]), focus_center)[:2])
    _, _, ylim_top, ylim_bottom = _crop_limits((background.shape[1], background.shape[0]), focus_center)
    ax.set_ylim(ylim_top, ylim_bottom)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#08111b")

    for _, row in frame_df.iterrows():
        track_id = int(row["trackId"])
        polygon = np.asarray(row["bbox_poly"], dtype=float)
        if track_id == cutter_id:
            face = "#f59e0b"
            edge = "#fff7ed"
            alpha = 0.95
            lw = 1.6
            zorder = 6
        elif track_id == follower_id:
            face = "#0ea5e9"
            edge = "#ecfeff"
            alpha = 0.95
            lw = 1.6
            zorder = 6
        else:
            face = "#ef4444"
            edge = "#f8fafc"
            alpha = 0.40
            lw = 0.4
            zorder = 4

        ax.add_patch(
            patches.Polygon(
                polygon,
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
            )
        )

    label_box = dict(boxstyle="round,pad=0.25", fc="#08111b", ec="#e2e8f0", lw=0.8, alpha=0.92)
    ax.text(
        float(cutter_center[0]),
        float(cutter_center[1] - 90.0),
        f"Cutter {cutter_id}",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#fff7ed",
        bbox=label_box,
        zorder=10,
    )
    ax.text(
        float(follower_center[0]),
        float(follower_center[1] + 90.0),
        f"Follower {follower_id}",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#ecfeff",
        bbox=label_box,
        zorder=10,
    )

    ttc_value = pd.to_numeric(pd.Series([example["execution_ttc_min"]]), errors="coerce").iloc[0]
    ttc_text = "n/a" if not math.isfinite(float(ttc_value)) else f"{float(ttc_value):.2f}s"
    ax.set_title(
        (
            f"Recording {recording_id} | Frame {frame}\n"
            f"THW={float(example['execution_thw_min']):.2f}s | "
            f"DHW={float(example['execution_dhw_min']):.2f}m | TTC={ttc_text}"
        ),
        fontsize=10.5,
        color="#e2e8f0",
        pad=10.0,
    )
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.0)


def generate_qualitative_figure(*, dataset_root: Path, examples_csv: Path, target_path: Path) -> Path:
    examples = pd.read_csv(examples_csv)
    if examples.empty:
        raise ValueError(f"No examples found in {examples_csv}")
    selected = examples.head(2).reset_index(drop=True)

    fig, axes = plt.subplots(1, len(selected), figsize=(12.8, 5.8), facecolor="#020617")
    if len(selected) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, selected.iterrows()):
        _render_example_panel(ax, dataset_root=dataset_root, example=row)
    fig.suptitle(
        "Exploratory exiD qualitative examples",
        fontsize=13,
        color="#f8fafc",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return target_path


def generate_ttc_summary_figure(*, merged_csv: Path, target_path: Path) -> Path:
    merged = pd.read_csv(merged_csv).copy()
    if merged.empty:
        raise ValueError(f"No exiD stage features found in {merged_csv}")

    ttc = pd.to_numeric(merged["execution_ttc_min"], errors="coerce")
    merged["ttc_finite"] = np.isfinite(ttc)
    merged["execution_ttc_finite"] = ttc.replace([np.inf, -np.inf], np.nan)

    summary = (
        merged.groupby("recording_id", as_index=False)
        .agg(
            exported_events=("cutter_id", "count"),
            execution_ttc_finite_share=("ttc_finite", "mean"),
            execution_ttc_median=("execution_ttc_finite", "median"),
        )
        .sort_values("recording_id")
    )
    if summary.empty:
        raise ValueError(f"No exiD TTC summary rows could be derived from {merged_csv}")

    overall_finite_share = float(merged["ttc_finite"].mean())
    overall_median = float(merged["execution_ttc_finite"].median())
    x = np.arange(len(summary), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor="white")

    axes[0].bar(x, summary["execution_ttc_finite_share"], color="#0ea5e9", width=0.72)
    axes[0].axhline(overall_finite_share, color="#0f172a", linestyle="--", linewidth=1.2)
    axes[0].set_title("Finite execution-stage TTC share")
    axes[0].set_xlabel("Recording")
    axes[0].set_ylabel("Share of exported events")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[1].bar(x, summary["execution_ttc_median"], color="#f59e0b", width=0.72)
    axes[1].axhline(overall_median, color="#0f172a", linestyle="--", linewidth=1.2)
    axes[1].set_title("Median execution-stage TTC")
    axes[1].set_xlabel("Recording")
    axes[1].set_ylabel("TTC [s] (finite only)")
    axes[1].set_ylim(0.0, float(summary["execution_ttc_median"].max()) * 1.12)

    for axis in axes:
        axis.set_xticks(x, summary["recording_id"].astype(str))
        axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", rotation=45)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)

    axes[0].text(
        0.99,
        0.96,
        f"Overall: {overall_finite_share:.1%}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color="#0f172a",
    )
    axes[1].text(
        0.99,
        0.96,
        f"Overall median: {overall_median:.2f}s",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color="#0f172a",
    )

    fig.tight_layout()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-facing exiD artifacts.")
    parser.add_argument("--dataset-root", type=Path, default=exid_dataset_root_path())
    parser.add_argument("--overview-source", type=Path, default=OVERVIEW_SOURCE)
    parser.add_argument("--merged-csv", type=Path, default=MERGED_SOURCE)
    parser.add_argument("--ttc-summary-output", type=Path, default=TTC_SUMMARY_OUTPUT)
    parser.add_argument("--latex-dir", type=Path, default=project_root() / "latex")
    args = parser.parse_args()

    latex_dir = args.latex_dir.resolve()
    latex_dir.mkdir(parents=True, exist_ok=True)

    overview_target = latex_dir / "exid_exploratory_overview.png"
    ttc_summary_target = latex_dir / "exid_ttc_summary.png"

    if not args.overview_source.exists():
        raise FileNotFoundError(f"Missing overview source: {args.overview_source}")
    if not args.merged_csv.exists():
        raise FileNotFoundError(f"Missing merged exiD features: {args.merged_csv}")

    shutil.copy2(args.overview_source, overview_target)
    args.ttc_summary_output.parent.mkdir(parents=True, exist_ok=True)
    generated_ttc_summary = generate_ttc_summary_figure(
        merged_csv=args.merged_csv.resolve(),
        target_path=args.ttc_summary_output.resolve(),
    )
    shutil.copy2(generated_ttc_summary, ttc_summary_target)

    print("Saved:", overview_target)
    print("Saved:", generated_ttc_summary)
    print("Saved:", ttc_summary_target)


if __name__ == "__main__":
    main()
