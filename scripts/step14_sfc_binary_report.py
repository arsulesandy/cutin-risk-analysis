"""Step 14 report: decode and aggregate binary SFC codes into occupancy heatmaps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.encoding.sfc_binary import decode_grid_4x4_bits
from cutin_risk.io.step_reports import mirror_file_to_step, write_step_markdown
from cutin_risk.paths import output_path, step14_codes_csv_path


def mean_grid(df: pd.DataFrame, *, order: str) -> np.ndarray:
    acc = np.zeros((3, 3), dtype=float)
    n = 0
    for code in df["code"].to_numpy(dtype=int):
        g4 = decode_grid_4x4_bits(int(code), order=order)  # 4x4
        g3 = g4[:3, :3]  # 3x3
        acc += g3
        n += 1
    return acc / max(n, 1)


def _load_pyplot(out_dir: Path):
    # Keep plotting reproducible in headless/sandboxed environments.
    cache_home = out_dir / ".cache"
    mpl_config = out_dir / ".mplconfig"
    cache_home.mkdir(parents=True, exist_ok=True)
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_home))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_grid(plt, g: np.ndarray, title: str, out: Path) -> None:
    plt.figure()
    plt.imshow(g, vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xticks([0, 1, 2], ["left", "same", "right"])
    plt.yticks([0, 1, 2], ["preceding", "alongside", "following"])
    plt.colorbar(label="occupancy probability")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes-csv", default=str(step14_codes_csv_path()))
    ap.add_argument("--out-dir", default=str(output_path("reports/step14_sfc_binary_report")))
    ap.add_argument(
        "--make-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate PNG heatmaps in addition to CSV grids.",
    )
    args = ap.parse_args()

    codes_csv = Path(args.codes_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(codes_csv)
    if df.empty:
        raise ValueError("codes csv is empty")

    order = str(df["sfc_order"].iloc[0])

    plt = _load_pyplot(out_dir) if args.make_plot else None

    for stage in ["intention", "decision", "execution"]:
        d_stage = df[df["stage"] == stage]
        if d_stage.empty:
            continue

        risk_mask = d_stage["risk_thw"].astype(bool)
        d_risk = d_stage[risk_mask]
        d_safe = d_stage[~risk_mask]

        g_risk = mean_grid(d_risk, order=order)
        g_safe = mean_grid(d_safe, order=order)
        g_diff = g_risk - g_safe

        if plt is not None:
            plot_grid(plt, g_safe, f"{stage}: non-risk mean occupancy", out_dir / f"{stage}_nonrisk.png")
            plot_grid(plt, g_risk, f"{stage}: risk mean occupancy", out_dir / f"{stage}_risk.png")
            plot_grid(plt, g_diff, f"{stage}: (risk - nonrisk) difference", out_dir / f"{stage}_diff.png")

        # also save raw numbers for the thesis
        np.savetxt(out_dir / f"{stage}_nonrisk.csv", g_safe, delimiter=",", fmt="%.6f")
        np.savetxt(out_dir / f"{stage}_risk.csv", g_risk, delimiter=",", fmt="%.6f")
        np.savetxt(out_dir / f"{stage}_diff.csv", g_diff, delimiter=",", fmt="%.6f")

    mirrored: list[Path] = []
    for p in sorted(out_dir.glob("*.csv")):
        mirrored.append(mirror_file_to_step(p, 14))
    for p in sorted(out_dir.glob("*.png")):
        mirrored.append(mirror_file_to_step(p, 14, kind="figures"))

    details_md = write_step_markdown(
        14,
        "sfc_binary_report_details.md",
        [
            "# Step 14 Binary SFC Report",
            "",
            f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Input codes CSV: `{codes_csv}`",
            f"- Local report directory: `{out_dir}`",
            f"- Plot generation enabled: `{bool(args.make_plot)}`",
            f"- Artifacts mirrored: `{len(mirrored)}`",
        ],
    )

    print("Saved report to:", out_dir)
    print("Saved summary markdown:", details_md)


if __name__ == "__main__":
    main()
