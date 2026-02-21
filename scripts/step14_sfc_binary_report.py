"""Step 14 report: decode and aggregate binary SFC codes into occupancy heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cutin_risk.encoding.sfc_binary import decode_grid_4x4_bits
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


def plot_grid(g: np.ndarray, title: str, out: Path) -> None:
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
    args = ap.parse_args()

    codes_csv = Path(args.codes_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(codes_csv)
    if df.empty:
        raise ValueError("codes csv is empty")

    order = str(df["sfc_order"].iloc[0])

    for stage in ["intention", "decision", "execution"]:
        d_stage = df[df["stage"] == stage]
        if d_stage.empty:
            continue

        d_risk = d_stage[d_stage["risk_thw"] == True]
        d_safe = d_stage[d_stage["risk_thw"] == False]

        g_risk = mean_grid(d_risk, order=order)
        g_safe = mean_grid(d_safe, order=order)
        g_diff = g_risk - g_safe

        plot_grid(g_safe, f"{stage}: non-risk mean occupancy", out_dir / f"{stage}_nonrisk.png")
        plot_grid(g_risk, f"{stage}: risk mean occupancy", out_dir / f"{stage}_risk.png")
        plot_grid(g_diff, f"{stage}: (risk - nonrisk) difference", out_dir / f"{stage}_diff.png")

        # also save raw numbers for the thesis
        np.savetxt(out_dir / f"{stage}_nonrisk.csv", g_safe, delimiter=",", fmt="%.6f")
        np.savetxt(out_dir / f"{stage}_risk.csv", g_risk, delimiter=",", fmt="%.6f")
        np.savetxt(out_dir / f"{stage}_diff.csv", g_diff, delimiter=",", fmt="%.6f")

    print("Saved report to:", out_dir)


if __name__ == "__main__":
    main()
