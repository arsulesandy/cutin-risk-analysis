"""Step 15B: build weighted SFC stage features using distance or TTC scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.paths import dataset_root_path, output_path
from cutin_risk.reconstruction.lanes import parse_lane_markings, infer_lane_index

from cutin_risk.encoding.sfc_binary import build_lane_snapshots
from cutin_risk.encoding.sfc_weighted import (
    WeightedSFCOptions,
    grid3x3_weighted,
    sfc_vector_4x4_from_3x3,
)


def normalize_recording_id(v: object) -> str:
    s = str(v).strip()
    return f"{int(s):02d}" if s.isdigit() else s


def add_sign_s(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dir_to_sign = {1: -1, 2: 1}
    df["sign"] = df["drivingDirection"].map(dir_to_sign)

    if df["sign"].isna().any():
        vx = pd.to_numeric(df["xVelocity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sx = np.sign(vx)
        sx[sx == 0] = 1
        df["sign"] = df["sign"].fillna(pd.Series(sx, index=df.index))

    df["sign"] = df["sign"].fillna(1).astype(int)
    df["s"] = df["sign"].astype(float) * df["x"].astype(float)
    return df


def stage_ranges(t0: int, *, pre4: int, pre2: int, post2: int) -> list[tuple[str, int, int]]:
    intention = (max(1, t0 - pre4), max(1, t0 - pre2 - 1))
    decision = (max(1, t0 - pre2), max(1, t0 - 1))
    execution = (t0, t0 + post2)

    out = []
    if intention[0] <= intention[1]:
        out.append(("intention", intention[0], intention[1]))
    if decision[0] <= decision[1]:
        out.append(("decision", decision[0], decision[1]))
    if execution[0] <= execution[1]:
        out.append(("execution", execution[0], execution[1]))
    return out


def mirror_3x3(g3: np.ndarray) -> np.ndarray:
    g = g3.copy()
    g[:, [0, 2]] = g[:, [2, 0]]
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 15B: Weighted SFC stage features (distance or TTC).")
    ap.add_argument("--events-csv", default=str(output_path("reports/step9_batch/cutin_stage_features_merged.csv")))
    ap.add_argument("--dataset-root", default=str(dataset_root_path()))
    ap.add_argument("--out-dir", default=str(output_path("reports/step15b_sfc_weighted")))

    ap.add_argument("--mode", choices=["distance", "ttc"], default="distance")
    ap.add_argument("--order", choices=["hilbert", "morton"], default="hilbert")
    ap.add_argument("--risk-thw", type=float, default=0.70)

    ap.add_argument("--pre4-seconds", type=float, default=4.0)
    ap.add_argument("--pre2-seconds", type=float, default=2.0)
    ap.add_argument("--post2-seconds", type=float, default=2.0)

    ap.add_argument("--alongside-thresh", type=float, default=5.0)
    ap.add_argument("--range-ahead", type=float, default=150.0)
    ap.add_argument("--range-behind", type=float, default=150.0)
    ap.add_argument("--ttc-max", type=float, default=10.0)

    ap.add_argument("--from-col", default="from_lane")
    ap.add_argument("--to-col", default="to_lane")
    ap.add_argument("--canonical-direction", choices=["positive", "negative"], default="positive")

    args = ap.parse_args()

    events_csv = Path(args.events_csv)
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events CSV: {events_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(events_csv).copy()

    need = {"recording_id", "cutter_id", "t0_frame", "execution_thw_min", args.from_col, args.to_col}
    missing = need - set(ev.columns)
    if missing:
        raise ValueError(f"events CSV missing columns: {sorted(missing)}")

    ev["recording_id"] = ev["recording_id"].apply(normalize_recording_id)
    ev["cutter_id"] = pd.to_numeric(ev["cutter_id"], errors="coerce").astype("Int64")
    ev["t0_frame"] = pd.to_numeric(ev["t0_frame"], errors="coerce").astype("Int64")
    ev["execution_thw_min"] = pd.to_numeric(ev["execution_thw_min"], errors="coerce")
    ev[args.from_col] = pd.to_numeric(ev[args.from_col], errors="coerce")
    ev[args.to_col] = pd.to_numeric(ev[args.to_col], errors="coerce")
    ev = ev.dropna(subset=["cutter_id", "t0_frame", "execution_thw_min", args.from_col, args.to_col]).copy()

    ev["risk_thw"] = ev["execution_thw_min"] < float(args.risk_thw)
    ev["lane_delta"] = ev[args.to_col] - ev[args.from_col]
    if args.canonical_direction == "positive":
        ev["mirror"] = ev["lane_delta"] < 0
    else:
        ev["mirror"] = ev["lane_delta"] > 0

    opt = WeightedSFCOptions(
        mode=args.mode,
        order=args.order,
        alongside_s_thresh=float(args.alongside_thresh),
        max_range_ahead=float(args.range_ahead),
        max_range_behind=float(args.range_behind),
        ttc_max=float(args.ttc_max),
        include_center=True,
    )

    rows_out: list[dict[str, object]] = []

    for rid in sorted(ev["recording_id"].unique().tolist()):
        ev_r = ev[ev["recording_id"] == rid].copy()
        if ev_r.empty:
            continue

        rec = load_highd_recording(Path(args.dataset_root), rid)
        df = build_tracking_table(rec)

        markings = parse_lane_markings(rec.recording_meta)
        df = df.join(infer_lane_index(df, markings))  # laneIndex_xy
        df = add_sign_s(df)

        indexed = df.set_index(["id", "frame"], drop=False).sort_index()

        frame_rate = float(rec.recording_meta.loc[0, "frameRate"])
        pre4 = int(round(float(args.pre4_seconds) * frame_rate))
        pre2 = int(round(float(args.pre2_seconds) * frame_rate))
        post2 = int(round(float(args.post2_seconds) * frame_rate))

        # frames needed for snapshots
        frames_needed: set[int] = set()
        for _, e in ev_r.iterrows():
            t0 = int(e["t0_frame"])
            for _, a, b in stage_ranges(t0, pre4=pre4, pre2=pre2, post2=post2):
                frames_needed.update(range(a, b + 1))

        snapshots = build_lane_snapshots(
            df,
            frames_needed,
            lane_col="laneIndex_xy",
            sign_col="sign",
            s_col="s",
        )

        for _, e in ev_r.iterrows():
            cutter_id = int(e["cutter_id"])
            t0 = int(e["t0_frame"])
            mirror = bool(e["mirror"])
            risk = bool(e["risk_thw"])

            out_row: dict[str, object] = {
                "recording_id": rid,
                "cutter_id": cutter_id,
                "t0_frame": t0,
                "risk_thw": risk,
                "lane_delta": float(e["lane_delta"]),
                "mirrored": mirror,
                "mode": args.mode,
                "order": args.order,
            }

            for stage, a, b in stage_ranges(t0, pre4=pre4, pre2=pre2, post2=post2):
                vecs = []
                for f in range(a, b + 1):
                    try:
                        cr = indexed.loc[(cutter_id, f)]
                    except KeyError:
                        continue

                    cutter_lane = int(cr["laneIndex_xy"])
                    cutter_sign = int(cr["sign"])
                    cutter_s = float(cr["s"])
                    cutter_v_s = float(cr["sign"]) * float(cr["xVelocity"])

                    g3 = grid3x3_weighted(
                        snapshots=snapshots,
                        indexed=indexed,
                        frame=int(f),
                        cutter_id=cutter_id,
                        cutter_lane=cutter_lane,
                        cutter_sign=cutter_sign,
                        cutter_s=cutter_s,
                        cutter_v_s=cutter_v_s,
                        options=opt,
                    )

                    if mirror:
                        g3 = mirror_3x3(g3)

                    vec = sfc_vector_4x4_from_3x3(g3, order=opt.order)
                    vecs.append(vec)

                if vecs:
                    M = np.vstack(vecs)
                    mean_vec = M.mean(axis=0)
                    out_row[f"{stage}_frames"] = int(len(vecs))
                    for k in range(16):
                        out_row[f"{stage}_sfc_v{k:02d}_mean"] = float(mean_vec[k])
                else:
                    out_row[f"{stage}_frames"] = 0
                    for k in range(16):
                        out_row[f"{stage}_sfc_v{k:02d}_mean"] = 0.0

            rows_out.append(out_row)

        print(f"[Step15B] recording {rid}: events={len(ev_r)}")

    out = pd.DataFrame(rows_out)
    out_path = out_dir / f"sfc_weighted_stage_features_{args.mode}_{args.order}.csv"
    out.to_csv(out_path, index=False)

    print("\n== Step 15B: Weighted SFC stage features ==")
    print("Saved:", out_path)
    print("Rows:", len(out))
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
