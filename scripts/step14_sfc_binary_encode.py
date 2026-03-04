"""Step 14: encode stage windows as binary SFC occupancy signatures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cutin_risk.datasets.highd.reader import load_highd_recording
from cutin_risk.datasets.highd.transforms import build_tracking_table
from cutin_risk.io.progress import iter_with_progress
from cutin_risk.io.step_reports import mirror_file_to_step, step_reports_dir, write_step_markdown
from cutin_risk.paths import dataset_root_path, output_path
from cutin_risk.reconstruction.lanes import parse_lane_markings, infer_lane_index

from cutin_risk.encoding.sfc_binary import (
    BinarySFCOptions,
    build_lane_snapshots,
    encode_frame_binary_sfc,
    SFCOrder,
)
from cutin_risk.thesis_config import thesis_float, thesis_str


def normalize_recording_id(v: object) -> str:
    s = str(v).strip()
    return f"{int(s):02d}" if s.isdigit() else s


def add_sign_and_s(df: pd.DataFrame) -> pd.DataFrame:
    """
    sign: +1 when moving in +x direction, -1 otherwise.
    s: signed longitudinal coordinate = sign * x, so "ahead" always means increasing s.
    """
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


def stage_ranges(t0: int, fr: int, *, pre4: int, pre2: int, post2: int) -> list[tuple[str, int, int]]:
    """
    Returns inclusive frame ranges for:
      intention: [-4s, -2s)
      decision:  [-2s, 0s)
      execution: [0s, +2s]
    """
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 14: Binary SFC encoding for cut-in events.")
    ap.add_argument("--merged-csv", default=str(output_path("reports/Step 09/cutin_stage_features_merged.csv")))
    ap.add_argument("--dataset-root", default=str(dataset_root_path()))
    ap.add_argument("--out-dir", default=str(step_reports_dir(14)))
    ap.add_argument("--risk-thw", type=float, default=thesis_float("step14.risk_thw", 0.70, min_value=0.0))

    ap.add_argument(
        "--sfc-order",
        choices=["hilbert", "morton"],
        default=thesis_str("step14.sfc_order", "hilbert", allowed={"hilbert", "morton"}),
    )
    ap.add_argument(
        "--alongside-thresh",
        type=float,
        default=thesis_float("step14.alongside_thresh", 5.0, min_value=0.0),
    )
    ap.add_argument("--range-ahead", type=float, default=thesis_float("step14.range_ahead", 150.0, min_value=0.0))
    ap.add_argument("--range-behind", type=float, default=thesis_float("step14.range_behind", 150.0, min_value=0.0))

    ap.add_argument("--pre4-seconds", type=float, default=thesis_float("step14.pre4_seconds", 4.0, min_value=0.0))
    ap.add_argument("--pre2-seconds", type=float, default=thesis_float("step14.pre2_seconds", 2.0, min_value=0.0))
    ap.add_argument("--post2-seconds", type=float, default=thesis_float("step14.post2_seconds", 2.0, min_value=0.0))

    args = ap.parse_args()

    merged_csv = Path(args.merged_csv)
    if not merged_csv.exists():
        raise FileNotFoundError(f"Missing input: {merged_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(merged_csv).copy()
    must = {"recording_id", "cutter_id", "t0_frame", "execution_thw_min"}
    missing = must - set(events.columns)
    if missing:
        raise ValueError(f"merged CSV missing required columns: {sorted(missing)}")

    events["recording_id"] = events["recording_id"].apply(normalize_recording_id)
    events["cutter_id"] = pd.to_numeric(events["cutter_id"], errors="coerce").astype("Int64")
    events["t0_frame"] = pd.to_numeric(events["t0_frame"], errors="coerce").astype("Int64")
    events["execution_thw_min"] = pd.to_numeric(events["execution_thw_min"], errors="coerce")

    events = events.dropna(subset=["cutter_id", "t0_frame", "execution_thw_min"]).copy()
    events["risk_thw"] = events["execution_thw_min"] < float(args.risk_thw)

    opt = BinarySFCOptions(
        alongside_s_thresh=float(args.alongside_thresh),
        max_range_ahead=float(args.range_ahead) if args.range_ahead is not None else None,
        max_range_behind=float(args.range_behind) if args.range_behind is not None else None,
        lane_col="laneIndex_xy",
        sign_col="sign",
        s_col="s",
    )

    order: SFCOrder = args.sfc_order  # type: ignore[assignment]

    rows: list[dict[str, object]] = []
    event_id = 0

    recording_ids = sorted(events["recording_id"].unique().tolist())
    for _, _, rid in iter_with_progress(
        recording_ids,
        label="Step 14 recordings",
        item_name="recording",
    ):
        ev_r = events.loc[events["recording_id"] == rid].copy()
        if ev_r.empty:
            continue

        rec = load_highd_recording(Path(args.dataset_root), rid)
        df = build_tracking_table(rec)

        markings = parse_lane_markings(rec.recording_meta)
        df = df.join(infer_lane_index(df, markings))  # adds laneIndex_xy
        df = add_sign_and_s(df)

        indexed = df.set_index(["id", "frame"], drop=False).sort_index()

        frame_rate = float(rec.recording_meta.loc[0, "frameRate"])
        pre4 = int(round(float(args.pre4_seconds) * frame_rate))
        pre2 = int(round(float(args.pre2_seconds) * frame_rate))
        post2 = int(round(float(args.post2_seconds) * frame_rate))

        # frames needed for snapshots
        frames_needed: set[int] = set()
        for _, e in ev_r.iterrows():
            t0 = int(e["t0_frame"])
            for _, a, b in stage_ranges(t0, int(frame_rate), pre4=pre4, pre2=pre2, post2=post2):
                frames_needed.update(range(a, b + 1))

        snapshots = build_lane_snapshots(
            df,
            frames_needed,
            lane_col=opt.lane_col,
            sign_col=opt.sign_col,
            s_col=opt.s_col,
        )

        for _, e in ev_r.iterrows():
            event_id += 1
            cutter_id = int(e["cutter_id"])
            t0 = int(e["t0_frame"])
            risk = bool(e["risk_thw"])

            for stage, a, b in stage_ranges(t0, int(frame_rate), pre4=pre4, pre2=pre2, post2=post2):
                for f in range(a, b + 1):
                    try:
                        cr = indexed.loc[(cutter_id, f)]
                    except KeyError:
                        continue

                    cutter_lane = int(cr["laneIndex_xy"])
                    cutter_sign = int(cr["sign"])
                    cutter_s = float(cr["s"])

                    code, _ = encode_frame_binary_sfc(
                        snapshots=snapshots,
                        indexed=indexed,
                        frame=f,
                        cutter_id=cutter_id,
                        cutter_lane=cutter_lane,
                        cutter_sign=cutter_sign,
                        cutter_s=cutter_s,
                        options=opt,
                        order=order,
                    )

                    rows.append(
                        {
                            "event_id": event_id,
                            "recording_id": rid,
                            "cutter_id": cutter_id,
                            "t0_frame": t0,
                            "stage": stage,
                            "frame": int(f),
                            "rel_t": float((f - t0) / frame_rate),
                            "risk_thw": risk,
                            "sfc_order": order,
                            "code": int(code),
                            "code_hex": f"{int(code):04x}",
                        }
                    )

        print(f"[Step14] recording {rid}: events={len(ev_r)}")

    out = pd.DataFrame(rows)
    out_path = out_dir / f"sfc_binary_codes_long_{order}.csv"
    out.to_csv(out_path, index=False)
    canonical_dir = step_reports_dir(14)
    if out_dir.resolve() == canonical_dir.resolve():
        canonical_codes = out_path
    else:
        canonical_codes = mirror_file_to_step(out_path, 14)
    details_md = write_step_markdown(
        14,
        "sfc_binary_encode_details.md",
        [
            "# Step 14 Binary SFC Encoding",
            "",
            f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Input merged CSV: `{merged_csv}`",
            f"- Dataset root: `{Path(args.dataset_root).resolve()}`",
            f"- Order: `{order}`",
            f"- Risk THW threshold: `{float(args.risk_thw):.3f}`",
            f"- Alongside threshold: `{float(args.alongside_thresh):.3f}`",
            f"- Range ahead: `{float(args.range_ahead):.3f}`",
            f"- Range behind: `{float(args.range_behind):.3f}`",
            f"- Rows exported: `{len(out)}`",
            f"- Output CSV: `{canonical_codes}`",
        ],
    )

    print("\n== Step 14: Binary SFC encoding ==")
    print("Saved:", out_path)
    print("Saved:", details_md)
    print("Rows:", len(out))
    if len(out) > 0:
        print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
