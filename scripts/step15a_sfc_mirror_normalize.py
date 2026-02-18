from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cutin_risk.encoding.sfc_binary import decode_grid_4x4_bits, encode_grid_4x4_bits


def normalize_recording_id(v: object) -> str:
    s = str(v).strip()
    return f"{int(s):02d}" if s.isdigit() else s


def mirror_grid_4x4(g4):
    """
    Mirror left/right for the meaningful 3 columns (0..2).
    Column 3 is padding (zeros) and left untouched.
    """
    g = g4.copy()
    g[:, [0, 2]] = g[:, [2, 0]]
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 15A: Mirror-normalize binary SFC codes (left vs right).")
    ap.add_argument("--codes-csv", required=True, help="Step14 long codes CSV (binary).")
    ap.add_argument(
        "--events-csv",
        default="outputs/reports/step9_batch/cutin_stage_features_merged.csv",
        help="Merged event table containing from_lane and to_lane.",
    )
    ap.add_argument("--from-col", default="from_lane")
    ap.add_argument("--to-col", default="to_lane")

    ap.add_argument(
        "--canonical-direction",
        choices=["positive", "negative"],
        default="positive",
        help="Canonical lane-change direction. If 'positive', mirror events where (to-from)<0.",
    )

    ap.add_argument("--out-dir", default="/Users/sandeep/IdeaProjects/cutin-risk-analysis/outputs/reports/step15a_sfc_mirror")
    args = ap.parse_args()

    codes_csv = Path(args.codes_csv)
    if not codes_csv.exists():
        raise FileNotFoundError(f"Missing codes CSV: {codes_csv}")

    events_csv = Path(args.events_csv)
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events CSV: {events_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = pd.read_csv(codes_csv)
    need_codes = {"recording_id", "cutter_id", "t0_frame", "code", "sfc_order"}
    missing = need_codes - set(codes.columns)
    if missing:
        raise ValueError(f"codes CSV missing columns: {sorted(missing)}")

    # Make sure we have an event id; if not, create one deterministically
    if "event_id" not in codes.columns:
        codes = codes.copy()
        codes["recording_id"] = codes["recording_id"].apply(normalize_recording_id)
        codes["event_id"] = (
                codes.groupby(["recording_id", "cutter_id", "t0_frame"], sort=False).ngroup() + 1
        )

    events = pd.read_csv(events_csv)
    need_events = {"recording_id", "cutter_id", "t0_frame", args.from_col, args.to_col}
    missing_e = need_events - set(events.columns)
    if missing_e:
        raise ValueError(
            f"events CSV missing columns: {sorted(missing_e)}. "
            f"Make sure it contains {args.from_col} and {args.to_col}."
        )

    events = events.copy()
    events["recording_id"] = events["recording_id"].apply(normalize_recording_id)
    events["cutter_id"] = pd.to_numeric(events["cutter_id"], errors="coerce").astype("Int64")
    events["t0_frame"] = pd.to_numeric(events["t0_frame"], errors="coerce").astype("Int64")
    events[args.from_col] = pd.to_numeric(events[args.from_col], errors="coerce")
    events[args.to_col] = pd.to_numeric(events[args.to_col], errors="coerce")
    events = events.dropna(subset=["cutter_id", "t0_frame", args.from_col, args.to_col]).copy()

    events["lane_delta"] = events[args.to_col] - events[args.from_col]

    if args.canonical_direction == "positive":
        events["mirror"] = events["lane_delta"] < 0
    else:
        events["mirror"] = events["lane_delta"] > 0

    # Join mirror flags into codes
    codes = codes.copy()
    codes["recording_id"] = codes["recording_id"].apply(normalize_recording_id)

    join_cols = ["recording_id", "cutter_id", "t0_frame"]
    mirror_map = events[join_cols + ["mirror", "lane_delta"]].drop_duplicates(subset=join_cols)

    merged = codes.merge(mirror_map, on=join_cols, how="left")
    merged["mirror"] = merged["mirror"].fillna(False).astype(bool)
    merged["lane_delta"] = merged["lane_delta"].fillna(0.0)

    # Fast caching: many codes repeat
    cache: dict[tuple[str, int], int] = {}

    def transform(row) -> int:
        if not row["mirror"]:
            return int(row["code"])
        order = str(row["sfc_order"])
        code = int(row["code"])
        key = (order, code)
        if key in cache:
            return cache[key]
        g4 = decode_grid_4x4_bits(code, order=order)
        g4m = mirror_grid_4x4(g4)
        out_code = encode_grid_4x4_bits(g4m, order=order)
        cache[key] = int(out_code)
        return int(out_code)

    merged["code_mirrored"] = merged.apply(transform, axis=1)
    merged["code_hex_mirrored"] = merged["code_mirrored"].map(lambda c: f"{int(c):04x}")

    # Replace primary fields with mirrored versions (keep originals too)
    merged["code_original"] = merged["code"]
    merged["code_hex_original"] = merged.get("code_hex", merged["code"].map(lambda c: f"{int(c):04x}"))

    merged["code"] = merged["code_mirrored"]
    merged["code_hex"] = merged["code_hex_mirrored"]

    out_path = out_dir / f"{codes_csv.stem}_mirrored.csv"
    merged.to_csv(out_path, index=False)

    # Summary
    n_events = merged["event_id"].nunique()
    n_mir_events = merged.loc[merged["mirror"], "event_id"].nunique()
    n_rows = len(merged)
    n_mir_rows = int(merged["mirror"].sum())

    print("== Step 15A: Mirror normalization ==")
    print("Input:", codes_csv)
    print("Saved:", out_path)
    print(f"Events: {n_events} | mirrored events: {n_mir_events}")
    print(f"Rows: {n_rows} | mirrored rows: {n_mir_rows}")


if __name__ == "__main__":
    main()
