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
from cutin_risk.reconstruction.lanes import LaneInferenceOptions, parse_lane_markings, infer_lane_index

from cutin_risk.encoding.sfc_binary import (
    BinarySFCOptions,
    build_lane_snapshots,
    encode_frame_binary_sfc,
    SFCOrder,
)
from cutin_risk.thesis_config import thesis_float, thesis_int, thesis_optional_float, thesis_str


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


def parse_optional_nonnegative_float(value: str) -> float | None:
    """Parse non-negative float or disable token for optional CLI numeric args."""
    token = str(value).strip().lower()
    if token in {"none", "null", "off", "disable", "disabled", "no-limit", "nolimit", "inf", "infinite"}:
        return None
    try:
        parsed = float(token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected non-negative float or 'none', got {value!r}") from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("Value must be >= 0.0")
    return parsed


def parse_neighbor_id(raw_value: object) -> int:
    """Normalize neighbor ids to positive ints, mapping missing/invalid to 0."""
    try:
        value = int(raw_value)
    except Exception:
        try:
            value = int(float(raw_value))
        except Exception:
            return 0
    return value if value > 0 else 0


def build_highd_reference_matrix(cutter_row: pd.Series) -> np.ndarray:
    """
    Build a 3x3 occupancy matrix from highD raw neighbor ids.

    rows: preceding/alongside/following
    cols: left/same/right
    """
    g = np.zeros((3, 3), dtype=np.uint8)
    g[1, 1] = 1

    id_columns = (
        ("leftPrecedingId", "leftAlongsideId", "leftFollowingId"),
        ("precedingId", None, "followingId"),
        ("rightPrecedingId", "rightAlongsideId", "rightFollowingId"),
    )

    for col, (preceding_col, alongside_col, following_col) in enumerate(id_columns):
        for row, column_name in ((0, preceding_col), (1, alongside_col), (2, following_col)):
            if column_name is None:
                continue
            if parse_neighbor_id(cutter_row.get(column_name, 0)) != 0:
                g[row, col] = 1

    return g


def matrix_signature(g3: np.ndarray) -> str:
    """Compact 3x3 signature string, e.g. 011|010|011."""
    return "|".join("".join(str(int(v)) for v in row) for row in np.asarray(g3, dtype=int))


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
    ap.add_argument(
        "--range-ahead",
        type=parse_optional_nonnegative_float,
        default=thesis_optional_float("step14.range_ahead", None, min_value=0.0),
        help="Max ahead range in meters; use 'none' to disable.",
    )
    ap.add_argument(
        "--range-behind",
        type=parse_optional_nonnegative_float,
        default=thesis_optional_float("step14.range_behind", None, min_value=0.0),
        help="Max behind range in meters; use 'none' to disable.",
    )

    ap.add_argument("--pre4-seconds", type=float, default=thesis_float("step14.pre4_seconds", 4.0, min_value=0.0))
    ap.add_argument("--pre2-seconds", type=float, default=thesis_float("step14.pre2_seconds", 2.0, min_value=0.0))
    ap.add_argument("--post2-seconds", type=float, default=thesis_float("step14.post2_seconds", 2.0, min_value=0.0))
    ap.add_argument(
        "--lane-boundary-eps",
        type=float,
        default=thesis_float("step14.lane_boundary_eps", 0.0, min_value=0.0),
        help="Directional lane-boundary bias (meters) used during lane inference.",
    )
    ap.add_argument(
        "--mismatch-sample-limit",
        type=int,
        default=thesis_int("step14.mismatch_sample_limit", 12, min_value=0),
        help="Maximum mismatch examples written into Step 14 markdown details.",
    )

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
        max_range_ahead=args.range_ahead,
        max_range_behind=args.range_behind,
        lane_col="laneIndex_xy",
        sign_col="sign",
        s_col="s",
    )

    order: SFCOrder = args.sfc_order  # type: ignore[assignment]

    rows: list[dict[str, object]] = []
    event_id = 0
    eval_rows = 0
    eval_match_raw = 0
    eval_mismatch_raw = 0
    eval_match = 0
    eval_mismatch = 0
    eval_by_stage: dict[str, dict[str, int]] = {}
    eval_flags: list[dict[str, object]] = []
    mismatch_samples: list[dict[str, object]] = []

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
        lane_options = LaneInferenceOptions(lane_boundary_eps=float(args.lane_boundary_eps))
        df = df.join(infer_lane_index(df, markings, options=lane_options))  # adds laneIndex_xy
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

                    code, derived_g3 = encode_frame_binary_sfc(
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
                    reference_g3 = build_highd_reference_matrix(cr)
                    is_match_raw = bool(
                        np.array_equal(
                            np.asarray(derived_g3, dtype=np.uint8),
                            np.asarray(reference_g3, dtype=np.uint8),
                        )
                    )
                    derived_g3_eval = np.asarray(derived_g3, dtype=np.uint8)
                    try:
                        driving_direction = int(cr.get("drivingDirection", 0))
                    except Exception:
                        driving_direction = 0
                    if driving_direction == 1:
                        # For upper-lane direction, lane-index left/right is mirrored w.r.t. highD ego-centric IDs.
                        derived_g3_eval = np.fliplr(derived_g3_eval)
                    is_match = bool(
                        np.array_equal(
                            np.asarray(derived_g3_eval, dtype=np.uint8),
                            np.asarray(reference_g3, dtype=np.uint8),
                        )
                    )
                    eval_rows += 1
                    if is_match_raw:
                        eval_match_raw += 1
                    else:
                        eval_mismatch_raw += 1
                    if is_match:
                        eval_match += 1
                    else:
                        eval_mismatch += 1
                    eval_flags.append(
                        {
                            "event_id": int(event_id),
                            "stage": str(stage),
                            "match": bool(is_match),
                        }
                    )

                    stage_key = str(stage)
                    if stage_key not in eval_by_stage:
                        eval_by_stage[stage_key] = {"rows": 0, "match": 0, "mismatch": 0}
                    eval_by_stage[stage_key]["rows"] += 1
                    eval_by_stage[stage_key]["match" if is_match else "mismatch"] += 1

                    if (not is_match) and (len(mismatch_samples) < int(args.mismatch_sample_limit)):
                        mismatch_samples.append(
                            {
                                "recording_id": rid,
                                "event_id": event_id,
                                "cutter_id": cutter_id,
                                "stage": stage,
                                "frame": int(f),
                                "code_hex": f"{int(code):04x}",
                                "derived": matrix_signature(np.asarray(derived_g3, dtype=np.uint8)),
                                "derived_dirnorm": matrix_signature(np.asarray(derived_g3_eval, dtype=np.uint8)),
                                "highd": matrix_signature(np.asarray(reference_g3, dtype=np.uint8)),
                            }
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
    stage_window_any_match_rate = float("nan")
    stage_window_all_match_rate = float("nan")
    event_any_match_rate = float("nan")
    event_all_match_rate = float("nan")
    eval_event_stage_by_stage: dict[str, dict[str, float]] = {}
    if eval_flags:
        eval_df = pd.DataFrame(eval_flags)
        event_stage_df = (
            eval_df.groupby(["event_id", "stage"], as_index=False)["match"]
            .agg(stage_any="max", stage_all="min", row_match_rate="mean")
            .copy()
        )
        if not event_stage_df.empty:
            stage_window_any_match_rate = float(event_stage_df["stage_any"].mean())
            stage_window_all_match_rate = float(event_stage_df["stage_all"].mean())
            grouped = event_stage_df.groupby("stage", as_index=False).agg(
                stage_windows=("stage_any", "size"),
                any_match_rate=("stage_any", "mean"),
                all_match_rate=("stage_all", "mean"),
                row_match_rate=("row_match_rate", "mean"),
            )
            for _, g_row in grouped.iterrows():
                stage_key = str(g_row["stage"])
                eval_event_stage_by_stage[stage_key] = {
                    "stage_windows": float(g_row["stage_windows"]),
                    "any_match_rate": float(g_row["any_match_rate"]),
                    "all_match_rate": float(g_row["all_match_rate"]),
                    "row_match_rate": float(g_row["row_match_rate"]),
                }

        event_df = eval_df.groupby("event_id", as_index=False)["match"].agg(any_match="max", all_match="min")
        if not event_df.empty:
            event_any_match_rate = float(event_df["any_match"].mean())
            event_all_match_rate = float(event_df["all_match"].mean())
    details_lines = [
        "# Step 14 Binary SFC Encoding",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input merged CSV: `{merged_csv}`",
        f"- Dataset root: `{Path(args.dataset_root).resolve()}`",
        f"- Order: `{order}`",
        f"- Risk THW threshold: `{float(args.risk_thw):.3f}`",
        f"- Alongside threshold: `{float(args.alongside_thresh):.3f}`",
        f"- Range ahead: `{'disabled' if args.range_ahead is None else f'{float(args.range_ahead):.3f}'}`",
        f"- Range behind: `{'disabled' if args.range_behind is None else f'{float(args.range_behind):.3f}'}`",
        f"- Lane-boundary epsilon: `{float(args.lane_boundary_eps):.3f}`",
        f"- Rows exported: `{len(out)}`",
        f"- Output CSV: `{canonical_codes}`",
        "",
        "## Matrix Verification Against highD Raw IDs",
        "",
        "- Comparison mode: direction-normalized (mirror columns for `drivingDirection=1`).",
        f"- Verified rows: `{eval_rows}`",
        f"- Matches: `{eval_match}`",
        f"- Mismatches: `{eval_mismatch}`",
        (
            f"- Match rate: `{(100.0 * eval_match / eval_rows):.2f}%`"
            if eval_rows > 0
            else "- Match rate: `n/a`"
        ),
        f"- Raw (unmirrored) matches: `{eval_match_raw}`",
        f"- Raw (unmirrored) mismatches: `{eval_mismatch_raw}`",
        (
            f"- Raw (unmirrored) match rate: `{(100.0 * eval_match_raw / eval_rows):.2f}%`"
            if eval_rows > 0
            else "- Raw (unmirrored) match rate: `n/a`"
        ),
        "",
        "### Event-Level Verification (Thesis View)",
        "",
        (
            f"- Event-stage window ANY-frame match rate: `{(100.0 * stage_window_any_match_rate):.2f}%`"
            if pd.notna(stage_window_any_match_rate)
            else "- Event-stage window ANY-frame match rate: `n/a`"
        ),
        (
            f"- Event-stage window ALL-frame match rate: `{(100.0 * stage_window_all_match_rate):.2f}%`"
            if pd.notna(stage_window_all_match_rate)
            else "- Event-stage window ALL-frame match rate: `n/a`"
        ),
        (
            f"- Event ANY-frame match rate: `{(100.0 * event_any_match_rate):.2f}%`"
            if pd.notna(event_any_match_rate)
            else "- Event ANY-frame match rate: `n/a`"
        ),
        (
            f"- Event ALL-frame match rate: `{(100.0 * event_all_match_rate):.2f}%`"
            if pd.notna(event_all_match_rate)
            else "- Event ALL-frame match rate: `n/a`"
        ),
        "",
        "### Per-stage Verification",
        "",
        "| Stage | Rows | Matches | Mismatches | Match rate |",
        "|---|---:|---:|---:|---:|",
    ]
    stage_lines = [
        (
            f"| {stage_name} | {stats['rows']} | {stats['match']} | {stats['mismatch']} | "
            f"{(100.0 * stats['match'] / stats['rows']):.2f}% |"
        )
        for stage_name, stats in sorted(eval_by_stage.items())
        if stats["rows"] > 0
    ]
    if stage_lines:
        details_lines.extend(stage_lines)
    else:
        details_lines.append("| n/a | 0 | 0 | 0 | n/a |")
    details_lines.extend(
        [
            "",
            "### Per-stage Event-Level Verification",
            "",
            "| Stage | Stage windows | ANY-frame match | ALL-frame match | Mean row match |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if eval_event_stage_by_stage:
        details_lines.extend(
            [
                (
                    f"| {stage_name} | {int(stats['stage_windows'])} | "
                    f"{(100.0 * float(stats['any_match_rate'])):.2f}% | "
                    f"{(100.0 * float(stats['all_match_rate'])):.2f}% | "
                    f"{(100.0 * float(stats['row_match_rate'])):.2f}% |"
                )
                for stage_name, stats in sorted(eval_event_stage_by_stage.items())
            ]
        )
    else:
        details_lines.append("| n/a | 0 | n/a | n/a | n/a |")

    details_lines.extend(
        [
            "",
            "### Sample Mismatches",
            "",
        ]
    )
    if mismatch_samples:
        details_lines.extend(
            [
                "| recording_id | event_id | cutter_id | stage | frame | code_hex | derived_g3 | derived_dirnorm_g3 | highd_g3 |",
                "|---|---:|---:|---|---:|---|---|---|---|",
            ]
        )
        details_lines.extend(
            [
                (
                    f"| {m['recording_id']} | {m['event_id']} | {m['cutter_id']} | {m['stage']} | "
                    f"{m['frame']} | {m['code_hex']} | {m['derived']} | {m['derived_dirnorm']} | {m['highd']} |"
                )
                for m in mismatch_samples
            ]
        )
    else:
        details_lines.append("None (no mismatches found in verification).")

    details_md = write_step_markdown(14, "sfc_binary_encode_details.md", details_lines)

    print("\n== Step 14: Binary SFC encoding ==")
    print("Saved:", out_path)
    print("Saved:", details_md)
    print("Rows:", len(out))
    if eval_rows > 0:
        print(
            "Matrix verification: match={} mismatch={} rate={:.2f}%".format(
                eval_match,
                eval_mismatch,
                100.0 * eval_match / eval_rows,
            )
        )
        print(
            "Matrix verification (raw/unmirrored): match={} mismatch={} rate={:.2f}%".format(
                eval_match_raw,
                eval_mismatch_raw,
                100.0 * eval_match_raw / eval_rows,
            )
        )
    if pd.notna(event_any_match_rate):
        print(
            "Matrix verification (event-level): stage-any={:.2f}% event-any={:.2f}%".format(
                100.0 * stage_window_any_match_rate if pd.notna(stage_window_any_match_rate) else float("nan"),
                100.0 * event_any_match_rate,
            )
        )
    if len(out) > 0:
        print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
