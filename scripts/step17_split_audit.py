from __future__ import annotations

import argparse
import importlib.util
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cutin_risk.paths import output_path, project_root


def normalize_recording_id(value: object) -> str:
    s = str(value).strip()
    if not s:
        return s
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    if s.isdigit():
        return str(int(s))
    return s


def _import_step15c_module() -> object:
    path = project_root() / "scripts" / "step15c_sfc_prediction.py"
    spec = importlib.util.spec_from_file_location("step15c_sfc_prediction", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_counts_step11(merged_csv: Path, thw_risk: float) -> dict[str, int]:
    df = pd.read_csv(merged_csv)
    df["risk_thw"] = pd.to_numeric(df["execution_thw_min"], errors="coerce") < float(thw_risk)
    df["decision_lat_v"] = pd.to_numeric(df["decision_cutter_lat_v_abs_max"], errors="coerce")
    df["decision_speed_delta"] = (
        pd.to_numeric(df["decision_cutter_speed_mean"], errors="coerce")
        - pd.to_numeric(df["decision_follower_speed_mean"], errors="coerce")
    )
    keep = df.dropna(subset=["recording_id", "risk_thw", "decision_lat_v", "decision_speed_delta"]).copy()
    return (
        keep["recording_id"]
        .map(normalize_recording_id)
        .value_counts(sort=False)
        .sort_index()
        .to_dict()
    )


def expected_counts_step12_or_13a(merged_csv: Path, thw_risk: float) -> dict[str, int]:
    df = pd.read_csv(merged_csv)
    df["risk_thw"] = pd.to_numeric(df["execution_thw_min"], errors="coerce") < float(thw_risk)
    df["decision_lat_v"] = pd.to_numeric(df["decision_cutter_lat_v_abs_max"], errors="coerce")
    df["decision_speed_delta"] = (
        pd.to_numeric(df["decision_cutter_speed_mean"], errors="coerce")
        - pd.to_numeric(df["decision_follower_speed_mean"], errors="coerce")
    )
    df["decision_acc_delta"] = (
        pd.to_numeric(df["decision_cutter_acc_mean"], errors="coerce")
        - pd.to_numeric(df["decision_follower_acc_mean"], errors="coerce")
    )
    df["decision_dy_abs"] = pd.to_numeric(df["decision_cutter_dy"], errors="coerce").abs()
    keep = df.dropna(
        subset=[
            "recording_id",
            "risk_thw",
            "decision_lat_v",
            "decision_speed_delta",
            "decision_acc_delta",
            "decision_dy_abs",
        ]
    ).copy()
    return (
        keep["recording_id"]
        .map(normalize_recording_id)
        .value_counts(sort=False)
        .sort_index()
        .to_dict()
    )


def expected_counts_step15c_binary(binary_long_csv: Path, stage: str, min_frames: int) -> dict[str, int]:
    module = _import_step15c_module()
    df = pd.read_csv(binary_long_csv)
    feats = module.build_features_from_binary_long(df, stage=stage, min_frames=min_frames)
    return (
        feats["recording_id"]
        .map(normalize_recording_id)
        .value_counts(sort=False)
        .sort_index()
        .to_dict()
    )


def expected_counts_step15c_weighted(weighted_csv: Path, stage: str) -> dict[str, int]:
    module = _import_step15c_module()
    df = pd.read_csv(weighted_csv)
    feats, _ = module.build_features_from_weighted_wide(df, stage=stage)
    return (
        feats["recording_id"]
        .map(normalize_recording_id)
        .value_counts(sort=False)
        .sort_index()
        .to_dict()
    )


def expected_counts_from_dataset(dataset_csv: Path) -> dict[str, int]:
    df = pd.read_csv(dataset_csv)
    if "recording_id" not in df.columns:
        raise ValueError(f"Missing recording_id in {dataset_csv}")
    return (
        df["recording_id"]
        .map(normalize_recording_id)
        .value_counts(sort=False)
        .sort_index()
        .to_dict()
    )


def _script_has_split_filters(script_path: Path) -> tuple[bool, bool]:
    if not script_path.exists():
        return False, False
    txt = script_path.read_text(encoding="utf-8")
    train_patterns = [
        r"recording_id.*!=",
        r"\brecs\b\s*!=",
        r"train_mask\s*=\s*.*!=",
        r"train\s*=\s*.*!=",
    ]
    test_patterns = [
        r"recording_id.*==",
        r"\brecs\b\s*==",
        r"test_mask\s*=\s*.*==",
        r"test\s*=\s*.*==",
    ]
    has_train = any(bool(re.search(p, txt)) for p in train_patterns)
    has_test = any(bool(re.search(p, txt)) for p in test_patterns)
    return has_train, has_test


def _map_from_loo(loo: pd.DataFrame, heldout_col: str, n_test_col: str) -> dict[str, int]:
    d = (
        loo.assign(_rid=loo[heldout_col].map(normalize_recording_id))
        .groupby("_rid", sort=False)[n_test_col]
        .sum()
    )
    return {str(k): int(v) for k, v in d.sort_index().to_dict().items()}


@dataclass(frozen=True)
class AuditTarget:
    name: str
    script_path: Path
    loo_csv: Path
    expected_counts: dict[str, int]
    expected_source: str


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = df.columns.tolist()
    rows = df.astype(str).values.tolist()
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 17: Split integrity audit for LOO outputs.")
    ap.add_argument(
        "--merged-csv",
        type=str,
        default=str(output_path("reports/step9_batch/cutin_stage_features_merged.csv")),
    )
    ap.add_argument("--thw-risk", type=float, default=0.7)
    ap.add_argument("--step15c-stage", type=str, default="decision")
    ap.add_argument("--step15c-min-frames", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default=str(output_path("reports/final")))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_csv = Path(args.merged_csv)
    if not merged_csv.exists():
        raise FileNotFoundError(f"Missing input: {merged_csv}")

    targets = [
        AuditTarget(
            name="step11_rule_search",
            script_path=project_root() / "scripts" / "step11_early_warning_rule_search.py",
            loo_csv=output_path("reports/step11_warning/leave_one_recording_out.csv"),
            expected_counts=expected_counts_step11(merged_csv, args.thw_risk),
            expected_source=str(merged_csv),
        ),
        AuditTarget(
            name="step12_logreg",
            script_path=project_root() / "scripts" / "step12_early_warning_logreg.py",
            loo_csv=output_path("reports/step12_warning_logreg/leave_one_recording_out_logreg.csv"),
            expected_counts=expected_counts_step12_or_13a(merged_csv, args.thw_risk),
            expected_source=str(merged_csv),
        ),
        AuditTarget(
            name="step13a_tuned_logreg",
            script_path=project_root() / "scripts" / "step13a_logreg_tuned.py",
            loo_csv=output_path("reports/step13a_warning_logreg/leave_one_recording_out_step13a.csv"),
            expected_counts=expected_counts_step12_or_13a(merged_csv, args.thw_risk),
            expected_source=str(merged_csv),
        ),
        AuditTarget(
            name="step15c_binary",
            script_path=project_root() / "scripts" / "step15c_sfc_prediction.py",
            loo_csv=output_path("reports/step15c_pred_binary/leave_one_recording_out.csv"),
            expected_counts=expected_counts_step15c_binary(
                output_path("reports/step15a_sfc_mirror/sfc_binary_codes_long_hilbert_mirrored.csv"),
                stage=args.step15c_stage,
                min_frames=int(args.step15c_min_frames),
            ),
            expected_source=str(output_path("reports/step15a_sfc_mirror/sfc_binary_codes_long_hilbert_mirrored.csv")),
        ),
        AuditTarget(
            name="step15c_weighted_distance",
            script_path=project_root() / "scripts" / "step15c_sfc_prediction.py",
            loo_csv=output_path("reports/step15c_pred_weighted_distance/leave_one_recording_out.csv"),
            expected_counts=expected_counts_step15c_weighted(
                output_path("reports/step15b_sfc_weighted/sfc_weighted_stage_features_distance_hilbert.csv"),
                stage=args.step15c_stage,
            ),
            expected_source=str(
                output_path("reports/step15b_sfc_weighted/sfc_weighted_stage_features_distance_hilbert.csv")
            ),
        ),
        AuditTarget(
            name="step15c_weighted_ttc",
            script_path=project_root() / "scripts" / "step15c_sfc_prediction.py",
            loo_csv=output_path("reports/step15c_pred_weighted_ttc/leave_one_recording_out.csv"),
            expected_counts=expected_counts_step15c_weighted(
                output_path("reports/step15b_sfc_weighted/sfc_weighted_stage_features_ttc_hilbert.csv"),
                stage=args.step15c_stage,
            ),
            expected_source=str(output_path("reports/step15b_sfc_weighted/sfc_weighted_stage_features_ttc_hilbert.csv")),
        ),
        AuditTarget(
            name="step16_lanechange",
            script_path=project_root() / "scripts" / "step16_sfc_predict_lanechange_cutin.py",
            loo_csv=output_path("reports/step16_sfc_predict/lanechange_loocv.csv"),
            expected_counts=expected_counts_from_dataset(output_path("reports/step16_sfc_predict/sfc_lanechange_dataset.csv")),
            expected_source=str(output_path("reports/step16_sfc_predict/sfc_lanechange_dataset.csv")),
        ),
        AuditTarget(
            name="step16_cutin",
            script_path=project_root() / "scripts" / "step16_sfc_predict_lanechange_cutin.py",
            loo_csv=output_path("reports/step16_sfc_predict/cutin_loocv.csv"),
            expected_counts=expected_counts_from_dataset(output_path("reports/step16_sfc_predict/sfc_cutin_dataset.csv")),
            expected_source=str(output_path("reports/step16_sfc_predict/sfc_cutin_dataset.csv")),
        ),
    ]

    rows: list[dict[str, object]] = []

    for t in targets:
        has_train_filter, has_test_filter = _script_has_split_filters(t.script_path)
        base = {
            "name": t.name,
            "script": str(t.script_path),
            "loo_csv": str(t.loo_csv),
            "expected_source": t.expected_source,
            "file_exists": t.loo_csv.exists(),
            "code_has_train_filter": has_train_filter,
            "code_has_test_filter": has_test_filter,
        }

        if not t.loo_csv.exists():
            rows.append(
                base
                | {
                    "rows": 0,
                    "unique_heldout": 0,
                    "duplicate_heldout": False,
                    "min_n_test": None,
                    "max_n_test": None,
                    "counts_match_expected": False,
                    "recording_set_match": False,
                    "sum_n_test_matches_expected": False,
                    "overall_pass": False,
                    "note": "Missing LOO CSV.",
                }
            )
            continue

        loo = pd.read_csv(t.loo_csv)
        required = {"heldout_recording", "n_test"}
        missing_cols = sorted(required - set(loo.columns))
        if missing_cols:
            rows.append(
                base
                | {
                    "rows": int(len(loo)),
                    "unique_heldout": 0,
                    "duplicate_heldout": False,
                    "min_n_test": None,
                    "max_n_test": None,
                    "counts_match_expected": False,
                    "recording_set_match": False,
                    "sum_n_test_matches_expected": False,
                    "overall_pass": False,
                    "note": f"Missing required columns: {missing_cols}",
                }
            )
            continue

        loo = loo.copy()
        loo["_rid"] = loo["heldout_recording"].map(normalize_recording_id)
        loo["n_test"] = pd.to_numeric(loo["n_test"], errors="coerce")
        loo = loo.dropna(subset=["_rid", "n_test"]).copy()
        loo["n_test"] = loo["n_test"].astype(int)

        loo_counts = _map_from_loo(loo, "_rid", "n_test")
        expected_counts = {normalize_recording_id(k): int(v) for k, v in t.expected_counts.items()}

        rec_set_match = set(loo_counts.keys()) == set(expected_counts.keys())
        counts_match = loo_counts == expected_counts
        sum_match = sum(loo_counts.values()) == sum(expected_counts.values())
        dup_heldout = int(loo["_rid"].nunique()) != int(len(loo))

        note = ""
        if not rec_set_match:
            miss = sorted(set(expected_counts.keys()) - set(loo_counts.keys()))
            extra = sorted(set(loo_counts.keys()) - set(expected_counts.keys()))
            note = f"Recording set mismatch. missing={miss} extra={extra}"
        elif not counts_match:
            bad = []
            for rid in sorted(expected_counts.keys()):
                if loo_counts.get(rid) != expected_counts.get(rid):
                    bad.append(f"{rid}:{loo_counts.get(rid)}!={expected_counts.get(rid)}")
            note = "Count mismatch: " + ", ".join(bad[:10])

        overall_pass = bool(
            rec_set_match
            and counts_match
            and sum_match
            and not dup_heldout
            and has_train_filter
            and has_test_filter
        )

        rows.append(
            base
            | {
                "rows": int(len(loo)),
                "unique_heldout": int(loo["_rid"].nunique()),
                "duplicate_heldout": bool(dup_heldout),
                "min_n_test": int(loo["n_test"].min()) if len(loo) else None,
                "max_n_test": int(loo["n_test"].max()) if len(loo) else None,
                "counts_match_expected": bool(counts_match),
                "recording_set_match": bool(rec_set_match),
                "sum_n_test_matches_expected": bool(sum_match),
                "overall_pass": overall_pass,
                "note": note,
            }
        )

    report_df = pd.DataFrame(rows)
    out_csv = out_dir / "split_audit.csv"
    report_df.to_csv(out_csv, index=False)

    now = datetime.now(timezone.utc).isoformat()
    passed = int(report_df["overall_pass"].fillna(False).sum())
    total = int(len(report_df))

    short_df = report_df[
        [
            "name",
            "overall_pass",
            "recording_set_match",
            "counts_match_expected",
            "sum_n_test_matches_expected",
            "code_has_train_filter",
            "code_has_test_filter",
            "note",
        ]
    ].copy()

    md_lines = [
        "# Split Audit Report",
        "",
        f"Generated: `{now}`",
        "",
        f"Overall: **{passed}/{total}** targets passed.",
        "",
        "## Summary",
        "",
        _to_markdown_table(short_df),
        "",
        "## Notes",
        "",
        "- `counts_match_expected` checks per-recording `n_test` from each LOO file against independently rebuilt source counts.",
        "- `code_has_train_filter` / `code_has_test_filter` are static checks for `recording_id != ...` and `recording_id == ...` logic in script source.",
        f"- Full machine-readable report: `{out_csv}`",
    ]

    out_md = out_dir / "split_audit.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("== Step 17: Split audit ==")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Pass: {passed}/{total}")


if __name__ == "__main__":
    main()
