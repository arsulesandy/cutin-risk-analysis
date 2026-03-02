#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RECORDINGS_BATCH="${RECORDINGS_BATCH:-$("$PYTHON_BIN" -c 'from cutin_risk.thesis_config import thesis_str; print(thesis_str("pipeline.recordings_batch", "all"))')}"
RECORDINGS_STEP16="${RECORDINGS_STEP16:-$("$PYTHON_BIN" -c 'from cutin_risk.thesis_config import thesis_str; print(thesis_str("pipeline.recordings_step16", "01,02,03,04,05"))')}"
THW_RISK="${THW_RISK:-$("$PYTHON_BIN" -c 'from cutin_risk.thesis_config import thesis_float; print(thesis_float("pipeline.thw_risk", 0.7, min_value=0.0))')}"
CI_BOOTSTRAP="${CI_BOOTSTRAP:-$("$PYTHON_BIN" -c 'from cutin_risk.thesis_config import thesis_int; print(thesis_int("pipeline.ci_bootstrap", 3000, min_value=1))')}"
CI_SEED="${CI_SEED:-$("$PYTHON_BIN" -c 'from cutin_risk.thesis_config import thesis_int; print(thesis_int("pipeline.ci_seed", 7))')}"

FINAL_DIR="outputs/reports/final"
mkdir -p "$FINAL_DIR"

CMD_LOG="$FINAL_DIR/run_final_commands.log"
: > "$CMD_LOG"

run_cmd() {
  echo
  echo "[RUN] $*"
  echo "$*" >> "$CMD_LOG"
  "$@"
}

echo "== Final Thesis Pipeline =="
echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "RECORDINGS_BATCH=$RECORDINGS_BATCH"
echo "RECORDINGS_STEP16=$RECORDINGS_STEP16"
echo "THW_RISK=$THW_RISK"
echo "CI_BOOTSTRAP=$CI_BOOTSTRAP"
echo "CI_SEED=$CI_SEED"

run_cmd "$PYTHON_BIN" scripts/step09_batch_stage_features.py --recordings "$RECORDINGS_BATCH"
run_cmd "$PYTHON_BIN" scripts/step10_risk_report.py --thw-risk "$THW_RISK"
run_cmd "$PYTHON_BIN" scripts/step11_early_warning_rule_search.py --thw-risk "$THW_RISK"
run_cmd "$PYTHON_BIN" scripts/step12_early_warning_logreg.py --thw-risk "$THW_RISK"
run_cmd "$PYTHON_BIN" scripts/step13a_logreg_tuned.py --thw-risk "$THW_RISK"
run_cmd "$PYTHON_BIN" scripts/step13b_realistic_follower_warning.py --thw-risk "$THW_RISK"

run_cmd "$PYTHON_BIN" scripts/step14_sfc_binary_encode.py
run_cmd "$PYTHON_BIN" scripts/step14_sfc_binary_report.py
run_cmd "$PYTHON_BIN" scripts/step15a_sfc_mirror_normalize.py \
  --codes-csv outputs/reports/step14_sfc_binary/sfc_binary_codes_long_hilbert.csv
run_cmd "$PYTHON_BIN" scripts/step15b_sfc_weighted_stage_features.py --mode distance
run_cmd "$PYTHON_BIN" scripts/step15b_sfc_weighted_stage_features.py --mode ttc

run_cmd "$PYTHON_BIN" scripts/step15c_sfc_prediction.py \
  --input-type binary-long \
  --input-csv outputs/reports/step15a_sfc_mirror/sfc_binary_codes_long_hilbert_mirrored.csv \
  --stage decision \
  --out-dir outputs/reports/step15c_pred_binary

run_cmd "$PYTHON_BIN" scripts/step15c_sfc_prediction.py \
  --input-type weighted-wide \
  --input-csv outputs/reports/step15b_sfc_weighted/sfc_weighted_stage_features_distance_hilbert.csv \
  --stage decision \
  --out-dir outputs/reports/step15c_pred_weighted_distance

run_cmd "$PYTHON_BIN" scripts/step15c_sfc_prediction.py \
  --input-type weighted-wide \
  --input-csv outputs/reports/step15b_sfc_weighted/sfc_weighted_stage_features_ttc_hilbert.csv \
  --stage decision \
  --out-dir outputs/reports/step15c_pred_weighted_ttc

run_cmd "$PYTHON_BIN" scripts/step16_sfc_predict_lanechange_cutin.py --recordings "$RECORDINGS_STEP16"

run_cmd "$PYTHON_BIN" scripts/step17_split_audit.py --thw-risk "$THW_RISK"
run_cmd "$PYTHON_BIN" scripts/step17_geometry_audit.py
run_cmd "$PYTHON_BIN" scripts/step18_metrics_ci.py --n-bootstrap "$CI_BOOTSTRAP" --seed "$CI_SEED"

run_cmd "$PYTHON_BIN" - <<'PY'
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
final_dir = root / "outputs" / "reports" / "final"
cmd_log = final_dir / "run_final_commands.log"
commands = cmd_log.read_text(encoding="utf-8").splitlines() if cmd_log.exists() else []

def run_quiet(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""

manifest = {
    "utc_time": datetime.now(timezone.utc).isoformat(),
    "root_dir": str(root),
    "python_version": platform.python_version(),
    "git_commit": run_quiet(["git", "rev-parse", "HEAD"]),
    "git_branch": run_quiet(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    "commands": commands,
}

out = final_dir / "run_manifest.json"
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote manifest: {out}")
PY

echo
echo "Pipeline completed."
echo "Key outputs:"
echo "  - $FINAL_DIR/split_audit.md"
echo "  - $FINAL_DIR/metrics_with_ci.md"
echo "  - $FINAL_DIR/run_manifest.json"
