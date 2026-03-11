#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

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
run_cmd "$PYTHON_BIN" scripts/run_thesis_phases.py \
  --python-bin "$PYTHON_BIN" \
  --phases interaction-mining core-reconstruction safety-quantification context-signatures reference-label-evaluation \
  --command-log "$CMD_LOG"

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
echo "  - outputs/reports/Step 03/cutin_details.md"
echo "  - outputs/reports/Step 07/xy_lane_pipeline_metrics_summary.csv"
echo "  - outputs/reports/Step 10/risk_summary_by_recording.csv"
echo "  - outputs/reports/Step 15D/archetype_report.md"
echo "  - $FINAL_DIR/metrics_with_ci.md"
echo "  - $FINAL_DIR/run_manifest.json"
