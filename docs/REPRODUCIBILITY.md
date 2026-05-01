# Reproducibility Guide

This guide documents the public rerun path for the approved thesis pipeline.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the code installation:

```bash
python -m pytest
python -m ruff check .
```

## 2. Configure Data Paths

Create `configs/paths.local.json`:

```json
{
  "paths": {
    "dataset_root": "/absolute/path/to/highD/data",
    "outputs_root": "outputs"
  }
}
```

Confirm that the dataset directory contains files such as:

```text
01_tracks.csv
01_tracksMeta.csv
01_recordingMeta.csv
```

## 3. Inspect Available Phases

```bash
python scripts/run_thesis_phases.py --list-phases
```

The retained macro-phases are:

| Phase | Purpose |
| --- | --- |
| `preliminary-processing` | Recording checks and lane-change summaries. |
| `interaction-mining` | Cut-in extraction and event statistics. |
| `core-reconstruction` | Neighbor and XY-lane reconstruction checks. |
| `safety-quantification` | Stage features and pair-specific risk summaries. |
| `context-signatures` | SFC encoding, mirroring, and archetype analysis. |
| `reference-label-evaluation` | Bootstrap confidence intervals for agreement metrics. |

## 4. Run The Approved Thesis Pipeline

```bash
PYTHON_BIN=.venv/bin/python ./scripts/run_final_thesis_pipeline.sh
```

The wrapper runs:

1. `interaction-mining`
2. `core-reconstruction`
3. `safety-quantification`
4. `context-signatures`
5. `reference-label-evaluation`

It also writes a command log and run manifest under:

```text
outputs/reports/final/
```

## 5. Expected Output Locations

| Output | Path |
| --- | --- |
| Cut-in details | `outputs/reports/Step 03/cutin_details.md` |
| Reconstruction metrics | `outputs/reports/Step 07/xy_lane_pipeline_metrics_summary.csv` |
| Risk summary | `outputs/reports/Step 10/risk_summary_by_recording.csv` |
| Archetype report | `outputs/reports/Step 15D/archetype_report.md` |
| Final confidence intervals | `outputs/reports/final/metrics_with_ci.md` |
| Final run manifest | `outputs/reports/final/run_manifest.json` |

## 6. Notes On Exact Reproduction

- Use the committed `configs/detection.json` and `configs/thesis.json` unless documenting a deliberate variant.
- Keep `indicators.position_reference = "bbox_topleft"` for the highD thesis run.
- Do not compare runs made with different dataset versions, local filters, or uncommitted config overrides without recording those differences.
- The generated manifest records Python version, git commit, branch, timestamp, and commands.
