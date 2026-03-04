# Cut-in Risk Analysis

This repository contains the code developed as part of a Master’s thesis on **cut-in lane-change risk analysis** using naturalistic highway trajectory data.

The project focuses on detecting cut-in scenarios in a **clear and reproducible way** and analysing how different risk indicators behave during these manoeuvres. The work is intended for **offline analysis and research use**, not for real-time driving systems or controllers.

The initial implementation is based on the **highD dataset**, but the codebase is structured to allow additional trajectory datasets to be integrated later without changing the core analysis logic.

---

## Scope of the project

The main objectives of this project are:

- To detect lane-change events from trajectory data using explicit, rule-based logic
- To identify cut-in scenarios based on vehicle neighbour relations
- To compute established surrogate risk indicators as time series
- To compare indicators in terms of timing, severity, and agreement
- To provide a maintainable and reproducible analysis pipeline

The goal is not to define a universal standard for cut-in risk, but to make assumptions, definitions, and results transparent and repeatable.

---

## What the pipeline does

The analysis pipeline is organized into four main stages:

### 1. Dataset preparation
- Load vehicle trajectories and metadata
- Normalize time, units, and coordinate conventions
- Validate lane assignments and neighbour relationships
- Perform basic data quality checks

### 2. Scenario detection
- Detect lane-change events based on lane assignment over time
- Identify cut-in situations when a lane-changing vehicle becomes the immediate predecessor of another vehicle in the target lane
- Extract consistent time windows before, during, and after the manoeuvre

### 3. Risk indicator computation
For each cut-in scenario, the thesis pipeline computes:
- Distance headway (DHW)
- Time headway (THW)
- Time-to-Collision (TTC)

Indicators are computed as time series to capture how risk evolves over the manoeuvre.
Additional stage-level features and SFC encodings are built on top of these base indicators.

### 4. Analysis and inspection
- Compare indicator timing and severity
- Analyse agreement and disagreement between indicators
- Generate summary statistics and figures
- Visually inspect selected scenarios for sanity checking

---

## Supported datasets

### highD (primary dataset)

The project currently supports the highD dataset, which consists of drone-recorded highway vehicle trajectories with detailed neighbour information.

Due to licensing restrictions, **raw highD data is not included** in this repository. You must obtain access separately and store the data locally.

### Extending to other datasets

The codebase is designed to be dataset-agnostic. Support for additional trajectory datasets can be added by implementing a dataset adapter, without modifying the detection or analysis logic.

---

## Repository structure (overview)

- `src/cutin_risk/`  
  Core library code, including dataset adapters, detection logic, risk indicators, analysis utilities, and visualization tools.

- `configs/`  
  Configuration files for datasets and experiments. All runs are driven by configuration to ensure reproducibility.

- `data/`  
  Local data directories. Raw data is gitignored. Only derived or intermediate artifacts should be stored here.

- `outputs/`  
  Generated results such as extracted scenarios, figures, and tables. Outputs are written to stable `Step NN` paths and overwritten on rerun.

- `tests/`  
  Unit and integration tests, primarily focused on indicator calculations and detection logic.

---

## Running the code

### Environment setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### Path configuration

All scripts now resolve default paths from `configs/paths.json`.

To set machine-specific paths without committing them, create `configs/paths.local.json`:

```json
{
  "paths": {
    "dataset_root": "/absolute/path/to/highD/data",
    "outputs_root": "outputs"
  }
}
```

Config precedence is:
1. Environment variables (highest priority)
2. `configs/paths.local.json`
3. `configs/paths.json` (default, committed)

Supported environment variables:
- `CUTIN_PATHS_FILE` (custom JSON config location)
- `CUTIN_DATASET_ROOT`
- `CUTIN_OUTPUTS_ROOT`
- `CUTIN_STEP14_CODES_CSV`

### Detection configuration (thesis-sensitive defaults)

Lane-change and cut-in defaults are resolved from `configs/detection.json`.
To override locally without committing, create `configs/detection.local.json`:

```json
{
  "detection": {
    "lane_change": {
      "min_stable_before_frames": 25,
      "min_stable_after_frames": 25,
      "ignore_lane_ids": [0]
    },
    "cutin": {
      "search_window_frames": 50,
      "start_offset_frames": 0,
      "max_relation_delay_frames": 15,
      "min_relation_frames": 10,
      "require_new_follower": true,
      "precheck_frames": 25,
      "no_neighbor_ids": [0, -1],
      "require_lane_match": true,
      "require_preceding_consistency": true
    }
  }
}
```

Detection config precedence is:
1. `CUTIN_DETECTION_CONFIG_FILE` (custom JSON config location)
2. `configs/detection.local.json`
3. `configs/detection.json`
4. Built-in defaults in code (used if config file is missing)

### Thesis pipeline configuration

Step-level thesis defaults are resolved from `configs/thesis.json`.
To override locally without committing, create `configs/thesis.local.json`.

Key groups include:
- `pipeline` (recording subsets, THW label threshold, CI settings)
- `risk_label` (risk/very-risk cutoffs)
- `step04`..`step18` (window sizes, threshold grids, model hyperparameters, SFC options)
- `indicators` (position reference, indicator numeric safeguards)
  For highD thesis runs, use `indicators.position_reference = "bbox_topleft"`.

Thesis config precedence is:
1. `CUTIN_THESIS_CONFIG_FILE` (custom JSON config location)
2. `configs/thesis.local.json`
3. `configs/thesis.json`
4. Built-in defaults in each script (used if config file is missing)

## Thesis alignment note

For proposal-vs-implementation scope transparency, see:
- `THESIS_SCOPE_DELTA.md`

---

## New Member Quick Start (Thesis Flow)

This section is the recommended onboarding flow for thesis users.
It starts from Python setup, then runs the core steps in order.

Scope for this guide:
- Included: Step 01 to Step 10, Step 14, Step 15A, Step 15B
- Skipped on purpose: Step 11 to Step 13B, and everything after Step 15B

### 1) Python setup (`.venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 2) Configure dataset path

Create `configs/paths.local.json`:

```json
{
  "paths": {
    "dataset_root": "/absolute/path/to/highD/data",
    "outputs_root": "outputs"
  }
}
```

Current project default dataset path (from `configs/paths.json`) is:
- `data/raw/highD-dataset-v1.0/data`

So if you keep the dataset inside this repository, place files like:
- `data/raw/highD-dataset-v1.0/data/01_tracks.csv`
- `data/raw/highD-dataset-v1.0/data/01_tracksMeta.csv`
- `data/raw/highD-dataset-v1.0/data/01_recordingMeta.csv`

Quick path check:

```bash
ls data/raw/highD-dataset-v1.0/data | head
```

### 3) Run thesis steps (recommended order)

```bash
.venv/bin/python scripts/step01_recording_report.py
.venv/bin/python scripts/step02_lane_change_report.py
.venv/bin/python scripts/step03_cutin_report.py
.venv/bin/python scripts/step04_risk_metrics_report.py
.venv/bin/python scripts/step05_visualize_top_cutins.py
.venv/bin/python scripts/step06_neighbor_reconstruction_report.py
.venv/bin/python scripts/step07_xy_lane_pipeline_report.py
.venv/bin/python scripts/step08_stage_features.py
.venv/bin/python scripts/step09_batch_stage_features.py
.venv/bin/python scripts/step10_risk_report.py
.venv/bin/python scripts/step14_sfc_binary_encode.py
.venv/bin/python scripts/step14_sfc_binary_report.py
.venv/bin/python scripts/step15a_sfc_mirror_normalize.py
.venv/bin/python scripts/step15b_sfc_weighted_stage_features.py --mode distance
.venv/bin/python scripts/step15b_sfc_weighted_stage_features.py --mode ttc
```

### 4) What each step does and where output goes

| Step | Purpose | Main output path | Output contains |
|---|---|---|---|
| Step 01 | Recording-level data quality and schema checks | `outputs/reports/Step 01` | Per-recording quality report and failure summary |
| Step 02 | Lane-change detection summary across recordings | `outputs/reports/Step 02` | Lane-change counts by recording and markdown summary |
| Step 03 | Cut-in detection across recordings | `outputs/reports/Step 03` | Cut-in counts, per-recording stats, and summary |
| Step 04 | Compute risk indicators (DHW/THW/TTC) for cut-ins | `outputs/reports/Step 04` | Per-event and per-recording risk metric tables |
| Step 05 | Visual sanity-check of top-risk cut-ins | `outputs/reports/Step 05` and `outputs/figures/Step 05` | Plots of selected high-risk scenarios |
| Step 06 | Validate geometry-based neighbor reconstruction | `outputs/reports/Step 06` | Reconstruction accuracy and cut-in agreement metrics |
| Step 07 | Validate XY-lane + reconstructed-neighbor pipeline | `outputs/reports/Step 07` | Accuracy and cut-in matching metrics by recording |
| Step 08 | Build stage-wise event features per recording | `outputs/reports/Step 08/recording_XX` | Stage features per event and per-recording details |
| Step 09 | Merge Step 08 outputs across all recordings | `outputs/reports/Step 09` | `cutin_stage_features_merged.csv` and recording summary |
| Step 10 | Apply risk labels and produce descriptive summaries | `outputs/reports/Step 10` | Risk summary by recording and pooled risk/non-risk table |
| Step 14 | Encode local occupancy as binary SFC codes | `outputs/reports/Step 14` | Long per-frame SFC code table (`code`, `code_hex`, stage, event) |
| Step 14 report | Decode/aggregate SFC occupancy patterns | `outputs/reports/Step 14` | Stage-wise risk/non-risk occupancy grids (CSV, optional PNG) |
| Step 15A | Mirror-normalize binary SFC codes to canonical direction | `outputs/reports/Step 15A` | Direction-normalized binary SFC code table |
| Step 15B | Build weighted SFC stage features (distance/TTC) | `outputs/reports/Step 15B` | Weighted SFC feature vectors per stage and event |

---

## Visualizer (after pipeline run)

Use visualizer with Step 15A canonical SFC codes when you want normalized SFC feature inspection:

```bash
.venv/bin/python visuaziler/main.py \
  --recording_id 03 \
  --sfc_codes_csv "outputs/reports/Step 15A/sfc_binary_codes_long_hilbert_mirrored.csv" \
  --sfc_codes_canonical true
```

Notes:
- `--recording_id` can be any available recording (for example `01` to `20` in your current local run).
- For strict matrix verification against highD raw neighbor IDs, prefer Step 14 codes with `--sfc_codes_canonical false` (default).
- If `sfc_codes_canonical` and code-table orientation are inconsistent, visualizer now auto-corrects mode and marks it with `mode=*` in the SFC panel.
