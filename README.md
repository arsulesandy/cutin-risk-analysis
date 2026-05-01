# Cut-in Risk Analysis

This repository contains the code developed as part of a Master’s thesis on **cut-in lane-change risk analysis** using naturalistic highway trajectory data.

The project focuses on detecting cut-in scenarios in a **clear and reproducible way** and analysing how different risk indicators behave during these manoeuvres. The work is intended for **offline analysis and research use**, not for real-time driving systems or controllers.

The initial implementation is based on the **highD dataset**, but the codebase is structured to allow additional trajectory datasets to be integrated later without changing the core analysis logic.

---

## Citation

Arsule, S., & Shinde, S. (2026). *A Minimal-Input Framework for Cut-In Detection and Pair-Specific Risk Analysis in Highway Trajectory Data*. Master's thesis in Software Engineering, Chalmers University of Technology and University of Gothenburg.

Repository citation metadata is provided in `CITATION.cff`.

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
- `step03`, `step06`..`step10`, `step14`, `step15a`, `step15d`, `step18` (thesis workflow settings)
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
- Core thesis pipeline phases: interaction mining, core reconstruction, safety quantification, context signatures, reference-label evaluation
- Optional support steps kept in the repository: Step 01, Step 02, Step 04, Step 05, Step 08

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

### 3) Run thesis phases (recommended order)

```bash
.venv/bin/python scripts/run_thesis_phases.py --list-phases
.venv/bin/python scripts/run_thesis_phases.py \
  --phases interaction-mining core-reconstruction safety-quantification
.venv/bin/python scripts/run_thesis_phases.py \
  --phases context-signatures reference-label-evaluation
PYTHON_BIN=.venv/bin/python ./scripts/run_final_thesis_pipeline.sh
```

### 4) What each phase does and where output goes

| Phase | Purpose | Underlying scripts | Main output path |
|---|---|---|---|
| Preliminary processing | Basic recording checks and lane-change summary | `step01`, `step02` | `outputs/reports/Step 01`, `outputs/reports/Step 02` |
| Interaction mining | Cut-in extraction and event statistics | `step03` | `outputs/reports/Step 03` |
| Core reconstruction | Neighbour and XY-lane reconstruction with agreement checks | `step06`, `step07` | `outputs/reports/Step 06`, `outputs/reports/Step 07` |
| Safety quantification | Stage features, pair-consistent indicators, and risk summaries | `step09`, `step10` | `outputs/reports/Step 09`, `outputs/reports/Step 10` |
| Context signatures | SFC encoding, mirroring, and archetype analysis | `step14`, `step14 report`, `step15A`, `step15D` | `outputs/reports/Step 14`, `outputs/reports/Step 15A`, `outputs/reports/Step 15D` |
| Reference-label evaluation | Confidence intervals over reconstruction agreement | `step18` | `outputs/reports/Step 18` |

The low-level step scripts remain in `scripts/` as implementation detail, but the recommended entrypoint is now the phase runner.

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

## Thesis document assets

The thesis source now lives in `latex/`.
To refresh generated figures used by the document from the current `outputs/` tree:

```bash
.venv/bin/python scripts/sync_latex_assets.py
```

This copies only the figure files referenced by `latex/main.tex` that also exist under `outputs/`.
