# Cut-in Risk Analysis

Research code for the thesis **"A Minimal-Input Framework for Cut-In Detection and Pair-Specific Risk Analysis in Highway Trajectory Data"** by Sandeep Arsule and Shradha Shinde.

The repository implements a reproducible offline pipeline for detecting cut-in lane-change interactions in highway trajectory data, reconstructing pair-specific vehicle context, and comparing surrogate safety indicators over maneuver stages. It is intended as a research artifact, not as a real-time driving system or vehicle controller.

## What This Repository Contains

- Dataset adapters and preprocessing utilities for trajectory datasets, with highD as the primary thesis dataset.
- Rule-based lane-change and cut-in detection.
- Neighbor and lane reconstruction checks used to validate pair-specific context.
- Surrogate safety indicators including distance headway, time headway, time-to-collision, safe gap, and braking-demand style measures.
- Stage-level feature extraction, risk summaries, space-filling-curve encodings, and archetype analysis.
- Scripts for regenerating reproducibility reports, tables, and figures.
- A lightweight visual inspection tool under `visuaziler/` (historical directory name).

Raw datasets, generated outputs, and third-party literature PDFs are intentionally not included.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/cutin_risk/` | Python package with dataset readers, detection logic, indicators, analysis, encoding, IO, and visualization helpers. |
| `scripts/` | Reproducible thesis pipeline steps and phase runners. |
| `configs/` | Committed default configuration. Use matching `*.local.json` files for machine-specific overrides. |
| `tests/` | Unit tests for geometry, indicator behavior, and configuration defaults. |
| `data/` | Local dataset mount point. Raw and derived data are gitignored. |
| `outputs/` | Generated reports, figures, tables, and manifests. Gitignored. |
| `docs/` | Public-release notes for data access and reproducibility. |

## Requirements

- Python 3.10 or newer
- Access to the relevant trajectory dataset files, primarily highD for the thesis pipeline
- A Unix-like shell for the provided runner scripts

Install the package in editable mode with development tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the checks:

```bash
make test
make lint
```

## Data Setup

The thesis data cannot be redistributed with this repository. Obtain dataset access from the original providers and store the files locally.

By default, highD is expected at:

```text
data/raw/highD-dataset-v1.0/data
```

For a machine-specific path, create `configs/paths.local.json`:

```json
{
  "paths": {
    "dataset_root": "/absolute/path/to/highD/data",
    "outputs_root": "outputs"
  }
}
```

Path precedence is:

1. Environment variables
2. `configs/paths.local.json`
3. `configs/paths.json`

Supported environment variables:

- `CUTIN_PATHS_FILE`
- `CUTIN_DATASET_ROOT`
- `CUTIN_OUTPUTS_ROOT`
- `CUTIN_STEP14_CODES_CSV`

See [docs/DATA.md](docs/DATA.md) for expected file names and data-publication notes.

## Running The Thesis Pipeline

The recommended entry point is the phase runner:

```bash
python scripts/run_thesis_phases.py --list-phases
```

Run the core phases in order:

```bash
python scripts/run_thesis_phases.py \
  --phases interaction-mining core-reconstruction safety-quantification

python scripts/run_thesis_phases.py \
  --phases context-signatures reference-label-evaluation
```

For the approved thesis run, use the wrapper:

```bash
PYTHON_BIN=.venv/bin/python ./scripts/run_final_thesis_pipeline.sh
```

Key outputs are written under `outputs/reports/`, including:

- `outputs/reports/Step 03/cutin_details.md`
- `outputs/reports/Step 07/xy_lane_pipeline_metrics_summary.csv`
- `outputs/reports/Step 10/risk_summary_by_recording.csv`
- `outputs/reports/Step 15D/archetype_report.md`
- `outputs/reports/final/metrics_with_ci.md`
- `outputs/reports/final/run_manifest.json`

For a fuller reproduction guide, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Configuration

Committed defaults live in:

- `configs/paths.json`
- `configs/detection.json`
- `configs/thesis.json`

Use local overrides for private paths or experiment variants:

- `configs/paths.local.json`
- `configs/detection.local.json`
- `configs/thesis.local.json`

These local files are ignored by git. The thesis-sensitive defaults are kept in the committed configuration files so that public reruns use the same assumptions unless explicitly overridden.

## Visual Inspection

To inspect normalized SFC features after Step 15A:

```bash
python visuaziler/main.py \
  --recording_id 03 \
  --sfc_codes_csv "outputs/reports/Step 15A/sfc_binary_codes_long_hilbert_mirrored.csv" \
  --sfc_codes_canonical true
```

For strict checks against raw highD neighbor IDs, use Step 14 codes and keep `--sfc_codes_canonical false`.

## Scope And Limitations

- The pipeline is designed for offline research and thesis reproducibility.
- The default thesis configuration targets highD-style trajectory data.
- exiD and NGSIM-related scripts are exploratory/supporting utilities unless explicitly documented by a thesis phase.
- The rule-based definitions are intentionally transparent; they are not claimed as universal cut-in or risk standards.
- Results depend on the dataset license, local data version, and configuration files used for a run.

## Citation

If you use this repository, cite:

```text
Arsule, S., & Shinde, S. (2026). A Minimal-Input Framework for Cut-In Detection
and Pair-Specific Risk Analysis in Highway Trajectory Data. Master's thesis in
Software Engineering, Chalmers University of Technology and University of Gothenburg.
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

This repository's original code and documentation are released under the MIT License; see [LICENSE](LICENSE).

Dataset files, generated outputs derived from licensed datasets, thesis build artifacts, and third-party articles or PDFs are not covered by this license and are not redistributed here.
