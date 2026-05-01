# Data Access And Publication Notes

This repository does not include raw trajectory datasets. The thesis pipeline was written for local access to licensed datasets, primarily highD. Users must obtain datasets directly from the original providers and comply with their license terms.

## highD

Default expected layout:

```text
data/raw/highD-dataset-v1.0/data/
  01_tracks.csv
  01_tracksMeta.csv
  01_recordingMeta.csv
  02_tracks.csv
  ...
```

The default path is configured in `configs/paths.json`:

```json
{
  "paths": {
    "dataset_root": "data/raw/highD-dataset-v1.0/data"
  }
}
```

For local machines, prefer `configs/paths.local.json`:

```json
{
  "paths": {
    "dataset_root": "/absolute/path/to/highD/data",
    "outputs_root": "outputs"
  }
}
```

## exiD And NGSIM

The repository contains exploratory/support scripts for exiD and NGSIM-style feasibility checks. These datasets are also not redistributed here. Configure local paths before running those scripts.

## What Should Not Be Committed

- Raw dataset files
- Derived per-vehicle or per-event extracts containing licensed data
- Generated output folders under `outputs/`
- Local configuration files such as `configs/paths.local.json`
- Third-party literature PDFs
- Thesis build artifacts such as local LaTeX source/output folders

## What Can Be Published

- Source code
- Configuration defaults
- Reproducibility instructions
- Aggregated summary tables or figures only when they comply with the source dataset license
