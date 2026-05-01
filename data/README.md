# Data Directory

Use this directory as a local mount point for datasets and derived data. Raw and processed data are intentionally ignored by git.

Expected highD layout for the default configuration:

```text
data/raw/highD-dataset-v1.0/data/
  01_tracks.csv
  01_tracksMeta.csv
  01_recordingMeta.csv
  ...
```

If your data lives elsewhere, create `configs/paths.local.json` and point `paths.dataset_root` to the dataset directory. Do not commit raw trajectory data or generated dataset extracts.

See `docs/DATA.md` for dataset access and publication notes.
