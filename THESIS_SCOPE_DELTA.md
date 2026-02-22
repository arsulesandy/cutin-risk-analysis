# Thesis Scope Delta (Proposal vs Implemented Pipeline)

This file documents explicit scope differences between the accepted proposal and the implemented thesis pipeline, so defense claims stay aligned with reproducible code outputs.

## Implemented in final pipeline

- Rule-based lane-change and cut-in extraction (highD).
- Surrogate safety indicators used in thesis steps:
  - DHW (distance headway)
  - THW (time headway)
  - TTC (time-to-collision)
- Stage-wise feature engineering (`intention`, `decision`, `execution`, `recovery`).
- Warning-rule and logistic baselines with leave-one-recording-out evaluation.
- SFC-based encodings and prediction experiments.
- Split-integrity and metrics-with-CI audits.

## Not implemented as thesis output metrics

- Brake-demand style indicator evaluation in the final reporting pipeline.
- Safe-gap indicator evaluation in the final reporting pipeline.

Note: placeholder APIs exist under `src/cutin_risk/indicators/`, but these are not part of the final thesis evaluation scripts.

## Critical geometry assumption (highD)

- highD `x` is treated as bbox top-left; thesis runs use `indicators.position_reference = "bbox_topleft"`.
- Geometry/model consistency is audited by:
  - `scripts/step17_geometry_audit.py`
  - output: `outputs/reports/Step 17/geometry_audit.md`

## Dataset coverage in current reproducible defaults

- `configs/thesis.json` default pipeline subset:
  - Step 09 batch recordings: `1-10`
  - Step 16 recordings: `01,02,03,04,05`
- Claims in thesis text/results should state this exact evaluated subset unless rerun on all 60 recordings.

