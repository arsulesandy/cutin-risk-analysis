# Thesis Scope Delta (Proposal vs Implemented Pipeline)

This file documents explicit scope differences between the accepted proposal and the implemented thesis pipeline, so defense claims stay aligned with reproducible code outputs.

## Implemented in final pipeline

- Rule-based lane-change and cut-in extraction (highD).
- Surrogate safety indicators used in thesis steps:
  - DHW (distance headway)
  - THW (time headway)
  - TTC (time-to-collision)
- Stage-wise feature engineering (`intention`, `decision`, `execution`, `recovery`).
- Minimal-input neighbour and XY-lane reconstruction audits.
- Binary SFC encoding, canonical mirroring, and reversible occupancy verification.
- SFC archetype analysis with execution-stage risk prevalence comparison.
- Bootstrap confidence intervals for the Step 07 reconstruction metrics.

## Not implemented as thesis output metrics

- Brake-demand style indicator evaluation in the final reporting pipeline.
- Safe-gap indicator evaluation in the final reporting pipeline.

Note: placeholder APIs exist under `src/cutin_risk/indicators/`, but these are not part of the final thesis evaluation scripts.

## Critical geometry assumption (highD)

- highD `x` is treated as bbox top-left; thesis runs use `indicators.position_reference = "bbox_topleft"`.
- Geometry consistency is handled directly in the retained reconstruction and indicator steps rather than by a separate audit branch.

## Dataset coverage in current reproducible defaults

- `configs/thesis.json` default pipeline subset:
  - Step 09 batch recordings: `all`
- The retained thesis pipeline is aligned to the reported all-recordings evaluation over highD 01--60.
