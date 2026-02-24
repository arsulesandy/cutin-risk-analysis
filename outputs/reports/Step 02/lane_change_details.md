# Step 02 Lane Change Detailed Statistics

- Generated at: 2026-02-24T10:05:12
- Dataset root: `/Users/sandeep.arsule/IdeaProjects/cutin-risk-analysis/data/raw/highD-dataset-v1.0/data`

## Detection configuration
- `min_stable_before_frames`: 25
- `min_stable_after_frames`: 25
- `ignore_lane_ids`: [0]

## Dataset-wide summary
- Recordings discovered: 20
- Recordings processed: 20
- Recordings failed: 0
- Total lane change events: 3174
- Mean lane changes per processed recording: 158.70

### Top transitions (dataset-wide)
| From lane | To lane | Count |
|-----------|---------|------:|
| 3         | 2       |   526 |
| 2         | 3       |   471 |
| 4         | 3       |   353 |
| 6         | 7       |   333 |
| 3         | 4       |   326 |
| 7         | 6       |   313 |
| 5         | 6       |   293 |
| 7         | 8       |   200 |
| 6         | 5       |   199 |
| 8         | 7       |   160 |

### Lane-change duration stats (dataset-wide)
- Events included: 3174 lane changes
- Duration (frames) min: 25
- Duration (frames) p25: 90
- Duration (frames) median: 151
- Duration (frames) mean: 157
- Duration (frames) p75: 218
- Duration (frames) max: 538
- Duration (seconds) min: 0.960
- Duration (seconds) p25: 3.560
- Duration (seconds) median: 6.000
- Duration (seconds) mean: 6.259
- Duration (seconds) p75: 8.680
- Duration (seconds) max: 21.480

## Per-recording counts
| Recording | Vehicles | Lane changes |
|-----------|---------:|-------------:|
| 01        |     1047 |          118 |
| 02        |     1113 |          107 |
| 03        |      914 |           99 |
| 04        |     1163 |          216 |
| 05        |     1216 |          216 |
| 06        |     1368 |          261 |
| 07        |      855 |          116 |
| 08        |     1620 |          233 |
| 09        |     1420 |          210 |
| 10        |      856 |          116 |
| 11        |     1776 |          131 |
| 12        |     2728 |          192 |
| 13        |     2949 |          271 |
| 14        |     2844 |          236 |
| 15        |      922 |           97 |
| 16        |      998 |          126 |
| 17        |      977 |          117 |
| 18        |      607 |           58 |
| 19        |      999 |          131 |
| 20        |     1199 |          123 |

## Failed recordings
- None
