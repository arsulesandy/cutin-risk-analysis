# Step 03 Cut-in Detailed Statistics

- Generated at: 2026-02-24T10:07:03
- Dataset root: `/Users/sandeep.arsule/IdeaProjects/cutin-risk-analysis/data/raw/highD-dataset-v1.0/data`

## Detection configuration
- Lane-change `min_stable_before_frames`: 25
- Lane-change `min_stable_after_frames`: 25
- Cut-in `search_window_frames`: 50
- Cut-in `min_relation_frames`: 10
- Cut-in `max_relation_delay_frames`: 15

## Dataset-wide summary
- Recordings discovered: 20
- Recordings processed: 20
- Recordings failed: 0
- Total lane changes: 3174
- Total cut-ins: 2563
- Overall cut-in ratio: 80.75%

### Top cut-in transitions (dataset-wide)
| From lane | To lane | Count |
|-----------|---------|------:|
| 3         | 2       |   501 |
| 4         | 3       |   334 |
| 2         | 3       |   318 |
| 6         | 7       |   318 |
| 5         | 6       |   284 |
| 7         | 8       |   193 |
| 3         | 4       |   192 |
| 7         | 6       |   180 |
| 8         | 7       |   124 |
| 6         | 5       |   119 |

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

### Cut-in relation duration stats (dataset-wide)
- Events included: 2563 cut-ins
- Duration (frames) min: 10
- Duration (frames) p25: 50
- Duration (frames) median: 50
- Duration (frames) mean: 48
- Duration (frames) p75: 50
- Duration (frames) max: 50
- Duration (seconds) min: 0.360
- Duration (seconds) p25: 1.960
- Duration (seconds) median: 1.960
- Duration (seconds) mean: 1.872
- Duration (seconds) p75: 1.960
- Duration (seconds) max: 1.960

## Per-recording counts
| Recording | Vehicles | Lane changes | Cut-ins | Cut-in ratio |
|-----------|---------:|-------------:|--------:|-------------:|
| 01        |     1047 |          118 |     103 |       87.29% |
| 02        |     1113 |          107 |      83 |       77.57% |
| 03        |      914 |           99 |      78 |       78.79% |
| 04        |     1163 |          216 |     153 |       70.83% |
| 05        |     1216 |          216 |     160 |       74.07% |
| 06        |     1368 |          261 |     189 |       72.41% |
| 07        |      855 |          116 |      99 |       85.34% |
| 08        |     1620 |          233 |     167 |       71.67% |
| 09        |     1420 |          210 |     156 |       74.29% |
| 10        |      856 |          116 |      93 |       80.17% |
| 11        |     1776 |          131 |     126 |       96.18% |
| 12        |     2728 |          192 |     183 |       95.31% |
| 13        |     2949 |          271 |     253 |       93.36% |
| 14        |     2844 |          236 |     211 |       89.41% |
| 15        |      922 |           97 |      82 |       84.54% |
| 16        |      998 |          126 |     103 |       81.75% |
| 17        |      977 |          117 |      90 |       76.92% |
| 18        |      607 |           58 |      43 |       74.14% |
| 19        |      999 |          131 |      99 |       75.57% |
| 20        |     1199 |          123 |      92 |       74.80% |

## Failed recordings
- None
