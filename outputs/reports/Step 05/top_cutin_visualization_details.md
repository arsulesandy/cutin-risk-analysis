# Step 05 Top Cut-in Visualization Report

- Generated at: 2026-02-24T10:28:07
- Dataset root: `/Users/sandeep.arsule/IdeaProjects/cutin-risk-analysis/data/raw/highD-dataset-v1.0/data`
- Step 04 per-event source: `/Users/sandeep.arsule/IdeaProjects/cutin-risk-analysis/outputs/reports/Step 04/risk_metrics_events.csv`
- Step 04 per-recording source: `/Users/sandeep.arsule/IdeaProjects/cutin-risk-analysis/outputs/reports/Step 04/risk_metrics_by_recording.csv`

## Configuration
- Indicator model `position_reference`: bbox_topleft
- Indicator `min_speed`: 0.1
- Indicator `closing_speed_epsilon`: 1e-06
- Event window `pre_seconds`: 2.0
- Event window `post_seconds`: 3.0
- Plot `ttc_clip`: 60.0
- Global top-k visualized events: 3

## Dataset-wide summary
- Recordings discovered (from Step 04): 20
- Total lane changes (from Step 04): 3174
- Total cut-ins (from Step 04): 2563
- Events with computed metrics (from Step 04): 2563
- Metrics coverage over cut-ins: 100.00%

## Top events by TTC(min)
| Rank | Recording | Cutter | Follower | Lane | t_rel_start (s) | DHW(min) | THW(min) | TTC(min) |
|-----:|-----------|-------:|---------:|------|----------------:|---------:|---------:|---------:|
|    1 | 17        |    491 |      495 | 6->5 |          451.36 |     7.33 |     0.19 |     2.16 |
|    2 | 12        |   2141 |     2143 | 6->7 |          727.20 |     5.27 |     0.18 |     2.90 |
|    3 | 16        |    623 |      630 | 2->3 |          602.24 |    35.97 |     0.96 |     2.93 |
|    4 | 13        |   2179 |     2189 | 3->4 |          796.52 |    12.88 |     0.42 |     2.99 |
|    5 | 07        |     71 |       74 | 2->3 |           47.08 |    15.69 |     0.43 |     3.04 |
|    6 | 19        |    846 |      851 | 2->3 |          815.20 |    19.17 |     0.66 |     3.11 |
|    7 | 18        |    210 |      216 | 2->3 |          195.56 |    24.86 |     0.73 |     3.13 |
|    8 | 12        |   1317 |     1322 | 3->2 |          430.32 |    10.44 |     0.54 |     3.54 |
|    9 | 03        |    618 |      623 | 2->3 |          688.24 |    21.96 |     0.50 |     3.68 |
|   10 | 06        |   1040 |     1044 | 8->7 |          883.88 |    29.67 |     0.80 |     3.79 |
|   11 | 08        |    274 |      280 | 8->7 |          176.72 |    38.00 |     0.88 |     3.85 |
|   12 | 05        |    846 |      850 | 3->4 |          781.00 |    25.53 |     0.68 |     3.86 |
|   13 | 11        |   1436 |     1443 | 2->3 |          501.96 |    10.65 |     0.40 |     4.00 |
|   14 | 09        |   1251 |     1257 | 3->4 |          950.48 |    20.01 |     0.50 |     4.25 |
|   15 | 17        |     26 |       34 | 2->3 |            4.48 |    53.33 |     1.48 |     4.28 |
|   16 | 19        |    109 |      111 | 6->5 |          112.04 |    14.61 |     0.42 |     4.41 |
|   17 | 14        |   2515 |     2514 | 3->4 |          990.72 |     4.25 |     0.16 |     4.43 |
|   18 | 02        |    986 |      997 | 2->3 |          908.32 |    21.55 |     0.75 |     4.48 |
|   19 | 09        |    654 |      658 | 2->3 |          510.44 |    70.92 |     1.65 |     4.65 |
|   20 | 16        |    733 |      735 | 6->5 |          706.16 |    34.53 |     0.97 |     4.74 |

## Plot reconstruction issues
- None
