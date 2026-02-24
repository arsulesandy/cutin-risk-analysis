# Step 04 Surrogate Risk Detailed Statistics

- Generated at: 2026-02-24T10:12:40
- Dataset root: `/Users/sandeep.arsule/IdeaProjects/cutin-risk-analysis/data/raw/highD-dataset-v1.0/data`

## Configuration
- Indicator `min_speed`: 0.1
- Indicator `closing_speed_epsilon`: 1e-06
- Event window `pre_frames`: 50
- Event window `post_frames`: 75
- Event minima are computed on relation frames only (`relation_start_frame..relation_end_frame`).
- Thesis-facing minima are clipped to non-negative for DHW/THW.
- Raw window minima are still exported as diagnostic columns (`*_raw_window`).
- Validation `sample_n`: 20000
- Validation `random_state`: 7

## Dataset-wide summary
- Recordings discovered: 20
- Recordings processed: 20
- Recordings failed: 0
- Validation skipped: 0
- Total lane changes: 3174
- Total cut-ins: 2563
- Events with computed metrics: 2563
- Chosen model `center`: 0
- Chosen model `rear`: 0
- Chosen model `bbox_topleft`: 20
- Metrics coverage over cut-ins: 100.00%

### Risk metric summaries (event-level minima)
- DHW(min) [m] count: 2563
- DHW(min) [m] min: 3.790
- DHW(min) [m] p25: 20.940
- DHW(min) [m] median: 32.490
- DHW(min) [m] mean: 53.897
- DHW(min) [m] p75: 61.610
- DHW(min) [m] max: 371.600
- THW(min) [s] (finite only) count: 2563
- THW(min) [s] (finite only) min: 0.111
- THW(min) [s] (finite only) p25: 0.777
- THW(min) [s] (finite only) median: 1.200
- THW(min) [s] (finite only) mean: 1.737
- THW(min) [s] (finite only) p75: 2.075
- THW(min) [s] (finite only) max: 12.043
- TTC(min) [s] (finite only) count: 865
- TTC(min) [s] (finite only) min: 2.162
- TTC(min) [s] (finite only) p25: 11.471
- TTC(min) [s] (finite only) median: 21.042
- TTC(min) [s] (finite only) mean: 78.026
- TTC(min) [s] (finite only) p75: 43.906
- TTC(min) [s] (finite only) max: 16509.000

## Top events by TTC(min)
| Recording | Cutter | Follower | Lane | t_rel_start (s) | DHW(min) | THW(min) | TTC(min) |
|-----------|-------:|---------:|------|----------------:|---------:|---------:|---------:|
| 17        |    491 |      495 | 6->5 |         451.360 |    7.330 |    0.187 |    2.162 |
| 12        |   2141 |     2143 | 6->7 |         727.200 |    5.270 |    0.178 |    2.896 |
| 16        |    623 |      630 | 2->3 |         602.240 |   35.970 |    0.960 |    2.927 |
| 13        |   2179 |     2189 | 3->4 |         796.520 |   12.880 |    0.415 |    2.988 |
| 07        |     71 |       74 | 2->3 |          47.080 |   15.690 |    0.432 |    3.041 |
| 19        |    846 |      851 | 2->3 |         815.200 |   19.170 |    0.661 |    3.112 |
| 18        |    210 |      216 | 2->3 |         195.560 |   24.860 |    0.732 |    3.131 |
| 12        |   1317 |     1322 | 3->2 |         430.320 |   10.440 |    0.545 |    3.539 |
| 03        |    618 |      623 | 2->3 |         688.240 |   21.960 |    0.501 |    3.685 |
| 06        |   1040 |     1044 | 8->7 |         883.880 |   29.670 |    0.805 |    3.789 |
| 08        |    274 |      280 | 8->7 |         176.720 |   38.000 |    0.881 |    3.854 |
| 05        |    846 |      850 | 3->4 |         781.000 |   25.530 |    0.675 |    3.863 |
| 11        |   1436 |     1443 | 2->3 |         501.960 |   10.650 |    0.398 |    4.004 |
| 09        |   1251 |     1257 | 3->4 |         950.480 |   20.010 |    0.502 |    4.250 |
| 17        |     26 |       34 | 2->3 |           4.480 |   53.330 |    1.476 |    4.279 |
| 19        |    109 |      111 | 6->5 |         112.040 |   14.610 |    0.424 |    4.415 |
| 14        |   2515 |     2514 | 3->4 |         990.720 |    4.250 |    0.157 |    4.429 |
| 02        |    986 |      997 | 2->3 |         908.320 |   21.550 |    0.745 |    4.475 |
| 09        |    654 |      658 | 2->3 |         510.440 |   70.920 |    1.648 |    4.647 |
| 16        |    733 |      735 | 6->5 |         706.160 |   34.530 |    0.973 |    4.737 |

## Per-recording summary
| Recording | Vehicles | Lane changes | Cut-ins | Metric events | TTC(min) | Model        |
|-----------|---------:|-------------:|--------:|--------------:|---------:|--------------|
| 01        |     1047 |          118 |     103 |           103 |    4.993 | bbox_topleft |
| 02        |     1113 |          107 |      83 |            83 |    4.475 | bbox_topleft |
| 03        |      914 |           99 |      78 |            78 |    3.685 | bbox_topleft |
| 04        |     1163 |          216 |     153 |           153 |    5.719 | bbox_topleft |
| 05        |     1216 |          216 |     160 |           160 |    3.863 | bbox_topleft |
| 06        |     1368 |          261 |     189 |           189 |    3.789 | bbox_topleft |
| 07        |      855 |          116 |      99 |            99 |    3.041 | bbox_topleft |
| 08        |     1620 |          233 |     167 |           167 |    3.854 | bbox_topleft |
| 09        |     1420 |          210 |     156 |           156 |    4.250 | bbox_topleft |
| 10        |      856 |          116 |      93 |            93 |    5.301 | bbox_topleft |
| 11        |     1776 |          131 |     126 |           126 |    4.004 | bbox_topleft |
| 12        |     2728 |          192 |     183 |           183 |    2.896 | bbox_topleft |
| 13        |     2949 |          271 |     253 |           253 |    2.988 | bbox_topleft |
| 14        |     2844 |          236 |     211 |           211 |    4.429 | bbox_topleft |
| 15        |      922 |           97 |      82 |            82 |    6.674 | bbox_topleft |
| 16        |      998 |          126 |     103 |           103 |    2.927 | bbox_topleft |
| 17        |      977 |          117 |      90 |            90 |    2.162 | bbox_topleft |
| 18        |      607 |           58 |      43 |            43 |    3.131 | bbox_topleft |
| 19        |      999 |          131 |      99 |            99 |    3.112 | bbox_topleft |
| 20        |     1199 |          123 |      92 |            92 |    4.989 | bbox_topleft |

## Failed recordings
- None

## Validation skipped recordings
- None
