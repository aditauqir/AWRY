# AWRY Report Bundle

This bundle aggregates the existing AWRY artifacts for paper writing.
It is generated deterministically from files already present in the repository.

**Read/write scope:** reads existing artifacts and raw cached data; writes only this report under `artifacts/report_bundle/`.

## Feature Set Summary

| file | feature_set | alpha | threshold | composite_auroc | composite_brier | composite_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| artifacts/models/walk_forward_summary_baseline.json | baseline | 1 | 0.23254 | 0.887248 | 0.0465556 | 0.571429 |
| artifacts/models/walk_forward_summary_full.json | full | 0.6 | 0.323802 | 0.949519 | 0.0341841 | 0.666667 |
| artifacts/models/walk_forward_summary_full_news.json | full_news | 0.25 | 0.157946 | 0.768429 | 0.0586605 | 0.294118 |
| artifacts/models/walk_forward_summary_stress.json | stress | 0.8 | 0.421087 | 0.947459 | 0.0337766 | 0.711111 |

## Fitted Parameters: Threshold, Alpha, Stacker

| source | parameter | value |
| --- | --- | --- |
| thresholds.json | class_imbalance | 0.0823529 |
| thresholds.json | f1 | 0.68 |
| thresholds.json | precision | 0.772727 |
| thresholds.json | recall | 0.607143 |
| thresholds.json | threshold | 0.23254 |
| composite_baseline_metrics.json | alpha | 1 |
| composite_full_metrics.json | alpha | 0.6 |
| composite_full_news_metrics.json | alpha | 0.25 |
| composite_stress_metrics.json | alpha | 0.8 |
| forecast3_baseline_metrics.json | meta_intercept | -2.23816 |
| forecast3_baseline_metrics.json | meta_logit | 0.2133 |
| forecast3_baseline_metrics.json | meta_rf | -0.210081 |
| forecast3_baseline_metrics.json | fixed_weights | [0.09, 0.91] |
| forecast3_full_metrics.json | meta_intercept | -4.24706 |
| forecast3_full_metrics.json | meta_logit | 0.0539938 |
| forecast3_full_metrics.json | meta_rf | 0.952542 |
| forecast3_full_metrics.json | meta_xgb | -1.15431 |
| forecast3_full_metrics.json | fixed_weights | [0.15, 0.85] |
| forecast3_full_news_metrics.json | meta_intercept | -1.56693 |
| forecast3_full_news_metrics.json | meta_logit | 0.158522 |
| forecast3_full_news_metrics.json | meta_rf | -0.0791378 |
| forecast3_full_news_metrics.json | fixed_weights | [0.15, 0.85] |
| forecast3_stress_metrics.json | meta_intercept | -4.51763 |
| forecast3_stress_metrics.json | meta_logit | 0.0154606 |
| forecast3_stress_metrics.json | meta_rf | 1.03891 |
| forecast3_stress_metrics.json | meta_xgb | -1.21946 |
| forecast3_stress_metrics.json | fixed_weights | [0.1, 0.9] |
| nowcast_baseline_metrics.json | meta_intercept | -2.02165 |
| nowcast_baseline_metrics.json | meta_logit | 0.358218 |
| nowcast_baseline_metrics.json | meta_rf | -0.315354 |
| nowcast_baseline_metrics.json | fixed_weights | [0.18, 0.8200000000000001] |
| nowcast_full_metrics.json | meta_intercept | -3.6343 |
| nowcast_full_metrics.json | meta_logit | 0.239581 |
| nowcast_full_metrics.json | meta_rf | 0.85483 |
| nowcast_full_metrics.json | meta_xgb | -1.19817 |
| nowcast_full_metrics.json | fixed_weights | [0.11, 0.89] |
| nowcast_full_news_metrics.json | meta_intercept | -1.0738 |
| nowcast_full_news_metrics.json | meta_logit | 0.18899 |
| nowcast_full_news_metrics.json | meta_rf | -0.0328117 |
| nowcast_full_news_metrics.json | fixed_weights | [0.12, 0.88] |
| nowcast_stress_metrics.json | meta_intercept | -4.08299 |
| nowcast_stress_metrics.json | meta_logit | 0.200684 |
| nowcast_stress_metrics.json | meta_rf | 0.988929 |
| nowcast_stress_metrics.json | meta_xgb | -1.3237 |
| nowcast_stress_metrics.json | fixed_weights | [0.1, 0.9] |

- File not found: artifacts/models/alpha_tuned.json

## OOS Metrics

### Composite and Horizon Summaries

| feature_set | metric_block | auroc | brier | f1 |
| --- | --- | --- | --- | --- |
| baseline | now_metrics | 0.887248 | 0.0465556 | 0.571429 |
| baseline | forecast_metrics | 0.737179 | 0.0677669 | 0.176471 |
| baseline | composite_metrics | 0.887248 | 0.0465556 | 0.571429 |
| baseline | in_sample_metrics | 0.969723 | 0.0385772 | 0.536585 |
| full | now_metrics | 0.950206 | 0.0353194 | 0.695652 |
| full | forecast_metrics | 0.906593 | 0.0450319 | 0.55814 |
| full | composite_metrics | 0.949519 | 0.0341841 | 0.666667 |
| full | in_sample_metrics | 0.984252 | 0.0351612 | 0.4 |
| full_news | now_metrics | 0.69185 | 0.0607919 | 0.388889 |
| full_news | forecast_metrics | 0.69288 | 0.0711 | 0.117647 |
| full_news | composite_metrics | 0.768429 | 0.0586605 | 0.294118 |
| full_news | in_sample_metrics | 1 | 0.0198113 | 0.782609 |
| stress | now_metrics | 0.948489 | 0.0340363 | 0.711111 |
| stress | forecast_metrics | 0.893201 | 0.0452905 | 0.604651 |
| stress | composite_metrics | 0.947459 | 0.0337766 | 0.711111 |
| stress | in_sample_metrics | 0.961192 | 0.0454447 | 0.133333 |

### Per-Fold Metrics When Available

_No per-fold metric rows found in metrics JSON artifacts._

## Ablation

| ablation | feature_set | auroc | brier | f1 | lead_2001 | lead_2008 | lead_2020 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | baseline | 0.949061 | 0.0359592 | 0.723404 | 23 | 2 |  |
| B | stress | 0.947459 | 0.0337766 | 0.711111 | 0 | 3 |  |
| C | full | 0.949519 | 0.0341841 | 0.666667 | 1 | 3 | 1 |
| D | full_news | 0.949977 | 0.034121 | 0.666667 | 7 | 2 |  |

## Diagnostics

### Fold positive counts

| fold | train_start | train_end | test_start | test_end | train_n | train_pos | train_pos_rate | test_n | test_pos | test_pos_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1997-10-31 | 2002-06-30 | 2002-10-31 | 2007-05-31 | 57 | 8 | 0.140351 | 56 | 0 | 0 |
| 1 | 1997-10-31 | 2007-02-28 | 2007-06-30 | 2012-01-31 | 113 | 8 | 0.0707965 | 56 | 18 | 0.321429 |
| 2 | 1997-10-31 | 2011-10-31 | 2012-02-29 | 2016-09-30 | 169 | 26 | 0.153846 | 56 | 0 | 0 |
| 3 | 1997-10-31 | 2016-06-30 | 2016-10-31 | 2021-05-31 | 225 | 26 | 0.115556 | 56 | 2 | 0.0357143 |
| 4 | 1997-10-31 | 2021-02-28 | 2021-06-30 | 2026-01-31 | 281 | 28 | 0.0996441 | 56 | 0 | 0 |

### Pre-recession probability peaks

| scenario | r0 | window_start | window_end | peak_pre_r0 | peak_month | peak_to_threshold_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 2001_dotcom | 2001-03-31 | 2000-03-31 | 2001-02-28 | 0.0583635 | 2000-03-31 | -0.174137 |
| 2008_gfc | 2007-12-31 | 2006-12-31 | 2007-11-30 | 0.128798 | 2007-05-31 | -0.103702 |
| 2020_covid | 2020-02-29 | 2019-02-28 | 2019-12-31 | 0.141869 | 2019-02-28 | -0.0906312 |

### Threshold sweep

| scenario | 0.1 | 0.15 | 0.2 | 0.2325 | 0.3 |
| --- | --- | --- | --- | --- | --- |
| 2001_dotcom |  |  |  |  |  |
| 2008_gfc | -21 |  |  |  |  |
| 2020_covid | -12 |  |  |  |  |

### Threshold false positives

| threshold | tp | fp | fn | tn | precision | recall | false_alarm_months |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 17 | 52 | 11 | 260 | 0.246377 | 0.607143 | 52 |
| 0.15 | 14 | 8 | 14 | 304 | 0.636364 | 0.5 | 8 |
| 0.2 | 11 | 3 | 17 | 309 | 0.785714 | 0.392857 | 3 |
| 0.2325 | 8 | 1 | 20 | 311 | 0.888889 | 0.285714 | 1 |
| 0.3 | 7 | 1 | 21 | 311 | 0.875 | 0.25 | 1 |

## Scenario Backtests From OOF Composite

| scenario | r0 | window_start | window_end | peak_pre_r0 | peak_month | rows |
| --- | --- | --- | --- | --- | --- | --- |
| 2001_dotcom | 2001-03-31 | 2000-03-31 | 2001-02-28 | 0.0682538 | 2000-03-31 | 12 |
| 2008_gfc | 2007-12-31 | 2006-12-31 | 2007-11-30 | 0.269116 | 2007-08-31 | 12 |
| 2020_covid | 2020-02-29 | 2019-02-28 | 2019-12-31 | 0.152932 | 2019-02-28 | 11 |

## ALFRED Vintage-vs-Revised Comparison

| scenario | as_of_date | p_awry_vintage | p_awry_revised | delta_pp | payems_delta_jobs |
| --- | --- | --- | --- | --- | --- |
| 2001_dotcom | 2001-02-28 | 0.0950545 | 0.161608 | 6.65533 | 659000 |
| 2008_gfc | 2007-11-30 | 0.173323 | 0.172001 | -0.132217 | -131000 |
| 2020_covid | 2020-01-31 | 0.0436807 | 0.0190506 | -2.46301 | -352000 |

LaTeX companion file found: `artifacts/figures/alfred_comparison.tex`.

## Current State: Latest OOF Row

| date | fold_idx | USREC | target_h0 | target_h3 | P_now | P_3m | P_AWRY |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-31 | 5 | 0 | 0 | 0 | 0.0339492 | 0.126174 | 0.0339492 |

## Recent Raw Feature Values

| series | latest_date | latest_value | observations |
| --- | --- | --- | --- |
| BAA10Y | 2026-04-17 | 1.72 | 10073 |
| BAMLH0A0HYM2 | 2026-04-17 | 2.83 | 785 |
| CFNAI | 2026-02-01 | -0.11 | 674 |
| CSUSHPINSA | 2026-01-01 | 326.612 | 469 |
| DCOILWTICO | 2026-04-13 | 100.72 | 10138 |
| DJIA | 2026-03-23 | 46208.5 | 2513 |
| FEDFUNDS | 2026-03-01 | 3.64 | 675 |
| HOUST | 2026-01-01 | 1487 | 673 |
| ICSA | 2026-04-11 | 207000 | 2937 |
| INDPRO | 2026-03-01 | 101.79 | 675 |
| NASDAQCOM | 2026-04-17 | 24468.5 | 13917 |
| NFCI | 2026-04-10 | -0.465 | 2884 |
| PAYEMS | 2026-03-01 | 158637 | 675 |
| PERMIT | 2026-01-01 | 1386 | 673 |
| RRSFS | 2026-03-01 | 227696 | 410 |
| SP500 | 2026-03-23 | 6581 | 2513 |
| T10Y3M | 2026-04-20 | 0.55 | 11076 |
| TEDRATE | 2022-01-21 | 0.09 | 8853 |
| UMCSENT | 2026-02-01 | 56.6 | 610 |
| UNRATE | 2026-03-01 | 4.3 | 674 |
| USEPUNEWSINDXM | 2026-03-01 | 382.222 | 675 |
| USREC | 2026-03-01 | 0 | 675 |
| VIXCLS | 2026-04-17 | 17.48 | 9167 |
| W875RX1 | 2026-02-01 | 16678.7 | 674 |

## Full Artifact File Inventory

| path | size_bytes | suffix |
| --- | --- | --- |
| artifacts/figures/ablation_comparison.png | 37024 | .png |
| artifacts/figures/alfred_comparison.csv | 355 | .csv |
| artifacts/figures/alfred_comparison.tex | 350 | .tex |
| artifacts/figures/alpha_curve.png | 55201 | .png |
| artifacts/figures/diagnostic_fold_counts.csv | 554 | .csv |
| artifacts/figures/diagnostic_peaks.csv | 370 | .csv |
| artifacts/figures/diagnostic_threshold_fp.csv | 307 | .csv |
| artifacts/figures/diagnostic_threshold_sweep.csv | 94 | .csv |
| artifacts/figures/reliability.png | 68660 | .png |
| artifacts/figures/threshold_pr_curve.png | 43553 | .png |
| artifacts/models/ablation_summary.json | 810 | .json |
| artifacts/models/calibration.json | 45 | .json |
| artifacts/models/composite_baseline_metrics.json | 142 | .json |
| artifacts/models/composite_full_metrics.json | 156 | .json |
| artifacts/models/composite_full_news_metrics.json | 144 | .json |
| artifacts/models/composite_stress_metrics.json | 141 | .json |
| artifacts/models/forecast3_baseline_metrics.json | 1048 | .json |
| artifacts/models/forecast3_full_metrics.json | 2850 | .json |
| artifacts/models/forecast3_full_news_metrics.json | 3136 | .json |
| artifacts/models/forecast3_stress_metrics.json | 2558 | .json |
| artifacts/models/in_sample_metrics.json | 97 | .json |
| artifacts/models/nowcast_baseline_metrics.json | 1061 | .json |
| artifacts/models/nowcast_full_metrics.json | 2851 | .json |
| artifacts/models/nowcast_full_news_metrics.json | 3136 | .json |
| artifacts/models/nowcast_stress_metrics.json | 2559 | .json |
| artifacts/models/thresholds.json | 167 | .json |
| artifacts/models/walk_forward_summary_baseline.json | 944 | .json |
| artifacts/models/walk_forward_summary_full.json | 906 | .json |
| artifacts/models/walk_forward_summary_full_news.json | 910 | .json |
| artifacts/models/walk_forward_summary_stress.json | 913 | .json |
| artifacts/oof_preds/alpha_curve_baseline.parquet | 3273 | .parquet |
| artifacts/oof_preds/alpha_curve_full.parquet | 3273 | .parquet |
| artifacts/oof_preds/alpha_curve_full_news.parquet | 3273 | .parquet |
| artifacts/oof_preds/alpha_curve_stress.parquet | 3265 | .parquet |
| artifacts/oof_preds/composite_oof.parquet | 16268 | .parquet |
| artifacts/oof_preds/composite_oof_calibrated.parquet | 16977 | .parquet |
| artifacts/oof_preds/composite_reference_baseline.parquet | 20041 | .parquet |
| artifacts/oof_preds/composite_reference_full.parquet | 20017 | .parquet |
| artifacts/oof_preds/composite_reference_full_news.parquet | 20017 | .parquet |
| artifacts/oof_preds/composite_reference_stress.parquet | 20017 | .parquet |
| artifacts/oof_preds/forecast3_fixed_baseline.parquet | 8999 | .parquet |
| artifacts/oof_preds/forecast3_fixed_full.parquet | 8999 | .parquet |
| artifacts/oof_preds/forecast3_fixed_full_news.parquet | 8999 | .parquet |
| artifacts/oof_preds/forecast3_fixed_stress.parquet | 8999 | .parquet |
| artifacts/oof_preds/forecast3_logit_baseline.parquet | 8987 | .parquet |
| artifacts/oof_preds/forecast3_logit_full.parquet | 8995 | .parquet |
| artifacts/oof_preds/forecast3_logit_full_news.parquet | 8995 | .parquet |
| artifacts/oof_preds/forecast3_logit_stress.parquet | 8995 | .parquet |
| artifacts/oof_preds/forecast3_rf_baseline.parquet | 8995 | .parquet |
| artifacts/oof_preds/forecast3_rf_full.parquet | 8955 | .parquet |
| artifacts/oof_preds/forecast3_rf_full_news.parquet | 8947 | .parquet |
| artifacts/oof_preds/forecast3_rf_stress.parquet | 8971 | .parquet |
| artifacts/oof_preds/forecast3_stacked_baseline.parquet | 8999 | .parquet |
| artifacts/oof_preds/forecast3_stacked_full.parquet | 8999 | .parquet |
| artifacts/oof_preds/forecast3_stacked_full_news.parquet | 8999 | .parquet |
| artifacts/oof_preds/forecast3_stacked_stress.parquet | 8999 | .parquet |
| artifacts/oof_preds/forecast3_xgb_baseline.parquet | 7484 | .parquet |
| artifacts/oof_preds/forecast3_xgb_full.parquet | 7089 | .parquet |
| artifacts/oof_preds/forecast3_xgb_full_news.parquet | 7120 | .parquet |
| artifacts/oof_preds/forecast3_xgb_stress.parquet | 6994 | .parquet |
| artifacts/oof_preds/nowcast_fixed_baseline.parquet | 8998 | .parquet |
| artifacts/oof_preds/nowcast_fixed_full.parquet | 8990 | .parquet |
| artifacts/oof_preds/nowcast_fixed_full_news.parquet | 8998 | .parquet |
| artifacts/oof_preds/nowcast_fixed_stress.parquet | 8998 | .parquet |
| artifacts/oof_preds/nowcast_logit_baseline.parquet | 8986 | .parquet |
| artifacts/oof_preds/nowcast_logit_full.parquet | 8986 | .parquet |
| artifacts/oof_preds/nowcast_logit_full_news.parquet | 8986 | .parquet |
| artifacts/oof_preds/nowcast_logit_stress.parquet | 8994 | .parquet |
| artifacts/oof_preds/nowcast_rf_baseline.parquet | 8962 | .parquet |
| artifacts/oof_preds/nowcast_rf_full.parquet | 8823 | .parquet |
| artifacts/oof_preds/nowcast_rf_full_news.parquet | 8807 | .parquet |
| artifacts/oof_preds/nowcast_rf_stress.parquet | 8815 | .parquet |
| artifacts/oof_preds/nowcast_stacked_baseline.parquet | 8998 | .parquet |
| artifacts/oof_preds/nowcast_stacked_full.parquet | 8990 | .parquet |
| artifacts/oof_preds/nowcast_stacked_full_news.parquet | 8998 | .parquet |
| artifacts/oof_preds/nowcast_stacked_stress.parquet | 8998 | .parquet |
| artifacts/oof_preds/nowcast_xgb_baseline.parquet | 7475 | .parquet |
| artifacts/oof_preds/nowcast_xgb_full.parquet | 7245 | .parquet |
| artifacts/oof_preds/nowcast_xgb_full_news.parquet | 7228 | .parquet |
| artifacts/oof_preds/nowcast_xgb_stress.parquet | 7169 | .parquet |
| artifacts/reports/awry_export_20260421_1005.tex | 32796 | .tex |
| artifacts/reports/awry_summary_20260421_1005.md | 27994 | .md |

## Report-Writing Checklist

- State which feature set is treated as the primary model in the paper.
- Report the fitted alpha and threshold as learned operating parameters, not user preferences.
- Distinguish OOF walk-forward results from fitted/reference-history probabilities.
- Disclose folds with zero positives when discussing AUROC/F1 stability.
- Discuss threshold sensitivity separately from the fitted operating point.
- Report vintage-vs-revised differences for ALFRED scenarios as data-revision sensitivity.
- Use the artifact inventory to cite exact files used for every table or figure.

## Bundle Generation Status

| section | populated | missing_files |
| --- | --- | --- |
| header | yes | 0 |
| feature set | yes | 0 |
| fitted params | yes | 1 |
| OOS metrics | yes | 0 |
| ablation | yes | 0 |
| diagnostics | yes | 0 |
| scenario backtests | yes | 0 |
| ALFRED vintage comparison | yes | 0 |
| current state | yes | 0 |
| recent raw feature values | yes | 0 |
| artifact inventory | yes | 0 |
| report-writing checklist | yes | 0 |

### Missing or Unreadable Inputs

- File not found: artifacts/models/alpha_tuned.json
