# AWRY Deep Diagnostics

This report extracts secondary diagnostics from existing AWRY artifacts for research-paper strengthening. It does not modify model, feature, OOF, or model-artifact files.

Primary baseline alpha: **1**; fitted threshold: **0.23254**.

## 1. Alpha-Search Curve Shape

**Method.** Read `artifacts/oof_preds/alpha_curve_baseline.parquet`, report the complete alpha grid, identify the minimum Brier score, and classify the curve shape from the near-minimum region.

Minimum alpha: **1.00**; minimum Brier: **0.046556**; Brier range across grid: **0.011994**; near-minimum width: **0.20**; shape classification: **narrow minimum**.

| alpha | brier | average_precision |
| --- | --- | --- |
| 0 | 0.0585495 | 0.447455 |
| 0.05 | 0.057363 | 0.465041 |
| 0.1 | 0.0562383 | 0.477353 |
| 0.15 | 0.0551753 | 0.497654 |
| 0.2 | 0.0541741 | 0.520102 |
| 0.25 | 0.0532347 | 0.523166 |
| 0.3 | 0.0523571 | 0.536386 |
| 0.35 | 0.0515412 | 0.556291 |
| 0.4 | 0.0507871 | 0.559993 |
| 0.45 | 0.0500947 | 0.566138 |
| 0.5 | 0.0494642 | 0.568791 |
| 0.55 | 0.0488954 | 0.570364 |
| 0.6 | 0.0483883 | 0.571849 |
| 0.65 | 0.047943 | 0.571988 |
| 0.7 | 0.0475595 | 0.575893 |
| 0.75 | 0.0472378 | 0.577954 |
| 0.8 | 0.0469778 | 0.580763 |
| 0.85 | 0.0467796 | 0.578644 |
| 0.9 | 0.0466432 | 0.575436 |
| 0.95 | 0.0465685 | 0.5573 |
| 1 | 0.0465556 | 0.535779 |

**Interpretation.** The alpha search selects alpha=1.00. Because the best point sits at the edge and the near-minimum region spans 0.20 alpha units, the paper should describe the composite weight as an empirical OOF calibration result rather than a user preference.

## 2. Per-Fold AUROC/Brier/F1 Breakdown

**Method.** Read `artifacts/oof_preds/composite_oof.parquet`, group by `fold_idx`, and compute AUROC, Brier, and F1 at the fitted threshold for each fold.

| fold_idx | start | end | n | positives | positive_rate | auroc | brier | f1_at_tau |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1997-10-31 | 2003-05-31 | 68 | 8 | 0.117647 | 0.5 | 0.106246 | 0 |
| 2 | 2003-06-30 | 2009-01-31 | 68 | 13 | 0.191176 | 0.944056 | 0.0734208 | 0.846154 |
| 3 | 2009-02-28 | 2014-09-30 | 68 | 5 | 0.0735294 | 1 | 0.0174442 | 0.888889 |
| 4 | 2014-10-31 | 2020-05-31 | 68 | 2 | 0.0294118 | 0.992424 | 0.0195563 | 0.666667 |
| 5 | 2020-06-30 | 2026-01-31 | 68 | 0 | 0 |  | 0.0161109 | 0 |

**Interpretation.** Per-fold scores are uneven because recession months are clustered in time. Folds [5] have zero positives, so AUROC is undefined there and F1 is structurally limited.

## 3. Random Forest Top-15 Nowcast Feature Importances

**Method.** Rebuild the baseline feature table from local cached FRED CSVs and re-fit the nowcast random forest in memory only; no fitted model or artifact is saved.

Top 15 features:

| feature | category | importance |
| --- | --- | --- |
| PAYEMS | Labor | 0.199425 |
| PAYEMS_lag1 | Labor | 0.118473 |
| BAA10Y | Rates/spreads/financial stress | 0.102061 |
| INDPRO | Production/income/sales | 0.0941128 |
| PAYEMS_lag2 | Labor | 0.0808224 |
| VIXCLS | Market/volatility | 0.073545 |
| INDPRO_lag1 | Production/income/sales | 0.0474749 |
| BAA10Y_lag2 | Rates/spreads/financial stress | 0.0411264 |
| VIXCLS_lag1 | Market/volatility | 0.0377854 |
| INDPRO_lag2 | Production/income/sales | 0.0364041 |
| W875RX1 | Production/income/sales | 0.0263031 |
| VIXCLS_lag2 | Market/volatility | 0.0248461 |
| T10Y3M | Rates/spreads/financial stress | 0.0104073 |
| W875RX1_lag1 | Production/income/sales | 0.00973939 |
| RRSFS | Production/income/sales | 0.00909586 |

Grouped top-15 importance:

| category | importance |
| --- | --- |
| Labor | 0.39872 |
| Production/income/sales | 0.22313 |
| Rates/spreads/financial stress | 0.153594 |
| Market/volatility | 0.136176 |

**Interpretation.** The nowcast RF concentrates its top split importance in the Labor category among the top-15 features. Treat these as model-internal split importances, not causal effects.

## 4. Monthly 2007-2008 OOF Probability Trajectory

**Method.** Extract monthly OOF `P_now`, `P_3m`, and `P_AWRY` for 2007-2008 and compare them with `USREC` and the fitted threshold.

| date | USREC | P_now | P_3m | P_AWRY | threshold | above_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 2007-01-31 | 0 | 0.0300641 | 0.070036 | 0.0300641 | 0.23254 | 0 |
| 2007-02-28 | 0 | 0.0172611 | 0.0421753 | 0.0172611 | 0.23254 | 0 |
| 2007-03-31 | 0 | 0.0434004 | 0.038689 | 0.0434004 | 0.23254 | 0 |
| 2007-04-30 | 0 | 0.0568719 | 0.0505845 | 0.0568719 | 0.23254 | 0 |
| 2007-05-31 | 0 | 0.141878 | 0.102354 | 0.141878 | 0.23254 | 0 |
| 2007-06-30 | 0 | 0.136871 | 0.0472184 | 0.136871 | 0.23254 | 0 |
| 2007-07-31 | 0 | 0.159812 | 0.102273 | 0.159812 | 0.23254 | 0 |
| 2007-08-31 | 0 | 0.269116 | 0.0905267 | 0.269116 | 0.23254 | 1 |
| 2007-09-30 | 0 | 0.208281 | 0.0643313 | 0.208281 | 0.23254 | 0 |
| 2007-10-31 | 0 | 0.16501 | 0.0649885 | 0.16501 | 0.23254 | 0 |
| 2007-11-30 | 0 | 0.0977633 | 0.0591268 | 0.0977633 | 0.23254 | 0 |
| 2007-12-31 | 0 | 0.115143 | 0.0594209 | 0.115143 | 0.23254 | 0 |
| 2008-01-31 | 1 | 0.0853588 | 0.053097 | 0.0853588 | 0.23254 | 0 |
| 2008-02-29 | 1 | 0.23254 | 0.0650652 | 0.23254 | 0.23254 | 1 |
| 2008-03-31 | 1 | 0.280105 | 0.127331 | 0.280105 | 0.23254 | 1 |
| 2008-04-30 | 1 | 0.359917 | 0.0990974 | 0.359917 | 0.23254 | 1 |
| 2008-05-31 | 1 | 0.289553 | 0.124636 | 0.289553 | 0.23254 | 1 |
| 2008-06-30 | 1 | 0.114353 | 0.131276 | 0.114353 | 0.23254 | 0 |
| 2008-07-31 | 1 | 0.285975 | 0.0671725 | 0.285975 | 0.23254 | 1 |
| 2008-08-31 | 1 | 0.628397 | 0.309128 | 0.628397 | 0.23254 | 1 |
| 2008-09-30 | 1 | 0.932589 | 0.45686 | 0.932589 | 0.23254 | 1 |
| 2008-10-31 | 1 | 0.933967 | 0.378228 | 0.933967 | 0.23254 | 1 |
| 2008-11-30 | 1 | 0.942072 | 0.300546 | 0.942072 | 0.23254 | 1 |
| 2008-12-31 | 1 | 0.938416 | 0.239971 | 0.938416 | 0.23254 | 1 |

**Interpretation.** The 2007-2008 trajectory shows whether the model rose before or during the GFC recession window. The first fitted-threshold crossing in this window occurs on 2007-08-31.

## 5. Calibration Reliability and ECE

**Method.** Use calibrated OOF probabilities when available, split them into 10 equal-frequency bins, and compute expected calibration error.

Probability column used: **P_AWRY_calibrated**. Expected Calibration Error (ECE): **0.026546**.

| bin | n | p_min | p_max | mean_pred | observed_rate | abs_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 34 | 0 | 0 | 0 | 0 | 0 |
| 2 | 34 | 0 | 0 | 0 | 0 | 0 |
| 3 | 34 | 0 | 0 | 0 | 0 | 0 |
| 4 | 34 | 0 | 0 | 0 | 0 | 0 |
| 5 | 34 | 0 | 0 | 0 | 0 | 0 |
| 6 | 34 | 0 | 0 | 0 | 0 | 0 |
| 7 | 34 | 0 | 0.102564 | 0.0241327 | 0 | 0.0241327 |
| 8 | 34 | 0.102564 | 0.102564 | 0.102564 | 0 | 0.102564 |
| 9 | 34 | 0.102564 | 0.102564 | 0.102564 | 0.235294 | 0.13273 |
| 10 | 34 | 0.102564 | 1 | 0.594268 | 0.588235 | 0.00603318 |

**Interpretation.** Equal-frequency calibration bins compare predicted probability to realized recession frequency. A lower ECE indicates better calibration, but sparse recession months can make individual bins noisy.

## 6. Feature Collinearity Summary

**Method.** Rebuild the baseline feature matrix from local cached data, compute absolute Pearson correlations, count pairs above 0.90, and list the top five pairs.

Pairs with |r| > 0.90: **6** out of **595** feature pairs.

Top 5 correlated pairs:

| feature_a | feature_b | abs_corr |
| --- | --- | --- |
| UNRATE | UNRATE_lag1 | 0.949122 |
| UNRATE_lag1 | UNRATE_lag2 | 0.949059 |
| T10Y3M | T10Y3M_lag2 | 0.948964 |
| PAYEMS_lag1 | ICSA_lag2 | 0.920234 |
| PAYEMS | ICSA_lag1 | 0.920122 |

**Interpretation.** High collinearity is expected because macro features include levels and lags. The pipeline drops near-duplicate lag columns above its redundancy threshold, but correlated economic indicators can still remain.

## 7. Pipeline Timing

**Method.** Run a baseline walk-forward sequence in memory using local cached data and `save_artifacts=False`; skip the in-sample writer to avoid modifying model artifacts.

| run_scope | elapsed_seconds | alpha | threshold | composite_auroc | composite_brier | composite_f1 | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline in-memory walk-forward, no persisted artifacts | 2.65693 | 1 | 0.23254 | 0.887248 | 0.0465556 | 0.571429 | 340 |

**Interpretation.** This timing is a local-machine benchmark for an in-memory baseline walk-forward run. It should be treated as operational context, not a model-quality metric.

## 8. Hyperparameter Grid Searches

**Method.** Summarize searched versus fixed hyperparameters from the model code and persisted baseline metric artifacts.

| component | searched | fixed | winner_or_recorded |
| --- | --- | --- | --- |
| Logistic regression | C in [0.01, 0.1, 1.0, 10.0] | median imputation, StandardScaler, lbfgs, class_weight=balanced, max_iter=2000 | best estimator saved implicitly in OOF/final pipeline; winning C not persisted in artifact JSON |
| Random forest | none | n_estimators=300, max_depth=5, min_samples_leaf=10, min_samples_split=20, max_features=sqrt, class_weight=balanced | fixed weights now=[0.18, 0.8200000000000001], forecast3=[0.09, 0.91] |
| XGBoost | none; optional dependency | n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, early_stopping_rounds=30 | only used if xgboost is installed; current baseline artifacts record logit/rf meta coefficients |
| Stacker | none | L2 logistic meta-learner, C=1.0, solver=liblinear | now coefs={'intercept': -2.0216501620905927, 'logit': 0.35821764088619734, 'rf': -0.3153544087849427}; forecast3 coefs={'intercept': -2.2381554933852934, 'logit': 0.21329974856692013, 'rf': -0.21008091640644175} |
| Composite alpha | alpha grid 0.00 to 1.00 by 0.05 | selected by OOF Brier score | alpha=1.0 |
| Operating threshold | precision-recall threshold candidates from OOF P_AWRY | selected by max F1 | tau=0.23254010963172672 |

**Interpretation.** Most model hyperparameters are fixed for small-sample stability. The main searched quantities persisted for the paper are logistic C through CV, composite alpha, and the operating threshold.

## 9. Class Imbalance Handling

**Method.** Compute class counts for `target_h0` and `target_h3` from the baseline model table and apply scikit-learn's balanced class-weight formula.

scikit-learn balanced class weights use n_samples / (n_classes * class_count).

| horizon | target | class_0_months | class_1_months | positive_rate | balanced_weight_0 | balanced_weight_1 |
| --- | --- | --- | --- | --- | --- | --- |
| nowcast | target_h0 | 381 | 28 | 0.0684597 | 0.536745 | 7.30357 |
| forecast3 | target_h3 | 381 | 28 | 0.0684597 | 0.536745 | 7.30357 |

**Interpretation.** The recession class receives a much larger weight because recession months are rare. This helps the classifiers pay attention to minority-class recession observations during fitting.

## Generation Status

| section | completed | note |
| --- | --- | --- |
| 1. Alpha-Search Curve Shape | yes |  |
| 2. Per-Fold AUROC/Brier/F1 Breakdown | yes |  |
| 3. Random Forest Top-15 Nowcast Feature Importances | yes |  |
| 4. Monthly 2007-2008 OOF Probability Trajectory | yes |  |
| 5. Calibration Reliability and ECE | yes |  |
| 6. Feature Collinearity Summary | yes |  |
| 7. Pipeline Timing | yes |  |
| 8. Hyperparameter Grid Searches | yes |  |
| 9. Class Imbalance Handling | yes |  |

Protected artifact write check:

| check | result |
| --- | --- |
| artifacts/models timestamp changes | 0 |
| artifacts/oof_preds timestamp changes | 0 |
| new files under protected artifact dirs | 0 |
