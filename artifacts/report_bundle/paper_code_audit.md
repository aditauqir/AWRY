# AWRY Paper-to-Code Audit

Audit date: 2026-04-23.

Scope note: `AWRY_Paper_Code_Audit.md` and `awry_phase3.pdf`/`awry_phase3.tex` were not present in this checkout. The closest paper source found was `artifacts/reports/awry_export_20260421_1005.tex`, with the generated markdown mirror at `artifacts/reports/awry_summary_20260421_1005.md`. The 15 targets below follow the concrete target list in the request and compare those claims to the checked-in code and artifacts.

Status legend: ✅ VERIFIED; ⚠️ MINOR DISCREPANCY; ❌ MISMATCH; 🔍 CANNOT VERIFY.

## 1. Composite Formula and Fitted Alpha

Status: ✅ VERIFIED

Paper claim: `P_AWRY = alpha P_now + (1-alpha) P_3m`, with fitted `alpha = 1`.

Code location: `src/models/composite.py:8-9`; `src/evaluation/walk_forward.py:306`; `src/evaluation/walk_forward.py:263-270`.

Evidence:

```python
# src/models/composite.py:8-9
def composite_score(p_now: np.ndarray, p_3m: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return alpha * p_now + (1.0 - alpha) * p_3m

# src/evaluation/walk_forward.py:306
out["P_AWRY"] = alpha * out["P_now"] + (1.0 - alpha) * out["P_3m"]
```

Artifact check:

```json
// artifacts/models/composite_baseline_metrics.json
{
  "alpha": 1.0
}
```

## 2. Composite Baseline Metrics: Alpha, AUROC, Brier, F1

Status: ✅ VERIFIED

Paper claim: composite `alpha = 1`, AUROC `0.8872`, Brier `0.0466`, F1 `0.5714`.

Code location: `src/evaluation/walk_forward.py:307-311`; `src/evaluation/metrics.py:9-20`.

Evidence:

```json
// artifacts/models/composite_baseline_metrics.json
{
  "alpha": 1.0,
  "metrics": {
    "brier": 0.046555626349682376,
    "auroc": 0.8872481684981686,
    "f1": 0.5714285714285714
  }
}
```

Parquet recomputation from `artifacts/oof_preds/composite_oof.parquet`:

```text
brier = 0.046555626349682376
auroc = 0.8872481684981686
f1@0.5 = 0.5714285714285714
```

Note: `src/evaluation/metrics.py:19-20` computes the metrics-file F1 at threshold `0.5`, while `thresholds.json` separately reports F1 at the fitted threshold.

## 3. Fitted Operating Threshold

Status: ✅ VERIFIED

Paper claim: fitted threshold `tau* = 0.232540`.

Code location: `src/evaluation/walk_forward.py:341-358`; `src/evaluation/walk_forward.py:373`.

Evidence:

```python
# src/evaluation/walk_forward.py:349-357
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
thresholds = np.append(thresholds, 1.0)
f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
best_idx = int(np.nanargmax(f1))
payload = {
    "threshold": float(thresholds[best_idx]),
    "precision": float(precision[best_idx]),
    "recall": float(recall[best_idx]),
    "f1": float(f1[best_idx]),
}
```

Artifact check:

```json
// artifacts/models/thresholds.json
{
  "threshold": 0.2325401096317268,
  "precision": 0.7727272727272727,
  "recall": 0.6071428571428571,
  "f1": 0.68,
  "class_imbalance": 0.08235294117647059
}
```

## 4. Nowcast Horizon Weights

Status: ✅ VERIFIED

Paper claim: nowcast horizon ensemble weights are logit `0.18`, RF `0.82`.

Code location: `src/evaluation/walk_forward.py:109-112`; `src/models/ensemble.py:18-35`.

Evidence:

```python
# src/evaluation/walk_forward.py:109-112
def _fixed_ensemble_predictions(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, tuple[float, float]]:
    w1, w2 = brier_optimal_weights(df["y_true"].values, df["logit"].values, df["rf"].values)
    p = ensemble_predict(w1, w2, df["logit"].values, df["rf"].values)
    return p, (float(w1), float(w2))
```

Artifact check:

```json
// artifacts/models/nowcast_baseline_metrics.json
{
  "target": "target_h0",
  "feature_set": "baseline",
  "fixed_weights": [0.18, 0.8200000000000001]
}
```

## 5. Three-Month Forecast Horizon Weights

Status: ✅ VERIFIED

Paper claim: 3-month horizon ensemble weights are logit `0.09`, RF `0.91`.

Code location: `src/evaluation/walk_forward.py:109-112`; `src/models/ensemble.py:18-35`.

Evidence:

```json
// artifacts/models/forecast3_baseline_metrics.json
{
  "target": "target_h3",
  "feature_set": "baseline",
  "fixed_weights": [0.09, 0.91]
}
```

The weights are found by the same Brier-grid code used for nowcast:

```python
# src/models/ensemble.py:24-31
grid = np.linspace(0.0, 1.0, 101)
for w1 in grid:
    w2 = 1.0 - w1
    logit_combo = w1 * _logit(p_logit) + w2 * _logit(p_rf)
    p = _sigmoid(logit_combo)
    score = brier_score_loss(y_val, p)
```

## 6. Composite OOF Fold Structure

Status: ✅ VERIFIED

Paper claim: purged walk-forward OOF folds are used; persisted `composite_oof.parquet` contains fold structure.

Code location: `src/evaluation/walk_forward.py:51-55`; `src/evaluation/walk_forward.py:121-133`; `src/evaluation/walk_forward.py:138-153`; `src/evaluation/walk_forward.py:300-310`.

Evidence:

```python
# src/evaluation/walk_forward.py:51-55
def make_cv(n_splits: int = 5, gap: int = 3) -> TimeSeriesSplit:
    """Build the purged walk-forward splitter used everywhere downstream."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)

# src/evaluation/walk_forward.py:153
oof.loc[X_test.index, "fold_idx"] = fold_idx
```

Parquet query result:

```text
artifacts/oof_preds/composite_oof.parquet shape = (340, 8)
date range = 1997-10-31 through 2026-01-31

fold_idx  start       end         rows  positives
1         1997-10-31  2003-05-31   68   8
2         2003-06-30  2009-01-31   68   13
3         2009-02-28  2014-09-30   68   5
4         2014-10-31  2020-05-31   68   2
5         2020-06-30  2026-01-31   68   0
```

## 7. Alpha Grid Shape

Status: ✅ VERIFIED

Paper claim: alpha grid runs `0.00` to `1.00` by `0.05`, selected by OOF Brier; fitted alpha is `1.0`.

Code location: `src/evaluation/walk_forward.py:240-272`; `src/evaluation/walk_forward.py:451`.

Evidence:

```python
# src/evaluation/walk_forward.py:263-267
for alpha in np.linspace(0.0, 1.0, 21):
    p = alpha * frame["P_now"].values + (1.0 - alpha) * frame["P_3m"].values
    brier = brier_score_loss(frame["y_true"].values, p)
    ap = average_precision_score(frame["y_true"].values, p)
    rows.append({"alpha": float(alpha), "brier": float(brier), "average_precision": float(ap)})
```

Parquet query result:

```text
artifacts/oof_preds/alpha_curve_baseline.parquet columns = [alpha, brier, average_precision]
shape = (21, 3)
unique alpha step = 0.05
minimum row: alpha = 1.0, brier = 0.046555626349682376, average_precision = 0.535778849655
```

## 8. Scenario Value: 2007-08-31

Status: ❌ MISMATCH

Paper/checklist claim under audit: `P_AWRY ≈ 0.323`.

Code location: `src/evaluation/walk_forward.py:300-306`; artifact `artifacts/oof_preds/composite_oof.parquet`.

Evidence:

```text
date        fold_idx  USREC  target_h0  target_h3  P_now          P_3m          P_AWRY
2007-08-31  2         0      0          0          0.269116080223 0.090526692256 0.269116080223
```

Difference from claim: `0.269116080223 - 0.323 = -0.053883919777`.

The generated paper export currently agrees with the parquet, not with the checklist value: `artifacts/reports/awry_export_20260421_1005.tex:222` reports `0.269116`.

## 9. Scenario Value: 2008-02-29

Status: ❌ MISMATCH

Paper/checklist claim under audit: `P_AWRY ≈ 0.395`.

Code location: `src/evaluation/walk_forward.py:300-306`; artifact `artifacts/oof_preds/composite_oof.parquet`.

Evidence:

```text
date        fold_idx  USREC  target_h0  target_h3  P_now          P_3m          P_AWRY
2008-02-29  2         1      1          1          0.232540109632 0.065065234208 0.232540109632
```

Difference from claim: `0.232540109632 - 0.395 = -0.162459890368`.

The generated paper export currently reports `0.232540` at `artifacts/reports/awry_export_20260421_1005.tex:228`.

## 10. Scenario Value: 2008-09-30

Status: ❌ MISMATCH

Paper/checklist claim under audit: `P_AWRY ≈ 0.952`.

Code location: `src/evaluation/walk_forward.py:300-306`; artifact `artifacts/oof_preds/composite_oof.parquet`.

Evidence:

```text
date        fold_idx  USREC  target_h0  target_h3  P_now          P_3m          P_AWRY
2008-09-30  2         1      1          1          0.932589023270 0.456859681172 0.932589023270
```

Difference from claim: `0.932589023270 - 0.952 = -0.019410976730`.

The generated paper export currently reports `0.932589` at `artifacts/reports/awry_export_20260421_1005.tex:235`.

## 11. Scenario Value: 2008-10-31

Status: ⚠️ MINOR DISCREPANCY

Paper/checklist claim under audit: `P_AWRY ≈ 0.929`.

Code location: `src/evaluation/walk_forward.py:300-306`; artifact `artifacts/oof_preds/composite_oof.parquet`.

Evidence:

```text
date        fold_idx  USREC  target_h0  target_h3  P_now          P_3m          P_AWRY
2008-10-31  2         1      1          1          0.933967019044 0.378228171376 0.933967019044
```

Difference from claim: `0.933967019044 - 0.929 = 0.004967019044`.

The generated paper export currently reports `0.933967` at `artifacts/reports/awry_export_20260421_1005.tex:236`.

## 12. Scenario Value: 2020-03-31

Status: ❌ MISMATCH

Paper/checklist claim under audit: `P_AWRY ≈ 0.571`.

Code location: `src/evaluation/walk_forward.py:300-306`; artifact `artifacts/oof_preds/composite_oof.parquet`.

Evidence:

```text
date        fold_idx  USREC  target_h0  target_h3  P_now          P_3m          P_AWRY
2020-03-31  4         1      1          0          0.957121969843 0.689966426219 0.957121969843
```

Difference from claim: `0.957121969843 - 0.571 = 0.386121969843`.

The generated paper export currently reports `0.957122` at `artifacts/reports/awry_export_20260421_1005.tex:373`.

## 13. Scenario Value: 2020-04-30

Status: ❌ MISMATCH

Paper/checklist claim under audit: `P_AWRY ≈ 0.504`.

Code location: `src/evaluation/walk_forward.py:300-306`; artifact `artifacts/oof_preds/composite_oof.parquet`.

Evidence:

```text
date        fold_idx  USREC  target_h0  target_h3  P_now          P_3m          P_AWRY
2020-04-30  4         1      1          0          0.939962437145 0.640370272386 0.939962437145
```

Difference from claim: `0.939962437145 - 0.504 = 0.435962437145`.

The generated paper export currently reports `0.939962` at `artifacts/reports/awry_export_20260421_1005.tex:374`.

## 14. January 2026 Latest Row

Status: ❌ MISMATCH

Paper/checklist claim under audit: `2026-01-31: P_AWRY ≈ 0.0554, P_now ≈ 0.0441, P_3m ≈ 0.1190`.

Code location: `src/evaluation/walk_forward.py:300-306`; artifact `artifacts/oof_preds/composite_oof.parquet`.

Evidence:

```text
date        fold_idx  USREC  target_h0  target_h3  P_now          P_3m          P_AWRY
2026-01-31  5         0      0          0          0.033949185022 0.126174077146 0.033949185022
```

Differences from claim:

```text
P_AWRY: 0.033949185022 - 0.0554 = -0.021450814978
P_now:  0.033949185022 - 0.0441 = -0.010150814978
P_3m:   0.126174077146 - 0.1190 =  0.007174077146
```

The generated paper export currently agrees with the parquet: `artifacts/reports/awry_export_20260421_1005.tex:66-70` reports January 2026, `P_AWRY = 3.39%`, `P_now = 3.39%`, `P_3m = 12.62%`, and `artifacts/reports/awry_export_20260421_1005.tex:443` reports the underlying row.

## 15. Constant Block: 1997-10-31 Through 2003-05-31

Status: ❌ MISMATCH

Paper/checklist claim under audit: all values constant at approximately `0.0951`.

Code location: `src/evaluation/walk_forward.py:300-306`; artifact `artifacts/oof_preds/composite_oof.parquet`.

Evidence:

```text
filter: date >= 1997-10-31 and date <= 2003-05-31
rows = 68
start = 1997-10-31
end = 2003-05-31

P_AWRY nunique = 1, min = 0.068253786961, max = 0.068253786961
P_now  nunique = 1, min = 0.068253786961, max = 0.068253786961
P_3m   nunique = 1, min = 0.092572185156, max = 0.092572185156
```

First and last rows:

```text
date        P_AWRY        P_now          P_3m          USREC  target_h0  target_h3  fold_idx
1997-10-31  0.068253786961 0.068253786961 0.092572185156 0     0          0          1
2003-05-31  0.068253786961 0.068253786961 0.092572185156 0     0          0          1
```

Difference from claimed `P_AWRY ≈ 0.0951`: `0.068253786961 - 0.0951 = -0.026846213039`.

The "constant" part is verified; the claimed value is not. The generated paper export currently reports `0.068254` for the early block, starting at `artifacts/reports/awry_export_20260421_1005.tex:104`.

## Summary Table

| # | Target | Status | Artifact result |
|---:|---|---|---|
| 1 | Composite formula and alpha | ✅ VERIFIED | Formula implemented; `alpha = 1.0` |
| 2 | Composite baseline metrics | ✅ VERIFIED | AUROC `0.8872481684981686`; Brier `0.046555626349682376`; F1@0.5 `0.5714285714285714` |
| 3 | Fitted threshold | ✅ VERIFIED | `0.2325401096317268` |
| 4 | Nowcast weights | ✅ VERIFIED | `[0.18, 0.8200000000000001]` |
| 5 | Forecast3 weights | ✅ VERIFIED | `[0.09, 0.91]` |
| 6 | OOF fold structure | ✅ VERIFIED | 5 folds, 68 rows each, 340 total rows |
| 7 | Alpha grid shape | ✅ VERIFIED | 21 rows, `0.00` to `1.00` by `0.05`; min at `1.0` |
| 8 | 2007-08-31 `P_AWRY ≈ 0.323` | ❌ MISMATCH | Actual `0.269116080223` |
| 9 | 2008-02-29 `P_AWRY ≈ 0.395` | ❌ MISMATCH | Actual `0.232540109632` |
| 10 | 2008-09-30 `P_AWRY ≈ 0.952` | ❌ MISMATCH | Actual `0.932589023270` |
| 11 | 2008-10-31 `P_AWRY ≈ 0.929` | ⚠️ MINOR DISCREPANCY | Actual `0.933967019044` |
| 12 | 2020-03-31 `P_AWRY ≈ 0.571` | ❌ MISMATCH | Actual `0.957121969843` |
| 13 | 2020-04-30 `P_AWRY ≈ 0.504` | ❌ MISMATCH | Actual `0.939962437145` |
| 14 | 2026-01-31 latest row | ❌ MISMATCH | Actual `P_AWRY = 0.033949185022`, `P_now = 0.033949185022`, `P_3m = 0.126174077146` |
| 15 | 1997-10-31 through 2003-05-31 constant block | ❌ MISMATCH | Constant verified, but at `P_AWRY = 0.068253786961`, not `0.0951` |

## Items Requiring Paper Update

1. If the missing audit checklist or paper draft states the requested scenario values, update them to the current `composite_oof.parquet` values listed above, or identify the older artifact/run that produced the requested values.
2. Update the January 2026 latest-row values to `P_AWRY = 0.033949185022`, `P_now = 0.033949185022`, and `P_3m = 0.126174077146` if the paper is meant to describe the current checked-in artifacts.
3. Update the early constant block claim from approximately `0.0951` to `P_AWRY = 0.068253786961`; `P_3m` is `0.092572185156`.
4. Clarify the F1 label: `composite_baseline_metrics.json` reports F1 at threshold `0.5` (`0.5714285714285714`), while `thresholds.json` reports F1 at the fitted threshold (`0.68`).
5. Add or restore the canonical paper/checklist files (`awry_phase3.pdf`/`.tex` and `AWRY_Paper_Code_Audit.md`) if future audits should target them exactly.

## Items Requiring Code Review

1. No code defect is evident for the verified formula, alpha search, metric generation, threshold selection, fold assignment, or horizon-weight artifacts.
2. Code review is warranted only if the requested scenario values are intended to be canonical; in that case, investigate artifact lineage or rerun differences for `artifacts/oof_preds/composite_oof.parquet`.
3. The first fold's constant predictions are real in the current artifact. Review model behavior for fold 1 only if a constant OOF block is scientifically undesirable rather than merely a reporting issue.
