# AWRY: Are We in a Recession Yet?

AWRY is a monthly U.S. recession-risk dashboard built from FRED macroeconomic and market data. It trains walk-forward recession nowcast and 3-month forecast models, combines them into an AWRY probability, and displays model diagnostics, scenario backtests, feature drivers, and research-paper export tables in Streamlit.

The current default dashboard model uses the `baseline` feature set because it is the strongest feature set under the data currently available in this repository.

## What AWRY Does

- Downloads and caches economic data from FRED.
- Builds a monthly feature panel from labor, production, sales, rates, spreads, housing, equity, and volatility series.
- Trains out-of-sample walk-forward models for `P(recession_t)` and `P(recession_t+3)`.
- Selects a composite blend by out-of-fold Brier score.
- Chooses an operating threshold from the out-of-fold precision-recall curve.
- Shows a Streamlit dashboard with current risk, historical probabilities, backtest scenarios, model diagnostics, and export buttons.
- Produces research artifacts under `artifacts/`.

## Model Summary

AWRY uses a stacked classification pipeline:

- Target labels come from FRED `USREC`.
- `target_h0` is the current-month NBER recession indicator.
- `target_h3` is the NBER recession indicator three months ahead.
- The default feature set is `baseline`.
- Base models are regularized logistic regression, capped random forest, and optional XGBoost if installed.
- The meta-model is an L2-regularized logistic stacker trained on base-model log-odds.
- Evaluation uses 5-fold purged walk-forward cross-validation with a 3-month gap.
- The headline AWRY probability is `alpha * P_now + (1 - alpha) * P_3m`, where `alpha` is selected by out-of-fold Brier score.

## Data Used

The default `baseline` model uses these FRED series and selected lags:

| Category | Series |
|---|---|
| Labor | `PAYEMS`, `UNRATE`, `ICSA` |
| Production and income | `INDPRO`, `W875RX1`, `RRSFS` |
| Rates and spreads | `T10Y3M`, `FEDFUNDS`, `BAA10Y` |
| Housing | `HOUST`, `PERMIT` |
| Market data | `NASDAQCOM`, `VIXCLS` |
| Recession target | `USREC` |

Additional feature sets exist for experiments:

| Feature set | Description |
|---|---|
| `baseline` | Core macro, rates, housing, equity return, VIX, and lags |
| `stress` | Baseline plus stress and structural indicators |
| `full` | Stress set plus additional macro/financial indicators |
| `full_news` | Full set plus optional local Excel news-sentiment features |

The local news-sentiment file is optional. If it is missing, AWRY fills that column with missing values and the default dashboard avoids relying on it.

## Requirements

- Python 3.10 or newer
- A FRED API key
- Internet access for the first data pull
- Windows PowerShell, macOS Terminal, or Linux shell

Get a free FRED API key here:

https://fred.stlouisfed.org/docs/api/api_key.html

## Installation

Clone the repository:

```powershell
git clone https://github.com/aditauqir/AWRY.git
cd AWRY
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once for the current shell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set:

```text
FRED_API_KEY=your_fred_api_key_here
```

macOS/Linux equivalent:

```bash
cp .env.example .env
```

## Run the Dashboard

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\dashboard\app.py
```

macOS/Linux equivalent:

```bash
.venv/bin/python -m streamlit run src/dashboard/app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

The first run may take a while because AWRY downloads FRED data, builds features, trains walk-forward models, and writes artifacts.

## Refresh Data

Use the dashboard sidebar button:

```text
Refresh FRED data
```

This clears Streamlit caches and reruns the data/model pipeline.

You can also delete cached CSV files in `data/raw/` if you want a fully fresh FRED pull.

## Regenerate Model Artifacts

To regenerate the default baseline pipeline artifacts from PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -c "from awry_pipeline import fit_awry_pipeline; fit_awry_pipeline(cached=True, feature_set='baseline', save_artifacts=True)"
```

To run the feature-set ablation suite:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -c "from evaluation.ablation import run_ablation_suite; print(run_ablation_suite(cached=True).to_string(index=False))"
```

To generate the ALFRED vintage-vs-revised comparison:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m evaluation.alfred_comparison
```

macOS/Linux equivalent:

```bash
PYTHONPATH=src .venv/bin/python -c "from awry_pipeline import fit_awry_pipeline; fit_awry_pipeline(cached=True, feature_set='baseline', save_artifacts=True)"
PYTHONPATH=src .venv/bin/python -c "from evaluation.ablation import run_ablation_suite; print(run_ablation_suite(cached=True).to_string(index=False))"
PYTHONPATH=src .venv/bin/python -m evaluation.alfred_comparison
```

## Run Tests

Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

macOS/Linux:

```bash
.venv/bin/python -m pytest
```

## Outputs

AWRY writes generated files to `artifacts/`:

| Path | Contents |
|---|---|
| `artifacts/models/` | Metrics, thresholds, model summaries, ablation JSON |
| `artifacts/oof_preds/` | Out-of-fold prediction parquet files |
| `artifacts/figures/` | Figures, precision-recall curves, alpha curves, ALFRED comparison tables |
| `artifacts/reports/` | Reserved for report outputs |

The dashboard also includes export buttons for Markdown and LaTeX research summaries.

## Project Layout

```text
AWRY/
  artifacts/              Generated model outputs, figures, and tables
  data/raw/               Cached FRED CSV files
  data/alfred/            Cached ALFRED vintage parquet files
  src/awry_pipeline.py    Main model pipeline wrapper
  src/features/           Dataset construction and feature engineering
  src/ingestion/          FRED and ALFRED data clients
  src/models/             Logistic, random forest, XGBoost, and stacker code
  src/evaluation/         Walk-forward CV, ablations, calibration, ALFRED comparison
  src/dashboard/          Streamlit app and dashboard components
  tests/                  Pytest tests
```

## Troubleshooting

If `FRED_API_KEY missing` appears, confirm `.env` exists in the project root and contains a valid key.

If Streamlit shows stale numbers, click `Refresh FRED data` in the sidebar or restart the Streamlit process.

If PowerShell cannot activate the virtual environment, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

If artifact writing fails with `PermissionError`, close any program that may have the artifact file open, such as another Python process, Streamlit session, file previewer, or editor extension.

If ALFRED vintage fetching fails for one series, the comparison script may fall back to revised data truncated at the requested date and print a warning.

## Research Notes

The walk-forward evaluation uses current revised FRED data for the full historical panel. The ALFRED comparison provides point-in-time checks for selected recession scenarios. For a formal paper, report both the out-of-fold walk-forward metrics and the ALFRED vintage-vs-revised comparison so readers can see the effect of data revisions.

Current default baseline results from the latest local run:

| Metric | Value |
|---|---:|
| OOF window | 1997-10-31 to 2026-01-31 |
| OOF rows | 340 |
| Composite AUROC | 0.887 |
| Composite Brier score | 0.0466 |
| Composite F1 | 0.571 |
| Operating threshold | 0.2325 |
| 2008 GFC AWRY lead | 3 months |
| 2020 COVID AWRY lead | No pre-recession signal |

## License

Add your preferred license before publishing this repository publicly.
