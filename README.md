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

## Required Systems and Software

Use these versions or newer unless your instructor requires an exact environment lock.

| Requirement | Version / Notes |
|---|---|
| Operating system | Windows 10/11, macOS 12+, or a modern Linux distribution |
| Python | Python 3.10 or newer; the local development environment used Python 3.13 |
| Package manager | `pip` bundled with Python |
| Shell | Windows PowerShell 5+ / PowerShell 7+, macOS Terminal, or Linux shell |
| Git | Git 2.x, only needed if cloning from GitHub |
| FRED API key | Required for fresh economic-data downloads |
| Internet access | Required on the first run to pull FRED/ALFRED data unless cached data is already included |
| LaTeX distribution | Optional; TeX Live or MiKTeX is needed only if compiling exported `.tex` files to PDF |

Primary Python libraries are listed in `requirements.txt`:

| Library | Minimum version | Purpose |
|---|---:|---|
| `pandas` | 2.0.0 | Time-series tables, monthly panel construction, artifact loading |
| `numpy` | 1.24.0 | Numerical arrays and probability calculations |
| `scikit-learn` | 1.3.0 | Logistic regression, random forest, metrics, walk-forward modeling |
| `streamlit` | 1.28.0 | Interactive dashboard |
| `plotly` | 5.17.0 | Dashboard charts |
| `matplotlib` | 3.7.0 | Static diagnostic figures |
| `fredapi` | 0.5.0 | FRED data access |
| `python-dotenv` | 1.0.0 | `.env` loading |
| `pytest` | 7.4.0 | Test runner |

Get a free FRED API key here:

https://fred.stlouisfed.org/docs/api/api_key.html

## Submission Contents

Submit the full repository, including:

- All project source code under `src/`, `scripts/`, and `tests/`.
- `requirements.txt`, `.env.example`, `README.md`, and `LICENSE`.
- Generated artifacts under `artifacts/` if your instructor wants reproducible report outputs without rerunning the full pipeline.
- Any exported report files under `artifacts/reports/` and `artifacts/report_bundle/`.

No separate compiled executable is required for the standard submission because AWRY is a Python project run from source. If a course platform specifically asks for executables, include the Python entry commands below as the required execution method.

## Demo File Walkthrough

Use these files when showing the project structure in a demo. They tell the whole story from data collection to model output without needing to open every helper file.

```text
AWRY/
|-- README.md
|   Project overview, required software, install steps, run commands, and submission notes.
|-- requirements.txt
|   Python package list used to recreate the environment.
|-- LICENSE
|   MIT license for the project.
|-- src/
|   |-- config.py
|   |   Central path and environment-variable setup.
|   |-- awry_pipeline.py
|   |   Main training/prediction entry point that returns the fitted AWRY pipeline.
|   |-- pipeline.py
|   |   Compatibility shim that re-exports the main pipeline functions.
|   |-- ingestion/
|   |   |-- fred_client.py
|   |   |   Downloads and caches FRED economic series.
|   |   |-- alfred_client.py
|   |   |   Pulls ALFRED point-in-time vintage data for revision checks.
|   |   |-- aggregator.py
|   |   |   Converts high-frequency market data into month-end model inputs.
|   |   |-- vintage_config.py
|   |       Defines which series are revision-sensitive for vintage analysis.
|   |-- features/
|   |   |-- dataset_builder.py
|   |   |   Builds the monthly modeling table, feature sets, lags, and targets.
|   |   |-- transforms.py
|   |       Small feature transformations such as log differences.
|   |-- models/
|   |   |-- logit.py
|   |   |   Regularized logistic-regression model.
|   |   |-- rf.py
|   |   |   Regularized random-forest model.
|   |   |-- stacker.py
|   |   |   Logistic meta-learner that stacks base model probabilities.
|   |   |-- ensemble.py
|   |       Brier-score weighting utilities for probability blends.
|   |-- evaluation/
|   |   |-- walk_forward.py
|   |   |   Purged walk-forward validation, OOF predictions, alpha tuning, and threshold selection.
|   |   |-- diagnostics.py
|   |   |   Diagnostic CSV generator for fold structure, lead times, and false positives.
|   |   |-- ablation.py
|   |       Compares feature-set variants.
|   |-- dashboard/
|   |   |-- app.py
|   |   |   Streamlit dashboard entry point.
|   |   |-- export_summary.py
|   |   |   Builds the Markdown research export.
|   |   |-- export_latex.py
|   |       Builds the LaTeX/IEEE export.
|   |-- report/
|       |-- autoreport.py
|       |   Regenerates dashboard-style report files.
|       |-- generate_bundle.py
|           Aggregates artifacts into one paper-writing bundle.
|-- scripts/
|   |-- verify_lead_time.py
|       Checks OOF vs reference lead-time series for the recession scenarios.
|-- tests/
|   Unit tests for ingestion, features, models, and backtests.
|-- artifacts/
    |-- models/
    |   JSON model metrics and fitted operating parameters.
    |-- oof_preds/
    |   Out-of-fold prediction parquet files.
    |-- figures/
    |   Diagnostic CSVs and plots.
    |-- reports/
    |   Generated Markdown and LaTeX exports.
    |-- report_bundle/
        Consolidated research-paper bundle.
```

## Step-by-Step Installation

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

## Step-by-Step: Run the Dashboard

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

## Step-by-Step: Regenerate Reports Without Retraining

If the model artifacts already exist and you only want to regenerate the paper/report exports:

```powershell
.\.venv\Scripts\python.exe -m src.report.autoreport
.\.venv\Scripts\python.exe -m src.report.generate_bundle
```

macOS/Linux equivalent:

```bash
.venv/bin/python -m src.report.autoreport
.venv/bin/python -m src.report.generate_bundle
```

This writes report outputs to:

| Path | Output |
|---|---|
| `artifacts/reports/` | Markdown and LaTeX dashboard-style exports |
| `artifacts/report_bundle/report_bundle.md` | Consolidated research-paper bundle |

## Refresh Data

Use the dashboard sidebar button:

```text
Refresh FRED data
```

This clears Streamlit caches and reruns the data/model pipeline.

You can also delete cached CSV files in `data/raw/` if you want a fully fresh FRED pull.

## Step-by-Step: Regenerate Model Artifacts

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

## Step-by-Step: Run Tests

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
| `artifacts/reports/` | Markdown and LaTeX report outputs |
| `artifacts/report_bundle/` | Consolidated paper-writing bundle |

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
  src/report/             Artifact-only report and bundle generators
  src/dashboard/          Streamlit app and dashboard components
  scripts/                Diagnostic helper scripts
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

AWRY is released under the MIT License. See `LICENSE` for the full license text.
