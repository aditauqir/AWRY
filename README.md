# AWRY — Are We in a Recession Yet?

Composite recession risk from FRED data: **0.5 × nowcast + 0.5 × 3-month-ahead** probability.

## Setup

1. Copy `.env.example` to `.env` and set `FRED_API_KEY` ([FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)).
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## Run dashboard

From the project root:

```bash
.\.venv\Scripts\python.exe -m streamlit run src\dashboard\app.py
```

## Tests

```bash
.\.venv\Scripts\pytest.exe
```

## Layout

See `AWRY_SPEC.md` for the full architecture (ingestion, features, models, Streamlit UI).
