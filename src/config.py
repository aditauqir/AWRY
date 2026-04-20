"""Shared project paths and environment helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OOF_PRED_DIR = ARTIFACTS_DIR / "oof_preds"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
ENV_PATH = PROJECT_ROOT / ".env"


def ensure_project_dirs() -> None:
    """Create the directories the evaluation/reporting pipeline writes to."""
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OOF_PRED_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_project_env(*, override: bool = False) -> None:
    """Load the repo-local .env file once per process or refresh when needed."""
    load_dotenv(ENV_PATH, override=override)


def require_env(name: str) -> str:
    """Return a required env var or raise a user-facing runtime error."""
    load_project_env()
    value = os.environ.get(name)
    if value:
        return value
    raise RuntimeError(f"{name} missing. Add it to {ENV_PATH.name} before running AWRY.")


ensure_project_dirs()
