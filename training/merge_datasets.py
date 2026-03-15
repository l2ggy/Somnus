"""
merge_datasets.py
-----------------
Merges the sleep_accel and DREAMT feature datasets into a single unified
training CSV using the canonical column schema.

Input files
-----------
    data/processed/sleep_training.csv
    data/processed/dreamt_training.csv

Output file
-----------
    data/processed/sleep_training_full.csv

This module is an offline training utility only.
It has no dependency on app/ and must never be imported from there.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

SLEEP_ACCEL_PATH = PROCESSED_DIR / "sleep_training.csv"
DREAMT_PATH = PROCESSED_DIR / "dreamt_training.csv"
OUTPUT_PATH = PROCESSED_DIR / "sleep_training_full.csv"

REQUIRED_COLUMNS = [
    "subject_id",
    "timestamp",
    "motion_mean",
    "motion_std",
    "motion_energy",
    "hr_mean",
    "hr_std",
    "hr_rmssd",
    "hr_sdnn",
    "steps_sum",
    "sleep_stage",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        logger.error("%s not found: %s", label, path)
        sys.exit(1)
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows, columns: %s", label, len(df), list(df.columns))
    return df


def _align_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Enforce REQUIRED_COLUMNS — add missing ones with 0, drop extras."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.info("%s: adding missing columns with 0: %s", label, missing)
        for col in missing:
            df[col] = 0

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        logger.info("%s: dropping extra columns: %s", label, extra)

    return df[REQUIRED_COLUMNS]


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------


def merge_datasets(
    sleep_accel_path: Path = SLEEP_ACCEL_PATH,
    dreamt_path: Path = DREAMT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Load, align, concatenate, and save both datasets.

    Args:
        sleep_accel_path: Path to sleep_training.csv.
        dreamt_path: Path to dreamt_training.csv.
        output_path: Destination for the merged CSV.

    Returns:
        The merged DataFrame.
    """
    df_accel = _load(sleep_accel_path, "sleep_accel")
    df_dreamt = _load(dreamt_path, "dreamt")

    df_accel = _align_columns(df_accel, "sleep_accel")
    df_dreamt = _align_columns(df_dreamt, "dreamt")

    df = pd.concat([df_accel, df_dreamt], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    return df


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df_accel = _load(SLEEP_ACCEL_PATH, "sleep_accel")
    df_dreamt = _load(DREAMT_PATH, "dreamt")

    df_accel = _align_columns(df_accel, "sleep_accel")
    df_dreamt = _align_columns(df_dreamt, "dreamt")

    print(f"\nRows per dataset:")
    print(f"  sleep_accel : {len(df_accel):>6}")
    print(f"  dreamt      : {len(df_dreamt):>6}")

    df = pd.concat([df_accel, df_dreamt], ignore_index=True)
    print(f"  total       : {len(df):>6}")

    print(f"\nStage distribution:")
    print(df["sleep_stage"].value_counts().to_string())

    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
