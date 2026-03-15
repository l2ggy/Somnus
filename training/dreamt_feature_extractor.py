"""
dreamt_feature_extractor.py
---------------------------
Converts raw DREAMT wearable recordings (*_whole_df.csv) into 30-second
feature windows using the same schema as the sleep_accel pipeline.

Output schema
-------------
subject_id, timestamp, motion_mean, motion_std, motion_energy,
hr_mean, hr_std, hr_rmssd, hr_sdnn, steps_sum, sleep_stage

Sleep-stage mapping (DREAMT labels → Somnus phases):
    W  → awake
    N1 → light
    N2 → light
    N3 → deep
    R  → rem
    P  → skipped (pre-sleep / scoring artifact)

This module is an offline training utility only.
It has no dependency on app/ and must never be imported from there.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SECONDS = 30

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "dreamt"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "dreamt_training.csv"

_STAGE_MAP: dict[str, str] = {
    "W": "awake",
    "N1": "light",
    "N2": "light",
    "N3": "deep",
    "R": "rem",
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------


def find_subject_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return all *_whole_df.csv paths sorted by filename."""
    return sorted(raw_dir.glob("*_whole_df.csv"))


def subject_id_from_path(path: Path) -> str:
    """Extract subject ID (e.g. 'S002') from filename."""
    match = re.match(r"^(S\d+)_", path.name)
    return match.group(1) if match else path.stem


# ---------------------------------------------------------------------------
# Per-signal feature functions (match sleep_accel feature_extractor interface)
# ---------------------------------------------------------------------------


def _motion_features(window: pd.DataFrame) -> dict:
    acc = window[["ACC_X", "ACC_Y", "ACC_Z"]].to_numpy(dtype=float)
    if len(acc) == 0:
        return {"motion_mean": np.nan, "motion_std": np.nan, "motion_energy": np.nan}

    magnitude = np.sqrt((acc ** 2).sum(axis=1))
    return {
        "motion_mean": float(magnitude.mean()),
        "motion_std": float(magnitude.std(ddof=0)),
        "motion_energy": float(magnitude.mean()),  # mean magnitude as proxy for energy
    }


def _hr_features(window: pd.DataFrame) -> dict:
    hr = window["HR"].dropna().to_numpy(dtype=float)
    if len(hr) == 0:
        return {"hr_mean": np.nan, "hr_std": np.nan, "hr_rmssd": 0.0, "hr_sdnn": 0.0}

    return {
        "hr_mean": float(hr.mean()),
        "hr_std": float(hr.std(ddof=0)),
    }


def _hrv_features(window: pd.DataFrame) -> dict:
    """Compute RMSSD and SDNN from the IBI column.

    IBI is sampled at the same rate as the other signals and holds the most
    recent inter-beat interval (seconds). Consecutive duplicate values are
    removed so that each unique value represents one actual heartbeat.
    """
    ibi_raw = window["IBI"].dropna().to_numpy(dtype=float)

    # Keep only values that differ from their predecessor (beat boundaries)
    if len(ibi_raw) == 0:
        return {"hr_rmssd": 0.0, "hr_sdnn": 0.0}

    ibi = ibi_raw[np.concatenate(([True], ibi_raw[1:] != ibi_raw[:-1]))]

    if len(ibi) < 2:
        return {"hr_rmssd": 0.0, "hr_sdnn": 0.0}

    diff = np.diff(ibi)
    return {
        "hr_rmssd": float(np.sqrt(np.mean(diff ** 2))),
        "hr_sdnn": float(np.std(ibi, ddof=0)),
    }


# ---------------------------------------------------------------------------
# Per-subject window builder
# ---------------------------------------------------------------------------


def extract_subject_windows(path: Path) -> list[dict]:
    """Load one subject file and return a list of feature-row dicts."""
    subject_id = subject_id_from_path(path)

    df = pd.read_csv(path, usecols=["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z", "HR", "IBI", "Sleep_Stage"])

    # Work in integer window bins to avoid floating-point drift
    df["_bin"] = (df["TIMESTAMP"] // WINDOW_SECONDS).astype(int)

    rows: list[dict] = []

    for bin_idx, window in df.groupby("_bin", sort=True):
        # Dominant sleep stage (most frequent non-null label in the window)
        stage_counts = window["Sleep_Stage"].dropna().value_counts()
        if stage_counts.empty:
            continue

        raw_stage = stage_counts.index[0]
        sleep_stage = _STAGE_MAP.get(raw_stage)
        if sleep_stage is None:
            # Pre-sleep or unrecognised label — skip
            continue

        timestamp = float(bin_idx * WINDOW_SECONDS)

        row = {
            "subject_id": subject_id,
            "timestamp": timestamp,
            **_motion_features(window),
            **_hr_features(window),
            **_hrv_features(window),
            "steps_sum": 0.0,
            "sleep_stage": sleep_stage,
        }
        rows.append(row)

    logger.debug("Subject %s: %d windows from %s", subject_id, len(rows), path.name)
    return rows


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

COLUMN_ORDER = [
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


def build_training_dataset(
    raw_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Extract features for all DREAMT subjects and save to CSV.

    Args:
        raw_dir: Directory containing *_whole_df.csv files.
        output_path: Destination CSV path.

    Returns:
        The full training DataFrame.
    """
    subject_files = find_subject_files(raw_dir)
    if not subject_files:
        logger.error("No subject files found in %s", raw_dir)
        return pd.DataFrame()

    all_rows: list[dict] = []

    for path in subject_files:
        try:
            rows = extract_subject_windows(path)
            all_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process %s: %s", path.name, exc)

    df = pd.DataFrame(all_rows, columns=COLUMN_ORDER)

    if df.empty:
        logger.error("No windows extracted — check dataset.")
        return df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    return df


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = build_training_dataset()

    if df.empty:
        print("No data extracted.")
    else:
        n_subjects = df["subject_id"].nunique()
        n_windows = len(df)
        print(f"\nSubjects processed : {n_subjects}")
        print(f"Windows generated  : {n_windows}")
        print(f"\nStage distribution :\n{df['sleep_stage'].value_counts().to_string()}")
        print(f"\nExample rows :\n{df.head(3).to_string(index=False)}")
