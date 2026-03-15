"""
feature_extractor.py
--------------------
Converts raw signals from the sleep_accel dataset into 30-second feature
windows ready for ML training.

Labels define the window grid (already 30s bins).  For each window the
corresponding motion, heart-rate, and step samples are aggregated into a
single feature row.  Rows with no signal data in a window are dropped.

Sleep-stage mapping (PSG labels → Somnus phases):
    0 → awake
    1 → light   (N1)
    2 → light   (N2)
    3 → deep    (N3)
    5 → rem

This module is an offline training utility only.
It has no dependency on app/ and must never be imported from there.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from training.dataset_parser import SubjectRecord, build_subject_index

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SECONDS = 30

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "sleep_training.csv"

# PSG integer stage → Somnus phase string
_STAGE_MAP: dict[int, str] = {
    0: "awake",
    1: "light",
    2: "light",
    3: "deep",
    5: "rem",
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File loaders (raw format details isolated here)
# ---------------------------------------------------------------------------


def _load_motion(path: Path) -> pd.DataFrame:
    """Load acceleration file → DataFrame with columns [t, x, y, z]."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=["t", "x", "y", "z"])


def _load_heart_rate(path: Path) -> pd.DataFrame:
    """Load heart-rate file → DataFrame with columns [t, hr]."""
    return pd.read_csv(path, header=None, names=["t", "hr"])


def _load_steps(path: Path) -> pd.DataFrame:
    """Load steps file → DataFrame with columns [t, steps]."""
    return pd.read_csv(path, header=None, names=["t", "steps"])


def _load_labels(path: Path) -> pd.DataFrame:
    """Load label file → DataFrame with columns [t, stage].

    t is seconds elapsed from recording start (0, 30, 60, …).
    stage is the PSG-coded sleep stage integer.
    """
    return pd.read_csv(path, sep=r"\s+", header=None, names=["t", "stage"])


# ---------------------------------------------------------------------------
# Per-signal feature functions
# ---------------------------------------------------------------------------


def extract_motion_features(df_motion: pd.DataFrame) -> dict:
    """Compute summary features from a window of accelerometer samples.

    Args:
        df_motion: DataFrame with columns x, y, z (one row per sample).

    Returns:
        Dict with motion_mean, motion_std, motion_energy.
        motion_energy is the mean Euclidean magnitude sqrt(x²+y²+z²).
    """
    if df_motion.empty:
        return {"motion_mean": np.nan, "motion_std": np.nan, "motion_energy": np.nan}

    magnitude = np.sqrt(df_motion["x"] ** 2 + df_motion["y"] ** 2 + df_motion["z"] ** 2)
    return {
        "motion_mean": float(magnitude.mean()),
        "motion_std": float(magnitude.std(ddof=0)),
        "motion_energy": float(magnitude.mean()),  # mean magnitude as proxy for energy
    }


def extract_hr_features(df_hr: pd.DataFrame) -> dict:
    """Compute heart-rate summary features from a single window.

    Args:
        df_hr: DataFrame with column hr (bpm values).

    Returns:
        Dict with hr_mean, hr_std.
    """
    if df_hr.empty:
        return {"hr_mean": np.nan, "hr_std": np.nan}

    return {
        "hr_mean": float(df_hr["hr"].mean()),
        "hr_std": float(df_hr["hr"].std(ddof=0)),
    }


def extract_step_features(df_steps: pd.DataFrame) -> dict:
    """Compute step-count features from a single window.

    Args:
        df_steps: DataFrame with column steps.

    Returns:
        Dict with steps_sum.
    """
    if df_steps.empty:
        return {"steps_sum": 0.0}

    return {"steps_sum": float(df_steps["steps"].sum())}


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------


def build_feature_windows(subject_record: SubjectRecord, subject_id: str) -> list[dict]:
    """Produce one feature row per 30-second label window for a single subject.

    Per-user normalization is the first personalization layer; future versions
    may support user-specific fine-tuning.  Currently features are extracted
    from raw signals; normalization is applied in the training step.

    Args:
        subject_record: Paths returned by build_subject_index().
        subject_id: String identifier used to tag output rows.

    Returns:
        List of row dicts ready for pd.DataFrame construction.
        Windows with no signal data at all are silently dropped.
    """
    # Load raw files
    df_motion = _load_motion(subject_record["motion"])
    df_hr = _load_heart_rate(subject_record["heart_rate"])
    df_steps = _load_steps(subject_record["steps"])
    df_labels = _load_labels(subject_record["labels"])

    rows: list[dict] = []

    for _, label_row in df_labels.iterrows():
        win_start = float(label_row["t"])
        win_end = win_start + WINDOW_SECONDS
        stage_int = int(label_row["stage"])
        sleep_stage = _STAGE_MAP.get(stage_int)

        if sleep_stage is None:
            # Unknown stage code — skip
            continue

        # Slice each signal to the window
        motion_win = df_motion[(df_motion["t"] >= win_start) & (df_motion["t"] < win_end)]
        hr_win = df_hr[(df_hr["t"] >= win_start) & (df_hr["t"] < win_end)]
        steps_win = df_steps[(df_steps["t"] >= win_start) & (df_steps["t"] < win_end)]

        # Drop windows with no data at all (before-sleep negative-timestamp region)
        if motion_win.empty and hr_win.empty:
            continue

        row = {
            "subject_id": subject_id,
            "timestamp": win_start,
            **extract_motion_features(motion_win),
            **extract_hr_features(hr_win),
            **extract_step_features(steps_win),
            "sleep_stage": sleep_stage,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_training_dataset(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Iterate over all subjects, extract windows, and save to CSV.

    This is the main entry point for the offline training pipeline.
    The output CSV is saved to data/processed/sleep_training.csv.

    Args:
        output_path: Where to write the CSV (default: OUTPUT_PATH).

    Returns:
        The full training DataFrame.
    """
    index = build_subject_index()
    if not index:
        logger.error("No subjects found — aborting.")
        return pd.DataFrame()

    all_rows: list[dict] = []

    for subject_id, record in tqdm(index.items(), desc="Extracting subjects", unit="subj"):
        try:
            rows = build_feature_windows(record, subject_id)
            all_rows.extend(rows)
            logger.debug("Subject %s: %d windows", subject_id, len(rows))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Subject %s failed: %s", subject_id, exc)

    df = pd.DataFrame(all_rows)

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

    print(f"\nTotal windows: {len(df)}")
    if not df.empty:
        print(f"Subjects: {df['subject_id'].nunique()}")
        print(f"Stage distribution:\n{df['sleep_stage'].value_counts().to_string()}")
        print(f"\nExample rows:\n{df.head(3).to_string(index=False)}")
