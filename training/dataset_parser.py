"""
dataset_parser.py
-----------------
Indexes and loads the raw sleep-accel dataset for feature extraction and ML training.

Dataset root: data/raw/sleep_accel/
Layout:
    heart_rate/   <subject_id>_heartrate.txt
    motion/       <subject_id>_acceleration.txt
    steps/        <subject_id>_steps.txt
    labels/       <subject_id>_labeled_sleep.txt

This module is an offline training utility only.
It has no dependency on the app/ runtime and must never be imported from there.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).parent.parent / "data" / "raw" / "sleep_accel"

_SUBFOLDERS: dict[str, tuple[str, str]] = {
    # logical key  →  (subfolder name,  filename suffix including extension)
    "heart_rate":  ("heart_rate",  "_heartrate.txt"),
    "motion":      ("motion",      "_acceleration.txt"),
    "steps":       ("steps",       "_steps.txt"),
    "labels":      ("labels",      "_labeled_sleep.txt"),
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class SubjectRecord(TypedDict):
    heart_rate: Path
    motion: Path
    steps: Path
    labels: Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_subject_index(root: Path = DATASET_ROOT) -> dict[str, SubjectRecord]:
    """Scan the dataset root and return a complete subject index.

    Only subjects for which *all four* required files are present are included.
    Subjects with any missing file are logged as warnings.

    Args:
        root: Path to the dataset root directory (contains heart_rate/, motion/, etc.).

    Returns:
        A dict mapping subject ID strings to a SubjectRecord of four file Paths.

    Example::

        index = build_subject_index()
        # {"1066528": {"heart_rate": Path(...), "motion": Path(...), ...}, ...}
    """
    if not root.exists():
        logger.error("Dataset root not found: %s", root)
        return {}

    logger.info("Dataset root: %s", root.resolve())

    # Collect all subject IDs present in any subfolder
    all_ids: set[str] = set()
    for key, (subfolder, suffix) in _SUBFOLDERS.items():
        folder = root / subfolder
        if not folder.exists():
            logger.warning("Expected subfolder missing: %s", folder)
            continue
        for f in folder.iterdir():
            m = re.match(r"^(\w+)" + re.escape(suffix) + r"$", f.name)
            if m:
                all_ids.add(m.group(1))

    index: dict[str, SubjectRecord] = {}
    missing_any: list[str] = []

    for subject_id in sorted(all_ids):
        record: dict[str, Path] = {}
        missing_keys: list[str] = []

        for key, (subfolder, suffix) in _SUBFOLDERS.items():
            candidate = root / subfolder / f"{subject_id}{suffix}"
            if candidate.exists():
                record[key] = candidate
            else:
                missing_keys.append(key)

        if missing_keys:
            logger.warning(
                "Subject %s skipped — missing files for: %s",
                subject_id,
                ", ".join(missing_keys),
            )
            missing_any.append(subject_id)
        else:
            index[subject_id] = SubjectRecord(**record)  # type: ignore[typeddict-item]

    logger.info(
        "Subjects found: %d complete, %d incomplete/skipped",
        len(index),
        len(missing_any),
    )
    return index


def load_subject_files(record: SubjectRecord) -> dict[str, pd.DataFrame]:
    """Load all four files for a subject into pandas DataFrames.

    Files are tab/whitespace-delimited .txt exports from the MMASH dataset.
    No column renaming is applied here — callers receive raw DataFrames for
    use in feature extraction.

    Args:
        record: A SubjectRecord returned by build_subject_index().

    Returns:
        A dict with keys "heart_rate", "motion", "steps", "labels", each
        mapping to a pandas DataFrame.
    """
    loaders: dict[str, pd.DataFrame] = {}
    for key, path in record.items():
        try:
            loaders[key] = pd.read_csv(path, sep=r"\s+", engine="python")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s (%s): %s", key, path, exc)
            loaders[key] = pd.DataFrame()
    return loaders


# ---------------------------------------------------------------------------
# CLI test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    index = build_subject_index()
    print(f"\nSubjects indexed: {len(index)}")

    if index:
        first_id = next(iter(index))
        first_record = index[first_id]
        print(f"\nExample subject: {first_id}")
        for key, path in first_record.items():
            print(f"  {key:12s}  {path}")
    else:
        print("No complete subjects found — check dataset root.")
