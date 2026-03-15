"""
train_sleep_model.py
--------------------
Trains a Random Forest sleep-state classifier on the extracted feature dataset.

Input:  data/processed/sleep_training_full.csv  (merged sleep_accel + DREAMT)
Output: models/sleep_model.pkl      (fitted RandomForestClassifier)
        models/label_encoder.pkl    (LabelEncoder so runtime can decode predictions)

This module is an offline training utility only.
It has no dependency on app/ and must never be imported from there.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "sleep_training_full.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "sleep_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# Feature / target config
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "motion_mean",
    "motion_std",
    "motion_energy",
    "hr_mean",
    "hr_std",
    "steps_sum",
    "prev_motion_mean",
    "time_since_sleep_start",
    "rolling_motion_mean",
    "rolling_hr_mean",
]

TARGET_COL = "sleep_stage"

# Canonical label order (preserved in encoder for deterministic class indices)
STAGE_ORDER = ["awake", "light", "deep", "rem"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-subject temporal features derived from base signal columns.

    Computed per subject (sorted by timestamp):
      prev_motion_mean       — motion_mean from the previous window (0 for first)
      time_since_sleep_start — seconds elapsed from the subject's first valid window
      rolling_motion_mean    — mean of motion_mean over the current + 2 prior windows
      rolling_hr_mean        — mean of hr_mean over the current + 2 prior windows

    Args:
        df: Merged feature DataFrame with at least subject_id, timestamp,
            motion_mean, and hr_mean columns.

    Returns:
        DataFrame with four new columns appended, sorted by subject_id, timestamp.
    """
    df = df.sort_values(["subject_id", "timestamp"]).reset_index(drop=True).copy()
    grp = df.groupby("subject_id")

    df["prev_motion_mean"] = grp["motion_mean"].shift(1).fillna(0.0)
    df["time_since_sleep_start"] = df["timestamp"] - grp["timestamp"].transform("min")
    df["rolling_motion_mean"] = grp["motion_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["rolling_hr_mean"] = grp["hr_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    return df


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the feature CSV, engineer temporal features, and drop NaN rows."""
    df = pd.read_csv(path)
    df = _engineer_temporal_features(df)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing values (%.1f%%)", dropped, 100 * dropped / before)
    logger.info("Dataset loaded: %d rows, %d subjects", len(df), df["subject_id"].nunique())
    return df


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string stage labels to integers using a fixed stage order.

    Returns:
        Tuple of (encoded integer array, fitted LabelEncoder).
    """
    le = LabelEncoder()
    le.fit(STAGE_ORDER)  # fix class order: awake=0, light=1, deep=2, rem=3
    encoded = le.transform(y)
    return encoded, le


def train(df: pd.DataFrame) -> tuple[RandomForestClassifier, LabelEncoder, dict]:
    """Full train/eval pipeline.

    Args:
        df: Feature DataFrame from load_dataset().

    Returns:
        Tuple of (fitted model, label encoder, metrics dict).
    """
    X = df[FEATURE_COLS].to_numpy()
    y_raw = df[TARGET_COL]
    y, le = encode_labels(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("Train: %d rows  |  Test: %d rows", len(X_train), len(X_test))

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    logger.info("Model trained.")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {"accuracy": acc, "report": report, "confusion_matrix": cm, "label_encoder": le}
    return clf, le, metrics


def save_artifacts(clf: RandomForestClassifier, le: LabelEncoder) -> None:
    """Persist model and label encoder to the models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    logger.info("Saved model   → %s", MODEL_PATH)
    logger.info("Saved encoder → %s", ENCODER_PATH)


def print_metrics(metrics: dict) -> None:
    acc = metrics["accuracy"]
    report = metrics["report"]
    cm = metrics["confusion_matrix"]
    le: LabelEncoder = metrics["label_encoder"]

    print(f"\n{'='*52}")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'='*52}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    header = "  ".join(f"{c:>6}" for c in le.classes_)
    print(f"         {header}")
    for label, row in zip(le.classes_, cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {label:<6}  {row_str}")
    print()


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = load_dataset()
    clf, le, metrics = train(df)
    print_metrics(metrics)
    save_artifacts(clf, le)
    print("Done.")
