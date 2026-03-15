"""
train_sleep_model.py
--------------------
Trains a Random Forest sleep-state classifier on the extracted feature dataset.

Input:  data/processed/sleep_training.csv
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

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "sleep_training.csv"
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
]

# Categorical column encoded via get_dummies before training
CATEGORICAL_COLS = ["prev_stage"]
TARGET_COL = "sleep_stage"

# Canonical label order (preserved in encoder for deterministic class indices)
STAGE_ORDER = ["awake", "light", "deep", "rem"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the feature CSV and drop rows with any NaN in feature or target columns."""
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + CATEGORICAL_COLS + [TARGET_COL])
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

    prev_stage is one-hot encoded via get_dummies() and concatenated with the
    numeric feature columns before training.

    Args:
        df: Feature DataFrame from load_dataset().

    Returns:
        Tuple of (fitted model, label encoder, metrics dict).
    """
    stage_dummies = pd.get_dummies(df["prev_stage"], prefix="prev_stage")
    X = pd.concat([df[FEATURE_COLS], stage_dummies], axis=1).to_numpy()
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
