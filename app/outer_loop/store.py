"""
Outer loop SQLite store.

Stores two things, both as JSON blobs in separate tables:
  1. UserPolicy   — one row per user, updated after each night.
  2. NightReport  — one row per (user, date), append-only audit log.

Follows the same pattern as app/store.py: JSON-in-SQLite, no migrations needed,
same _conn() helper, all operations open-and-close their own connection.

init_db() is called from app/store.init_db() via a delegation call added there,
so initialization is automatic at startup.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.models.outer_loop import NightSummary, InterventionReview, UserPolicy

logger = logging.getLogger(__name__)

DB_PATH: str = os.getenv("SOMNUS_DB_PATH", "somnus.db")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create outer-loop tables if they don't exist.  Safe to call repeatedly."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_policies (
                user_id    TEXT PRIMARY KEY,
                policy_json TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS night_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                date        TEXT NOT NULL,
                summary_json   TEXT NOT NULL,
                review_json    TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE(user_id, date)
            )
        """)
        conn.commit()
    logger.info("Outer loop tables initialised at %s", os.path.abspath(DB_PATH))


# ---------------------------------------------------------------------------
# UserPolicy CRUD
# ---------------------------------------------------------------------------

def save_policy(policy: UserPolicy) -> None:
    """Upsert a UserPolicy.  Creates if new, overwrites if existing."""
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO user_policies (user_id, policy_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                policy_json = excluded.policy_json,
                updated_at  = excluded.updated_at
            """,
            (policy.user_id, policy.model_dump_json(), _now()),
        )
        conn.commit()


def load_policy(user_id: str) -> UserPolicy | None:
    """Load a UserPolicy by user_id.  Returns None if not found or invalid."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT policy_json FROM user_policies WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    try:
        return UserPolicy.model_validate_json(row["policy_json"])
    except Exception as exc:
        logger.error("Failed to load UserPolicy for %s: %s", user_id, exc)
        return None


def default_policy(user_id: str) -> UserPolicy:
    """Return a fresh, zero-state policy for a new user."""
    return UserPolicy(
        user_id=user_id,
        updated_at=_now(),
    )


def load_or_default_policy(user_id: str) -> UserPolicy:
    """Load policy if it exists, otherwise return a fresh default."""
    return load_policy(user_id) or default_policy(user_id)


# ---------------------------------------------------------------------------
# NightReport CRUD
# ---------------------------------------------------------------------------

def save_night_report(
    user_id: str,
    date: str,
    summary: NightSummary,
    review: InterventionReview,
) -> None:
    """
    Persist one night's summary + review.  Overwrites if (user_id, date) exists.
    Silently logs errors — reporting should never block the main flow.
    """
    try:
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO night_reports (user_id, date, summary_json, review_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, date) DO UPDATE SET
                    summary_json = excluded.summary_json,
                    review_json  = excluded.review_json,
                    created_at   = excluded.created_at
                """,
                (
                    user_id,
                    date,
                    summary.model_dump_json(),
                    review.model_dump_json(),
                    _now(),
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.error("Failed to save night report for %s/%s: %s", user_id, date, exc)


def load_night_report(user_id: str, date: str) -> dict | None:
    """Load a night report for a specific date.  Returns raw dict or None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT summary_json, review_json FROM night_reports WHERE user_id = ? AND date = ?",
            (user_id, date),
        ).fetchone()

    if row is None:
        return None

    try:
        return {
            "summary": json.loads(row["summary_json"]),
            "review":  json.loads(row["review_json"]),
        }
    except Exception as exc:
        logger.error("Failed to parse night report for %s/%s: %s", user_id, date, exc)
        return None


def list_night_reports(user_id: str, limit: int = 14) -> list[str]:
    """Return dates of recent night reports for a user, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT date FROM night_reports WHERE user_id = ? ORDER BY date DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [row["date"] for row in rows]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
