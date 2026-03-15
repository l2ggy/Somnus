"""
Somnus SQLite session store.

Why JSON-in-SQLite at this stage
---------------------------------
SharedState is a deeply nested Pydantic model whose schema will keep evolving
as new agents are added.  Storing it as a single JSON blob lets us iterate on
the model freely without writing migrations for every new field.  SQLite gives
us durability (survives restarts), ACID writes, and zero infrastructure — it is
a single file on disk.

When to move to a normalized schema
-------------------------------------
Once the model stabilises and we need cross-session queries (e.g. "show me all
users with rested_score < 4 this week"), we can extract JournalEntry and
SensorSnapshot into their own tables while keeping the top-level SharedState
blob for the orchestrator.  The four-function interface below stays identical
no matter what the backing store looks like.

Thread safety
--------------
Each operation opens its own connection via _conn(), uses it, and closes it.
This is safe for single-process uvicorn.  For multi-worker deployments replace
_conn() with a connection-pool context (e.g. sqlite3 WAL mode + pool, or swap
to Postgres with psycopg2/asyncpg).

Database location
------------------
Defaults to somnus.db in the working directory (project root when uvicorn is
run from there).  Override with the SOMNUS_DB_PATH environment variable.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.models.userstate import SharedState

logger = logging.getLogger(__name__)

# Path to the SQLite file — override via env var for testing or deployment.
DB_PATH: str = os.getenv("SOMNUS_DB_PATH", "somnus.db")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create the database file and sessions table if they do not already exist.

    Call once at application startup (see lifespan handler in main.py).
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS is idempotent.
    Also initialises outer-loop tables (user_policies, night_reports).
    """
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                user_id    TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
    logger.info("SQLite store initialised at %s", os.path.abspath(DB_PATH))

    # Initialise outer loop tables in the same database file.
    from app.outer_loop import store as outer_store
    outer_store.init_db()


# ---------------------------------------------------------------------------
# Core CRUD
# ---------------------------------------------------------------------------

def save_state(state: SharedState) -> None:
    """
    Upsert a SharedState row.  Creates the row if user_id is new; overwrites
    state_json and updated_at if it already exists.
    """
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions (user_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (state.user_id, state.model_dump_json(), _now()),
        )
        conn.commit()


def load_state(user_id: str) -> SharedState | None:
    """
    Load and validate a SharedState from the database.

    Returns None if the user_id does not exist or if the stored JSON fails
    Pydantic validation (e.g. after a schema migration).  Never raises.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT state_json FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    try:
        return SharedState.model_validate_json(row["state_json"])
    except Exception as exc:
        # The row exists but the JSON doesn't fit the current schema.
        # Log it and return None so the caller can treat it as "not found"
        # rather than crashing the request.
        logger.error(
            "Failed to deserialise state for user_id=%s (schema mismatch?): %s",
            user_id, exc,
        )
        return None


def delete_state(user_id: str) -> bool:
    """Delete a session row.  Returns True if a row was actually deleted."""
    with _conn() as conn:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE user_id = ?", (user_id,)
        )
        conn.commit()
    return cursor.rowcount > 0


def list_states() -> list[str]:
    """Return all user_ids, most-recently-updated first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT user_id FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [row["user_id"] for row in rows]


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# The orchestrator and routes call these names; they map 1-to-1 onto the core
# functions above so none of the callers need to change.
# ---------------------------------------------------------------------------

def create_session(state: SharedState) -> SharedState:
    """Save a new session and return state (chainable)."""
    save_state(state)
    return state


def get_session(user_id: str) -> SharedState | None:
    """Load a session by user_id.  Returns None if not found."""
    return load_state(user_id)


def save_session(state: SharedState) -> None:
    """Persist an updated state, overwriting the previous version."""
    save_state(state)


def list_sessions() -> list[str]:
    """Return all active user_ids."""
    return list_states()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextmanager
def _conn():
    """Open a SQLite connection, yield it, then close it."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    try:
        yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
