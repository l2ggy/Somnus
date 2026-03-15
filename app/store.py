"""
Somnus in-memory session store.

A plain dict keyed by user_id.  This is intentionally the simplest possible
implementation — no locking, no TTL, no serialisation.  Everything lives in
process memory and is lost on restart.

Replacement path:
  When persistence is needed, swap the dict for a Redis client or a SQLAlchemy
  session and implement the same four-function interface.  The orchestrator and
  routes call only these functions, so nothing else changes.

Thread safety note:
  Python's GIL makes single-key reads and writes on a dict safe in a
  single-process server (e.g. uvicorn with one worker).  For multi-worker
  deployments this needs a shared store (Redis, etc.).
"""

from app.models.userstate import SharedState

# Module-level singleton — one dict for the lifetime of the process.
_sessions: dict[str, SharedState] = {}


def create_session(state: SharedState) -> SharedState:
    """
    Store a new session.  Overwrites any existing session for the same user_id.

    Returns the state unchanged so callers can chain: return create_session(state).
    """
    _sessions[state.user_id] = state
    return state


def get_session(user_id: str) -> SharedState | None:
    """Return the session for user_id, or None if it does not exist."""
    return _sessions.get(user_id)


def save_session(state: SharedState) -> None:
    """Persist an updated state, replacing the previous version."""
    _sessions[state.user_id] = state


def list_sessions() -> list[str]:
    """Return all active user_ids.  Useful for debugging and admin endpoints."""
    return list(_sessions.keys())
