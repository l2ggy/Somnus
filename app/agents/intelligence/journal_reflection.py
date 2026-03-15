"""
Journal Reflection Agent
------------------------
Runs once at the end of a sleep session to process the user's morning journal
entry and generate a reflection that feeds back into future planning.

Two execution paths:

  run(state, entry)      — deterministic, always available.
                           Computes quality label and rolling average.
                           No insights or suggestions.

  run_gpt(state, entry)  — GPT-backed, returns (SharedState, mode_used).
                           Sends the journal entry + recent history to the LLM
                           and asks for a structured reflection with pattern
                           insights and a practical suggestion.
                           Falls back to the deterministic path on any failure.

The GPT path enriches the reflection with two extra fields that the
deterministic path leaves empty:
  insights   — list of pattern observations across multiple nights
  suggestion — one actionable suggestion for tomorrow night

Both paths write the same base structure so downstream consumers (orchestrator,
strategist on the next night) can read either output uniformly.
"""

import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app import llm_client
from app.models.userstate import JournalEntry, SharedState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(state: SharedState, entry: JournalEntry) -> SharedState:
    """Deterministic reflection.  Always succeeds."""
    reflection = _reflect_deterministic(entry, state)
    return _apply(state, entry, reflection)


def run_gpt(state: SharedState, entry: JournalEntry) -> tuple[SharedState, str]:
    """
    GPT-backed reflection with automatic fallback.

    Returns:
        (updated_state, mode_used) where mode_used is "gpt" or "fallback".
    """
    try:
        reflection = _reflect_gpt(entry, state)
        return _apply(state, entry, reflection), "gpt"
    except Exception as exc:
        logger.warning("journal_reflection GPT path failed (%s), using fallback", exc)
        return run(state, entry), "fallback"


# ---------------------------------------------------------------------------
# Shared state-update helper
# ---------------------------------------------------------------------------

def _apply(state: SharedState, entry: JournalEntry, reflection: dict) -> SharedState:
    """Append the journal entry and reflection to shared state."""
    updated_history    = state.journal_history + [entry]
    updated_hypotheses = state.hypotheses + [
        {"source": "journal_reflection", "reflection": reflection}
    ]
    return state.model_copy(update={
        "journal_history": updated_history,
        "hypotheses":      updated_hypotheses,
    })


# ---------------------------------------------------------------------------
# Deterministic reflection
# ---------------------------------------------------------------------------

def _reflect_deterministic(entry: JournalEntry, state: SharedState) -> dict:
    quality = (
        "poor" if entry.rested_score <= 3
        else "fair" if entry.rested_score <= 6
        else "good"
    )

    avg_score = None
    if state.journal_history:
        scores    = [e.rested_score for e in state.journal_history]
        avg_score = round(sum(scores) / len(scores), 1)

    return {
        "date":               entry.date,
        "quality":            quality,
        "rested_score":       entry.rested_score,
        "awakenings":         entry.awakenings_reported,
        "rolling_avg_score":  avg_score,
        "notes":              entry.notes or "No notes provided.",
        "insights":           [],
        "suggestion":         None,
    }


# ---------------------------------------------------------------------------
# Pydantic model for GPT reflection output (constraint 4)
# ---------------------------------------------------------------------------

class _GptReflectionOutput(BaseModel):
    """
    Validates the raw JSON dict returned by the LLM before it enters the app.

    Using Pydantic here (rather than manual key checks) means:
      - Missing required fields raise a ValidationError, caught by run_gpt().
      - Type coercion is applied automatically (e.g. stringifying insight items).
      - The model definition is the authoritative source of what the LLM must return.
    """
    quality:    Literal["poor", "fair", "good"]
    notes:      str
    insights:   List[str] = Field(default_factory=list)
    suggestion: Optional[str] = None

    @field_validator("insights", mode="before")
    @classmethod
    def cap_and_stringify(cls, v):
        """Ensure insights is a list of strings, capped at 3 items."""
        if not isinstance(v, list):
            return []
        return [str(item) for item in v[:3]]

    @field_validator("suggestion", mode="before")
    @classmethod
    def stringify_suggestion(cls, v):
        """Coerce non-None suggestion to string."""
        return str(v) if v is not None else None


# ---------------------------------------------------------------------------
# GPT reflection
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a thoughtful sleep reflection assistant for an app called Somnus.
Your role is to analyse a user's sleep journal and identify useful patterns.

Rules:
- Respond with a single JSON object. No prose, no markdown, no extra keys.
- Do not make medical diagnoses or health claims.
- Be honest but constructive — if the data is insufficient for patterns, say so.
- quality must be exactly one of: poor, fair, good
- insights must be a JSON array of strings (up to 3 items), or [] if insufficient data
- suggestion must be a single actionable sentence, or null if nothing useful can be said
- Keep notes to 1–2 sentences.

Output schema (JSON only):
{
  "quality": "poor|fair|good",
  "notes": "...",
  "insights": ["..."],
  "suggestion": "..."
}"""


def _reflect_gpt(entry: JournalEntry, state: SharedState) -> dict:
    """Call the LLM and return an enriched reflection dict.  Raises on failure."""
    history  = state.journal_history[-7:]  # last 7 nights is enough for pattern detection
    avg_score = None
    if history:
        avg_score = round(sum(e.rested_score for e in history) / len(history), 1)

    history_lines = []
    for e in history:
        history_lines.append(
            f"  {e.date}: rested={e.rested_score}/10, "
            f"awakenings={e.awakenings_reported}, notes=\"{e.notes}\""
        )
    history_summary = "\n".join(history_lines) if history_lines else "  No prior history."

    iv   = state.active_intervention
    iv_context = f"{iv.type} at intensity {iv.intensity:.2f}" if iv.type != "none" else "none"

    user_prompt = f"""\
Tonight's journal entry:
  Date:                {entry.date}
  Rested score (1-10): {entry.rested_score}
  Reported awakenings: {entry.awakenings_reported}
  Notes:               {entry.notes or 'none'}

Recent history ({len(history)} nights):
{history_summary}

Rolling average rested score: {avg_score if avg_score is not None else 'n/a'}
Last night's active intervention: {iv_context}

Reflect on this night and identify any patterns."""

    raw = llm_client.chat_json(_SYSTEM_PROMPT, user_prompt)

    # Validate the GPT output with Pydantic before using any of it.
    # Raises ValidationError on missing/wrong-typed fields — caught by run_gpt().
    parsed = _GptReflectionOutput.model_validate(raw)

    # Base fields are always computed deterministically so GPT output
    # can never corrupt immutable facts (date, score, awakenings count).
    base_avg = None
    if state.journal_history:
        scores   = [e.rested_score for e in state.journal_history]
        base_avg = round(sum(scores) / len(scores), 1)

    return {
        "date":              entry.date,
        "quality":           parsed.quality,
        "rested_score":      entry.rested_score,
        "awakenings":        entry.awakenings_reported,
        "rolling_avg_score": base_avg,
        "notes":             parsed.notes or entry.notes or "No notes provided.",
        "insights":          parsed.insights,
        "suggestion":        parsed.suggestion,
    }
