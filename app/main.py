"""
Somnus — FastAPI entry point.

Routes are thin: they delegate all pipeline logic to app.orchestrator and all
persistence to app.store.  No agent is imported or called here.

Session lifecycle (live endpoints):
  POST /session/start               → create session, run pre-sleep planning
  POST /session/{user_id}/sensor    → ingest sensor data, run night tick
  GET  /session/{user_id}/state     → read current state
  POST /session/{user_id}/journal   → morning reflection, close session night

Demo/example endpoints (stateless, no session store):
  GET /pipeline/example  → tick on a hardcoded noisy-sleep scenario
  GET /plan/example      → planning on a hardcoded pre-sleep state
  GET /journal/example   → reflection on a hardcoded completed-night state
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app import orchestrator, store
from app.pipeline_contracts import agent_registry
from app.models.api import SensorPayload, StartSessionRequest
from app.models.userstate import JournalEntry, SharedState
from app.demo_support.demo_states import (
    deep_sleep_state,
    noisy_light_sleep_state,
    post_night_state,
    pre_sleep_state,
    sample_journal_entry,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    store.init_db()
    yield


app = FastAPI(
    title="Somnus",
    description="Agentic sleep assistant — shared-state backend",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Agent registry — describes the pipeline for dashboards and the orchestrator.
# ---------------------------------------------------------------------------

AGENT_REGISTRY = agent_registry()


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------

@app.post("/session/start", response_model=SharedState, tags=["session"])
def session_start(
    req: StartSessionRequest,
    mode: str = Query(default="deterministic", pattern="^(deterministic|gpt)$"),
):
    """
    Start a new sleep session for a user.

    Creates a SharedState from the supplied preferences, runs pre-sleep
    planning (strategist), stores the result, and returns it.

    ?mode=deterministic  (default) — rule-based planner, always fast
    ?mode=gpt            — LLM-backed planner; falls back if call fails

    Calling this again for the same user_id replaces the existing session.
    """
    state = SharedState(
        user_id=req.user_id,
        preferences=req.preferences,
        journal_history=req.journal_history,
    )
    result = orchestrator.run_pre_sleep_planning(state, mode=mode)
    planned_state = store.create_session(result["state"])
    return planned_state


@app.post("/session/{user_id}/sensor", tags=["session"])
def session_sensor(user_id: str, payload: SensorPayload):
    """
    Submit a sensor reading and run one night tick.

    Flow: load session → ingest sensor → run tick pipeline → save → return.

    Returns the tick result: updated state, per-agent trace, and any warnings.
    If latest_sensor was already set it is overwritten with the new reading.
    """
    state = _require_session(user_id)

    state = orchestrator.ingest_sensor(state, payload.model_dump())
    result = orchestrator.run_night_tick(state)

    # Always save — even if the tick reported ok=False the sensor was ingested.
    store.save_session(result["state"])
    return result


@app.get("/session/{user_id}/state", response_model=SharedState, tags=["session"])
def session_state(user_id: str):
    """Return the current SharedState for a session."""
    return _require_session(user_id)


@app.get("/session/{user_id}/journal", response_model=list[JournalEntry], tags=["session"])
def session_journal_history(user_id: str):
    """Return all journal entries for a session, newest last."""
    state = _require_session(user_id)
    return state.journal_history

@app.post("/session/{user_id}/journal", tags=["session"])
def session_journal(
    user_id: str,
    entry: JournalEntry,
    mode: str = Query(default="deterministic", pattern="^(deterministic|gpt)$"),
):
    """
    Submit a morning journal entry and run the reflection pipeline.

    ?mode=deterministic  (default) — rule-based reflection
    ?mode=gpt            — LLM-backed reflection with insights + suggestion

    Appends the entry to journal_history, generates a reflection, saves the
    updated state, and returns both the state and the reflection summary.
    """
    state = _require_session(user_id)
    result = orchestrator.run_morning_reflection(state, entry, mode=mode)
    store.save_session(result["state"])
    return result


@app.get("/session", tags=["session"])
def list_sessions():
    """List all active session user_ids."""
    return {"sessions": store.list_sessions()}


def _require_session(user_id: str) -> SharedState:
    """Load a session or raise 404."""
    state = store.get_session(user_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No session found for user_id '{user_id}'")
    return state


# ---------------------------------------------------------------------------
# Meta routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    """Liveness check."""
    return {"status": "ok", "service": "somnus"}


@app.get("/agents", tags=["meta"])
def list_agents():
    """Return the agent registry with lifecycle annotations."""
    return {"agents": AGENT_REGISTRY}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@app.get("/state/example", response_model=SharedState, tags=["state"])
def example_state():
    """
    Return a static example SharedState showing the full model structure.
    In production this will be GET /state/{user_id} returning the live session.
    """
    return deep_sleep_state()


# ---------------------------------------------------------------------------
# Pipeline demo routes — each delegates entirely to the orchestrator.
# ---------------------------------------------------------------------------

@app.get("/pipeline/example", tags=["pipeline"])
def pipeline_example():
    """
    Run the deterministic night tick on a noisy light-sleep scenario.

    Scenario: 03:15 am, light sleep, 62 dB street noise.
    Expected result: disturbance detected, brown_noise intervention activated.

    Returns the updated SharedState plus a per-agent trace.
    """
    state = noisy_light_sleep_state()
    return orchestrator.run_night_tick(state)


@app.get("/plan/example", tags=["pipeline"])
def plan_example(
    mode: str = Query(default="deterministic", pattern="^(deterministic|gpt)$"),
):
    """
    Run pre-sleep planning on a sample state and return the generated NightlyPlan.

    ?mode=deterministic  — rule-based planner (default, instant)
    ?mode=gpt            — LLM-backed planner; response includes mode_used field
                           so you can confirm whether GPT or fallback was used
    """
    state  = pre_sleep_state()
    result = orchestrator.run_pre_sleep_planning(state, mode=mode)
    return {
        "nightly_plan": result["state"].nightly_plan,
        "mode_used":    result["mode_used"],
        "final_state":  result["state"],
    }


@app.get("/journal/example", tags=["pipeline"])
def journal_example(
    mode: str = Query(default="deterministic", pattern="^(deterministic|gpt)$"),
):
    """
    Run morning reflection on a completed-night state plus a sample journal entry.

    ?mode=deterministic  — rule-based reflection (default, instant)
    ?mode=gpt            — LLM-backed reflection; adds insights + suggestion fields
    """
    state  = post_night_state()
    entry  = sample_journal_entry()
    return orchestrator.run_morning_reflection(state, entry, mode=mode)
