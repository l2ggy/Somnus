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
from app.models.api import SensorPayload, StartSessionRequest
from app.models.userstate import (
    ActiveIntervention,
    JournalEntry,
    NightlyPlan,
    SensorSnapshot,
    SharedState,
    SleepState,
    UserPreferences,
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

AGENT_REGISTRY = [
    {
        "name": "intake",
        "module": "app.agents.intake",
        "lifecycle": "on_sensor_arrival",
        "purpose": "Validates and normalises raw sensor dicts into a SensorSnapshot.",
        "reads": ["raw_payload"],
        "writes": ["latest_sensor"],
    },
    {
        "name": "sensor_interpreter",
        "module": "app.agents.sensor_interpreter",
        "lifecycle": "tick",
        "purpose": "Translates raw sensor values into labelled signals (hr_status, noise_alert, etc.).",
        "reads": ["latest_sensor"],
        "writes": ["hypotheses"],
    },
    {
        "name": "sleep_state",
        "module": "app.agents.sleep_state",
        "lifecycle": "tick",
        "purpose": "Classifies sleep phase (awake/light/deep/REM) and estimates wake_risk.",
        "reads": ["latest_sensor", "hypotheses"],
        "writes": ["sleep_state"],
    },
    {
        "name": "disturbance",
        "module": "app.agents.disturbance",
        "lifecycle": "tick",
        "purpose": "Detects noise/light/movement/HR threats and annotates sleep_state.",
        "reads": ["latest_sensor", "sleep_state"],
        "writes": ["sleep_state.disturbance_reason", "sleep_state.wake_risk"],
    },
    {
        "name": "intervention",
        "module": "app.agents.intervention",
        "lifecycle": "tick",
        "purpose": "Selects and activates an intervention when wake_risk is elevated.",
        "reads": ["sleep_state", "nightly_plan", "preferences"],
        "writes": ["active_intervention"],
    },
    {
        "name": "strategist",
        "module": "app.agents.strategist",
        "lifecycle": "pre_sleep",
        "purpose": "Builds the NightlyPlan from user goals and journal history.",
        "reads": ["preferences", "journal_history"],
        "writes": ["nightly_plan"],
    },
    {
        "name": "journal_reflection",
        "module": "app.agents.journal_reflection",
        "lifecycle": "morning",
        "purpose": "Processes the morning journal entry and generates a reflection.",
        "reads": ["journal_history", "sleep_state", "hypotheses"],
        "writes": ["journal_history", "hypotheses"],
    },
]


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
    return _deep_sleep_state()


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
    state = _noisy_light_sleep_state()
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
    state  = _pre_sleep_state()
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
    state  = _post_night_state()
    entry  = _sample_journal_entry()
    return orchestrator.run_morning_reflection(state, entry, mode=mode)


# ---------------------------------------------------------------------------
# Sample state builders — isolated here so routes stay readable.
# These will be replaced by real session loading once a store is added.
# ---------------------------------------------------------------------------

def _noisy_light_sleep_state() -> SharedState:
    """Light sleep with 62 dB street noise — exercises disturbance + intervention path."""
    return SharedState(
        user_id="user_pipeline_demo",
        preferences=UserPreferences(
            goals=["reduce awakenings", "maximize deep sleep"],
            preferred_audio=["brown_noise", "rain", "breathing_pace"],
            disliked_audio=["white_noise"],
            target_wake_time="07:00",
            intervention_aggressiveness="medium",
        ),
        latest_sensor=SensorSnapshot(
            timestamp="2026-03-14T03:15:00+00:00",
            heart_rate=68.0,
            hrv=44.0,
            movement=0.28,
            noise_db=62.0,
            light_level=0.04,
            breathing_rate=16.5,
        ),
        nightly_plan=NightlyPlan(
            target_bedtime="23:00",
            target_wake_time="07:00",
            sleep_goal="reduce_awakenings",
            notes="Pre-sleep plan: focus on masking environmental noise.",
            preferred_intervention_order=["brown_noise", "rain", "breathing_pace"],
        ),
    )


def _pre_sleep_state() -> SharedState:
    """User is about to sleep; has journal history for the strategist to learn from."""
    return SharedState(
        user_id="user_plan_demo",
        preferences=UserPreferences(
            goals=["maximize deep sleep", "feel rested"],
            preferred_audio=["brown_noise", "rain"],
            disliked_audio=["white_noise"],
            target_wake_time="06:30",
            intervention_aggressiveness="low",
        ),
        journal_history=[
            JournalEntry(
                date="2026-03-13",
                rested_score=6,
                awakenings_reported=2,
                notes="Woke up twice, once from a car outside.",
            ),
            JournalEntry(
                date="2026-03-12",
                rested_score=8,
                awakenings_reported=0,
                notes="Great night, used brown noise the whole time.",
            ),
        ],
    )


def _post_night_state() -> SharedState:
    """State at the end of a completed sleep session, ready for morning reflection."""
    return SharedState(
        user_id="user_journal_demo",
        preferences=UserPreferences(
            goals=["reduce awakenings"],
            preferred_audio=["brown_noise"],
            disliked_audio=[],
            target_wake_time="07:00",
            intervention_aggressiveness="medium",
        ),
        sleep_state=SleepState(
            phase="light",
            confidence=0.60,
            wake_risk=0.35,
            disturbance_reason=None,
        ),
        hypotheses=[
            {
                "source": "sensor_interpreter",
                "signals": {
                    "hr_status": "normal",
                    "hrv_quality": "moderate",
                    "movement_level": "minimal",
                    "noise_alert": "quiet",
                    "light_status": "dark",
                    "breathing_status": "normal",
                },
            }
        ],
        journal_history=[
            JournalEntry(
                date="2026-03-13",
                rested_score=5,
                awakenings_reported=3,
                notes="Rough night. Kept waking up for no obvious reason.",
            ),
        ],
    )


def _sample_journal_entry() -> JournalEntry:
    return JournalEntry(
        date="2026-03-14",
        rested_score=7,
        awakenings_reported=1,
        notes="Much better. Brown noise helped. One brief awakening around 4am.",
    )


def _deep_sleep_state() -> SharedState:
    """Quiet deep-sleep state — used by /state/example as a structural reference."""
    return SharedState(
        user_id="user_demo_001",
        preferences=UserPreferences(
            goals=["maximize deep sleep", "reduce awakenings"],
            preferred_audio=["brown_noise", "rain"],
            disliked_audio=["white_noise"],
            target_wake_time="07:00",
            intervention_aggressiveness="medium",
        ),
        latest_sensor=SensorSnapshot(
            timestamp="2026-03-14T02:31:00+00:00",
            heart_rate=56.0,
            hrv=72.0,
            movement=0.05,
            noise_db=38.0,
            light_level=0.02,
            breathing_rate=13.5,
        ),
        sleep_state=SleepState(
            phase="deep",
            confidence=0.78,
            wake_risk=0.12,
            disturbance_reason=None,
        ),
        active_intervention=ActiveIntervention(
            type="brown_noise",
            intensity=0.35,
            started_at="2026-03-14T02:10:00+00:00",
            rationale="wake_risk=0.12; disturbance=none; aggressiveness=medium",
        ),
        nightly_plan=NightlyPlan(
            target_bedtime="23:00",
            target_wake_time="07:00",
            sleep_goal="deep_sleep_focus",
            notes="Auto-generated plan. Goals: maximize deep sleep, reduce awakenings.",
            preferred_intervention_order=["brown_noise", "rain", "breathing_pace"],
        ),
        hypotheses=[
            {
                "source": "sensor_interpreter",
                "signals": {
                    "hr_status": "low_resting",
                    "hrv_quality": "excellent",
                    "movement_level": "still",
                    "noise_alert": "quiet",
                    "light_status": "dark",
                    "breathing_status": "normal",
                },
            }
        ],
        journal_history=[
            JournalEntry(
                date="2026-03-13",
                rested_score=7,
                awakenings_reported=1,
                notes="Felt pretty good. One brief awakening around 3am.",
            ),
            JournalEntry(
                date="2026-03-12",
                rested_score=5,
                awakenings_reported=3,
                notes="Noisy street outside. Brown noise helped a bit.",
            ),
        ],
    )
