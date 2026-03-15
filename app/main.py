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
from app.models.outer_loop import MorningFeedback
from app.models.userstate import JournalEntry, SharedState
from app.outer_loop import run as outer_loop_run
from app.outer_loop import store as outer_store
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

    # Inject outer-loop policy into the plan AFTER the strategist runs.
    # This re-ranks the intervention order and marks any policy-blocked types
    # without overwriting the strategist's sleep_goal or bedtime choices.
    policy = outer_store.load_or_default_policy(req.user_id)
    planned_state = outer_loop_run.apply_policy_to_state(result["state"], policy)

    store.create_session(planned_state)
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


@app.get("/session/{user_id}/journal", tags=["session"])
def session_journal_list(user_id: str):
    """Return a session's journal entries, newest first."""
    state = _require_session(user_id)
    entries = list(reversed(state.journal_history))
    return {"user_id": user_id, "entries": entries}


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


# ---------------------------------------------------------------------------
# Outer loop routes — cross-night learning layer
# ---------------------------------------------------------------------------

@app.post("/session/{user_id}/end", tags=["outer_loop"])
def session_end(
    user_id: str,
    feedback: MorningFeedback | None = None,
    mode: str = Query(default="deterministic", pattern="^(deterministic|gpt)$"),
):
    """
    Finish a sleep session and run the full outer loop pipeline.

    Chains four agents in order:
      1. NightSummaryAgent       — summarises the session from SharedState
      2. InterventionReviewAgent — evaluates which interventions helped/hurt
      3. PolicyUpdateAgent       — updates the persistent per-user policy
      4. NextNightPlannerAgent   — generates a NightlyPlan for tomorrow

    The user policy and night report are persisted to the database.

    Body (optional): MorningFeedback with rested_score, awakenings_reported,
                     intervention_helpful, and notes.  All fields are optional —
                     the outer loop runs correctly with no feedback at all.

    ?mode=deterministic  (default) — all agents use rule-based logic
    ?mode=gpt            — NightSummaryAgent and InterventionReviewAgent add
                           LLM-generated narrative/notes; NextNightPlannerAgent
                           uses GPT for richer plan reasoning

    Returns an OuterLoopReport with all four agent outputs.
    """
    state = _require_session(user_id)
    report = outer_loop_run.run_outer_loop(state, feedback=feedback, mode=mode)
    return report


@app.post("/session/{user_id}/morning-feedback", tags=["outer_loop"])
def session_morning_feedback(
    user_id: str,
    feedback: MorningFeedback,
    mode: str = Query(default="deterministic", pattern="^(deterministic|gpt)$"),
):
    """
    Submit morning feedback and re-run the outer loop with it.

    Use this endpoint when the user submits their rating after the session has
    already ended (e.g. they open the app the morning after).  Replaces any
    previous outer-loop results for the same night.

    This is equivalent to POST /session/{user_id}/end with a feedback body,
    but is semantically distinct — it implies the session has already closed.

    ?mode=deterministic / ?mode=gpt  — same as /end
    """
    state = _require_session(user_id)
    report = outer_loop_run.run_outer_loop(state, feedback=feedback, mode=mode)
    return report


@app.get("/user/{user_id}/policy", tags=["outer_loop"])
def get_user_policy(user_id: str):
    """
    Fetch the current accumulated UserPolicy for a user.

    Returns a fresh zero-state policy if the user has no history yet.
    The policy reflects learning accumulated across all completed nights.
    """
    from app.outer_loop import store as outer_store
    policy = outer_store.load_or_default_policy(user_id)
    return policy


@app.get("/user/{user_id}/next-night-plan", tags=["outer_loop"])
def get_next_night_plan(user_id: str):
    """
    Fetch the next-night plan derived from the user's accumulated policy.

    Re-derives the plan from the stored UserPolicy using the current session's
    preferences.  Returns 404 if the user has no policy history.

    For richer plans, use POST /session/start?mode=gpt which runs the full
    strategist with reflection signals.
    """
    from app.models.outer_loop import NightSummary
    from app.outer_loop.next_night_planner_agent import run as plan_deterministic
    from app.outer_loop.night_summary_agent import _infer_date

    state = _require_session(user_id)
    policy = outer_store.load_policy(user_id)
    if policy is None:
        raise HTTPException(
            status_code=404,
            detail=f"No outer-loop policy for '{user_id}'. Run POST /session/{user_id}/end first.",
        )

    stub_summary = NightSummary(user_id=user_id, date=_infer_date(state), total_ticks=0)
    plan, mode_used = plan_deterministic(policy, stub_summary, state.preferences)
    return {
        "user_id":      user_id,
        "plan":         plan,
        "mode_used":    mode_used,
        "policy_notes": policy.policy_notes,
    }


@app.get("/user/{user_id}/outer-loop-summary", tags=["outer_loop"])
def get_outer_loop_summary(user_id: str):
    """
    Compact read-only snapshot of the outer loop state for a user.

    Returns everything a frontend or demo needs to display the learning layer:
    - preferred / discouraged / blocked interventions (from current policy)
    - rested score trend (last 7 nights)
    - latest pattern hypothesis (GPT-generated, if available)
    - current risk posture and avg rested score
    - most recent night review summary
    - next-night plan summary

    Safe when no prior outer-loop run exists — returns a "no history yet" shape
    rather than 404, so the frontend can show a clean empty state.
    """
    from app.models.outer_loop import NightSummary
    from app.outer_loop.next_night_planner_agent import run as plan_deterministic
    from app.outer_loop.night_summary_agent import _infer_date
    from app.outer_loop.run import _BLOCK_THRESHOLD, _DEPRIORITIZE_THRESHOLD, _PREFER_THRESHOLD

    # Policy — always returns a usable object (zero-state default if no history)
    policy = outer_store.load_or_default_policy(user_id)

    scores = policy.intervention_scores
    preferred      = [iv for iv, s in scores.items() if s >= _PREFER_THRESHOLD]
    discouraged    = [iv for iv, s in scores.items() if _DEPRIORITIZE_THRESHOLD >= s > _BLOCK_THRESHOLD]
    blocked        = [iv for iv, s in scores.items() if s <= _BLOCK_THRESHOLD]

    preferred.sort(key=lambda iv: -scores[iv])

    # Most recent night report
    last_report = None
    last_dates = outer_store.list_night_reports(user_id, limit=1)
    if last_dates:
        raw = outer_store.load_night_report(user_id, last_dates[0])
        if raw:
            rev = raw.get("review", {})
            last_report = {
                "date":                rev.get("date"),
                "overall_quality":     rev.get("overall_night_quality"),
                "primary_disturbance": rev.get("primary_disturbance"),
                "verdicts": [
                    {"type": v["type"], "verdict": v["verdict"], "confidence": v["confidence"]}
                    for v in rev.get("verdicts", [])
                ],
            }

    # Next-night plan (derived from policy + current session prefs, or None)
    next_plan_summary = None
    session_state = store.get_session(user_id)
    if session_state and policy.nights_tracked > 0:
        stub = NightSummary(user_id=user_id, date=_infer_date(session_state), total_ticks=0)
        plan, _ = plan_deterministic(policy, stub, session_state.preferences)
        next_plan_summary = {
            "sleep_goal":                plan.sleep_goal,
            "preferred_intervention_order": plan.preferred_intervention_order,
            "target_wake_time":          plan.target_wake_time,
        }

    return {
        "user_id":              user_id,
        "nights_tracked":       policy.nights_tracked,
        "risk_posture":         policy.risk_posture,
        "avg_rested_score":     policy.avg_rested_score,
        "recent_rested_scores": policy.recent_rested_scores[-7:],
        "preferred_interventions":   preferred,
        "discouraged_interventions": discouraged,
        "blocked_interventions":     blocked,
        "score_snapshot":       {iv: round(s, 3) for iv, s in sorted(scores.items())},
        "latest_pattern_hypothesis": policy.pattern_hypothesis or None,
        "last_review":          last_report,
        "next_night_plan":      next_plan_summary,
        "has_history":          policy.nights_tracked > 0,
    }


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
