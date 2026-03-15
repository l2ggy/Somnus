"""
Somnus Orchestrator
-------------------
The backend core. Owns the three lifecycle phases of a sleep session and is
the only place in the codebase that knows the order agents run in.

Lifecycle phases
----------------
1. Pre-sleep planning  (once, before the user goes to bed)
   run_pre_sleep_planning → strategist
   Produces a NightlyPlan the tick agents reference all night.

2. Night tick          (repeatedly, every ~30–60 seconds during sleep)
   run_night_tick → sensor_interpreter → sleep_state → disturbance → intervention
   Each tick reads the latest sensor snapshot and decides whether to intervene.

3. Morning reflection  (once, after the user wakes)
   run_morning_reflection → journal_reflection
   Processes the user's subjective report and stores insights for the next night.

Why intake is not inside the tick
----------------------------------
intake.run() converts a raw dict payload into a SensorSnapshot and writes it
to SharedState.latest_sensor.  That conversion happens *before* the tick — at
the API boundary where raw sensor data arrives via POST.  By the time
run_night_tick is called, latest_sensor is already a validated SensorSnapshot.
The convenience function ingest_sensor() wraps intake for callers that do have
a raw dict (e.g. the future POST /sensor endpoint).

Why orchestration stays deterministic
--------------------------------------
This layer is intentionally rule-based with no randomness or external calls.
That means:
  - Results are reproducible given the same input state.
  - The pipeline can be unit-tested without mocks.
  - When LLM agents are added, they slot in as additional steps whose outputs
    are validated and written back into SharedState — the orchestrator stays
    in control of sequencing and never lets an agent mutate state directly.

How LLM agents will plug in
----------------------------
Future agents (e.g. a GPT-powered strategist, a narrative reflection writer)
will have the same signature: SharedState → SharedState.  The orchestrator
will call them the same way it calls rule-based agents today, then inspect
the returned state for required fields.  If a required field is missing or
invalid the orchestrator falls back to the deterministic agent.  This makes
the LLM layer opt-in and safe to ship incrementally.
"""

from app.agents import (
    intake,
    sensor_interpreter,
    sleep_state as sleep_state_agent,
    disturbance as disturbance_agent,
    intervention as intervention_agent,
    strategist,
    journal_reflection,
)
from app.models.userstate import JournalEntry, SharedState


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def ingest_sensor(state: SharedState, raw: dict) -> SharedState:
    """
    Convert a raw sensor payload dict into a SensorSnapshot and attach it
    to SharedState.latest_sensor.

    Call this at the API boundary (e.g. POST /sensor) before run_night_tick.
    The tick pipeline expects latest_sensor to already be populated.

    Args:
        state: Current shared state.
        raw:   Dict with any subset of: timestamp, heart_rate, hrv, movement,
               noise_db, light_level, breathing_rate.

    Returns:
        Updated SharedState with latest_sensor set.
    """
    return intake.run(state, raw)


def run_night_tick(state: SharedState) -> dict:
    """
    Run one real-time tick of the night pipeline.

    Pipeline order:
      sensor_interpreter → sleep_state → disturbance → intervention

    Each agent receives the output state of the previous agent, so later
    agents always see the most up-to-date values.

    Args:
        state: SharedState with latest_sensor populated.  If latest_sensor
               is None the tick is aborted and a structured error is returned.

    Returns:
        {
            "ok":      bool,
            "state":   SharedState,     # updated state after all agents ran
            "trace":   list[dict],      # per-agent summary of what changed
            "warnings": list[str],      # non-fatal issues (e.g. missing fields)
        }
    """
    warnings: list[str] = []

    if state.latest_sensor is None:
        return {
            "ok": False,
            "error": "latest_sensor is None — call ingest_sensor() before run_night_tick()",
            "state": state,
            "trace": [],
            "warnings": warnings,
        }

    # Check for any partially missing sensor fields and warn but continue.
    sensor = state.latest_sensor
    missing = [f for f in ("heart_rate", "hrv", "movement", "noise_db",
                           "light_level", "breathing_rate")
               if getattr(sensor, f) is None]
    if missing:
        warnings.append(f"sensor fields missing (agents will use defaults): {missing}")

    # -----------------------------------------------------------------------
    # Agent pipeline — each step is one immutable state transform.
    # -----------------------------------------------------------------------

    s1 = sensor_interpreter.run(state)
    s2 = sleep_state_agent.run(s1)
    s3 = disturbance_agent.run(s2)
    s4 = intervention_agent.run(s3)

    trace = [
        _trace_step(1, "sensor_interpreter", _sensor_interpreter_summary(s1)),
        _trace_step(2, "sleep_state",        _sleep_state_summary(s2)),
        _trace_step(3, "disturbance",        _disturbance_summary(s3)),
        _trace_step(4, "intervention",       _intervention_summary(s4)),
    ]

    return {
        "ok":       True,
        "state":    s4,
        "trace":    trace,
        "warnings": warnings,
    }


def run_pre_sleep_planning(state: SharedState, mode: str = "deterministic") -> dict:
    """
    Run the strategist to build tonight's NightlyPlan.

    Call this once when the user signals they're about to go to bed.
    The resulting NightlyPlan is carried in SharedState and read by the
    intervention agent on every subsequent tick.

    Args:
        state: SharedState with preferences and journal_history populated.
        mode:  "deterministic" (default) or "gpt".
               In GPT mode, if the LLM call fails the response will contain
               mode_used="fallback" and the deterministic plan is used instead.

    Returns:
        {
            "state":     SharedState,  # updated with nightly_plan set
            "mode_used": str,          # "deterministic", "gpt", or "fallback"
        }
    """
    if mode == "gpt":
        updated, mode_used = strategist.run_gpt(state)
    else:
        updated   = strategist.run(state)
        mode_used = "deterministic"

    return {"state": updated, "mode_used": mode_used}


def run_morning_reflection(
    state: SharedState,
    entry: JournalEntry,
    mode: str = "deterministic",
) -> dict:
    """
    Process the user's morning journal entry and generate a reflection.

    Call this once after the sleep session ends when the user submits their
    morning report.  The reflection is appended to hypotheses and the new
    journal entry is appended to journal_history — both persist into the next
    night's planning cycle.

    Args:
        state: SharedState from the completed sleep session.
        entry: JournalEntry submitted by the user.
        mode:  "deterministic" (default) or "gpt".
               In GPT mode, if the LLM call fails the response will contain
               mode_used="fallback" and the deterministic reflection is used.

    Returns:
        {
            "state":      SharedState,  # updated with new entry + reflection
            "reflection": dict,         # extracted reflection summary
            "mode_used":  str,          # "deterministic", "gpt", or "fallback"
        }
    """
    if mode == "gpt":
        updated, mode_used = journal_reflection.run_gpt(state, entry)
    else:
        updated   = journal_reflection.run(state, entry)
        mode_used = "deterministic"

    reflection = _extract_reflection(updated)

    return {
        "state":      updated,
        "reflection": reflection,
        "mode_used":  mode_used,
    }


# ---------------------------------------------------------------------------
# Trace helpers — each returns a small summary dict for the pipeline trace.
# These are intentionally narrow: only the fields that changed in that step.
# ---------------------------------------------------------------------------

def _trace_step(step: int, agent: str, output: dict) -> dict:
    return {"step": step, "agent": agent, "output": output}


def _sensor_interpreter_summary(state: SharedState) -> dict:
    """Return a compact but consistent summary for the latest sensor tick."""
    for h in reversed(state.hypotheses):
        if h.get("source") != "sensor_interpreter":
            continue

        signal_hypotheses = h.get("signal_hypotheses", [])
        return {
            "tick_timestamp": h.get("timestamp"),
            "tick_source": h.get("tick_source"),
            "signals": h.get("signal_summary", h.get("signals", {})),
            "signal_hypotheses": [
                {
                    "signal": sh.get("signal"),
                    "severity": sh.get("severity"),
                    "label": sh.get("label"),
                }
                for sh in signal_hypotheses
            ],
        }
    return {}


def _sleep_state_summary(state: SharedState) -> dict:
    return {
        "phase":      state.sleep_state.phase,
        "confidence": state.sleep_state.confidence,
        "wake_risk":  state.sleep_state.wake_risk,
    }


def _disturbance_summary(state: SharedState) -> dict:
    return {
        "disturbance_reason": state.sleep_state.disturbance_reason,
        "wake_risk":          state.sleep_state.wake_risk,
    }


def _intervention_summary(state: SharedState) -> dict:
    iv = state.active_intervention
    return {
        "type":      iv.type,
        "intensity": iv.intensity,
        "rationale": iv.rationale,
    }


def _extract_reflection(state: SharedState) -> dict:
    """Return the most recent journal_reflection hypothesis payload."""
    for h in reversed(state.hypotheses):
        if h.get("source") == "journal_reflection":
            return h.get("reflection", {})
    return {}
