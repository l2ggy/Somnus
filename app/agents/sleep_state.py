"""
Sleep State Agent
-----------------
First deterministic pass: classify the current sleep phase using a multi-signal
heuristic rule set.  This is the core inference step — every downstream agent
(disturbance, intervention) depends on what this agent decides.

Signal priority (most → least discriminative without EEG):
  1. Movement       — the strongest single predictor from wearables
  2. Heart rate     — secondary; must be read together with movement
  3. HRV            — deep/REM proxy; high HRV = parasympathetic dominance
  4. Breathing rate — slow + regular → deep; slightly elevated → possible REM
  5. Light level    — environmental hint; bright light strongly suggests awake
  6. Noise          — contextual; loud noise raises wake_risk but doesn't set phase

Phase taxonomy:
  awake   → movement high OR light bright OR hr elevated
  deep    → movement still, hr low, hrv high, breathing slow
  rem     → movement minimal, hrv good, breathing slightly elevated (active dreaming)
  light   → between deep and awake; also the default when signals conflict
  unknown → no sensor data

LLM integration plan: the phase + confidence from this agent will seed the LLM
prompt.  The model can then override with nuanced reasoning if signals conflict.
"""

from app.models.userstate import SharedState, SleepState


def run(state: SharedState) -> SharedState:
    """
    Infer the current sleep phase from the latest sensor snapshot.

    Preserves the existing disturbance_reason so the disturbance agent (which
    runs next) can compare old vs. new.

    Args:
        state: SharedState with latest_sensor populated (ideally also with
               sensor_interpreter hypotheses already appended).

    Returns:
        Updated SharedState with sleep_state recomputed.
    """
    sensor = state.latest_sensor
    if sensor is None:
        return state

    phase, confidence, wake_risk = _classify(sensor)

    new_sleep_state = SleepState(
        phase=phase,
        confidence=round(confidence, 2),
        wake_risk=round(wake_risk, 2),
        # Carry forward any disturbance reason the previous tick set; the
        # disturbance agent will update it on this tick.
        disturbance_reason=state.sleep_state.disturbance_reason,
    )

    return state.model_copy(update={"sleep_state": new_sleep_state})


# ---------------------------------------------------------------------------
# Multi-signal heuristic classifier
# ---------------------------------------------------------------------------

def _classify(sensor) -> tuple[str, float, float]:
    """
    Return (phase, confidence, wake_risk) using a priority-ordered rule set.

    Rules are intentionally conservative — we'd rather label something 'light'
    than confidently misclassify deep sleep as REM.  A proper ML model trained
    on PSG-labelled data will replace this later.
    """
    hr         = sensor.heart_rate    # bpm; None if unavailable
    hrv        = sensor.hrv           # ms; None if unavailable
    movement   = sensor.movement      # 0–1; None if unavailable
    br         = sensor.breathing_rate  # bpm; None if unavailable
    light      = sensor.light_level   # 0–1; None if unavailable
    noise      = sensor.noise_db      # dB; None if unavailable

    # --- Rule 1: Bright light almost certainly means awake ---
    if light is not None and light > 0.5:
        return "awake", 0.85, 0.9

    # --- Rule 2: High movement → awake or transitioning ---
    if movement is not None and movement > 0.55:
        return "awake", 0.80, 0.85

    # --- Rule 3: Elevated HR with some movement → awake ---
    if hr is not None and hr > 85:
        if movement is None or movement > 0.15:
            return "awake", 0.70, 0.80

    # --- Rule 4: Deep sleep — all quieting signals align ---
    #   Low HR + very low movement + good HRV + slow breathing
    deep_hr        = hr is None or hr < 60
    deep_movement  = movement is None or movement < 0.1
    deep_hrv       = hrv is None or hrv >= 55
    deep_breathing = br is None or br <= 15

    if deep_hr and deep_movement and deep_hrv and deep_breathing:
        # Confidence scales with how many signals we actually have.
        n_signals = sum([hr is not None, hrv is not None,
                         movement is not None, br is not None])
        confidence = 0.55 + 0.08 * n_signals   # 0.55 → 0.87
        return "deep", min(confidence, 0.87), 0.10

    # --- Rule 5: REM proxy — minimal movement + elevated HRV + slightly faster
    #   breathing (active dreaming increases respiratory rate slightly) ---
    rem_movement  = movement is None or movement < 0.15
    rem_hrv       = hrv is not None and hrv >= 50
    rem_breathing = br is None or (14 <= br <= 19)

    if rem_movement and rem_hrv and rem_breathing:
        return "rem", 0.58, 0.20

    # --- Rule 6: Loud noise raises wake_risk even in light sleep ---
    noise_penalty = 0.0
    if noise is not None and noise > 55:
        noise_penalty = min(0.20, (noise - 55) / 50)   # scales 0 → 0.20

    # --- Default: light sleep ---
    return "light", 0.55, round(0.30 + noise_penalty, 2)
