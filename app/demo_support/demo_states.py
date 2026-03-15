"""Demo/user-flow fixtures owned by product/frontend workstream.

These helpers power example endpoints used in demos and UI wiring.
They are isolated from runtime orchestration so presentation changes do not
conflict with backend orchestration or sleep-intelligence logic.
"""

from app.models.userstate import (
    ActiveIntervention,
    JournalEntry,
    NightlyPlan,
    SensorSnapshot,
    SharedState,
    SleepState,
    UserPreferences,
)


def noisy_light_sleep_state() -> SharedState:
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


def pre_sleep_state() -> SharedState:
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


def post_night_state() -> SharedState:
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


def sample_journal_entry() -> JournalEntry:
    return JournalEntry(
        date="2026-03-14",
        rested_score=7,
        awakenings_reported=1,
        notes="Much better. Brown noise helped. One brief awakening around 4am.",
    )


def deep_sleep_state() -> SharedState:
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
