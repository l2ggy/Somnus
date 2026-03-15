"""Scenario fixtures for Team C regression checks.

These fixtures complement the basic demo states with richer disturbance patterns
and explicit qualitative expectations. Team C can run them through deterministic
orchestrator flows and assert behavior trends (not brittle exact numbers).
"""

from dataclasses import dataclass

from app.models.userstate import (
    ActiveIntervention,
    JournalEntry,
    NightlyPlan,
    SensorSnapshot,
    SharedState,
    SleepState,
    UserPreferences,
)


@dataclass(frozen=True)
class ExpectedQualitativeOutcome:
    """Expected trend-level outcomes for a scenario regression check."""

    phase_trend: str
    wake_risk_movement: str
    intervention_behavior: str


@dataclass(frozen=True)
class ScenarioFixture:
    """Scenario state plus qualitative expectations."""

    key: str
    title: str
    description: str
    state: SharedState
    expected: ExpectedQualitativeOutcome


def stable_deep_sleep_baseline_fixture() -> ScenarioFixture:
    """Calm baseline where rules should preserve deep sleep with minimal action."""

    return ScenarioFixture(
        key="stable_deep_sleep_baseline",
        title="Stable deep sleep baseline",
        description="Low-noise, low-movement, high-HRV conditions with no active disturbance.",
        state=SharedState(
            user_id="user_team_c_baseline",
            preferences=UserPreferences(
                goals=["maximize deep sleep", "reduce awakenings"],
                preferred_audio=["brown_noise", "rain"],
                disliked_audio=["white_noise"],
                target_wake_time="07:00",
                intervention_aggressiveness="medium",
            ),
            latest_sensor=SensorSnapshot(
                timestamp="2026-03-15T01:40:00+00:00",
                heart_rate=54.0,
                hrv=78.0,
                movement=0.03,
                noise_db=35.0,
                light_level=0.01,
                breathing_rate=12.8,
            ),
            sleep_state=SleepState(phase="deep", confidence=0.84, wake_risk=0.1),
            active_intervention=ActiveIntervention(type="none", intensity=0.0),
        ),
        expected=ExpectedQualitativeOutcome(
            phase_trend="Remain in deep sleep or improve confidence toward stable deep.",
            wake_risk_movement="Stay low or decay slightly.",
            intervention_behavior="No intervention or quickly clear to type='none' with zero intensity.",
        ),
    )


def repeated_noise_spikes_fixture() -> ScenarioFixture:
    """Scenario with high ambient noise bursts that should keep pressure on wake risk."""

    return ScenarioFixture(
        key="repeated_noise_spikes",
        title="Repeated noise spikes",
        description="Recent hypotheses indicate recurring external noise disruptions.",
        state=SharedState(
            user_id="user_team_c_noise_spikes",
            preferences=UserPreferences(
                goals=["reduce awakenings"],
                preferred_audio=["brown_noise", "rain", "waves"],
                disliked_audio=[],
                target_wake_time="06:45",
                intervention_aggressiveness="high",
            ),
            latest_sensor=SensorSnapshot(
                timestamp="2026-03-15T03:21:00+00:00",
                heart_rate=66.0,
                hrv=39.0,
                movement=0.2,
                noise_db=69.0,
                light_level=0.03,
                breathing_rate=16.2,
            ),
            sleep_state=SleepState(phase="light", confidence=0.71, wake_risk=0.42),
            hypotheses=[
                {
                    "source": "sensor_interpreter",
                    "timestamp": "2026-03-15T03:20:30+00:00",
                    "signals": {"noise_alert": "loud", "noise_pattern": "spiking"},
                }
            ],
        ),
        expected=ExpectedQualitativeOutcome(
            phase_trend="Trend toward lighter/fragmented sleep unless noise subsides.",
            wake_risk_movement="Increase or remain elevated due to repeated spikes.",
            intervention_behavior="Sustain or ramp masking audio (e.g., brown_noise/rain) at medium-high intensity.",
        ),
    )


def intermittent_light_leakage_fixture() -> ScenarioFixture:
    """Scenario with transient light disturbances during otherwise moderate sleep."""

    return ScenarioFixture(
        key="intermittent_light_leakage",
        title="Intermittent light leakage",
        description="Light level periodically rises above dark-room baseline while noise stays mostly quiet.",
        state=SharedState(
            user_id="user_team_c_light_leak",
            preferences=UserPreferences(
                goals=["maximize deep sleep"],
                preferred_audio=["rain", "brown_noise"],
                disliked_audio=[],
                target_wake_time="07:15",
                intervention_aggressiveness="medium",
            ),
            latest_sensor=SensorSnapshot(
                timestamp="2026-03-15T02:52:00+00:00",
                heart_rate=61.0,
                hrv=52.0,
                movement=0.11,
                noise_db=41.0,
                light_level=0.24,
                breathing_rate=14.9,
            ),
            sleep_state=SleepState(phase="light", confidence=0.66, wake_risk=0.31),
        ),
        expected=ExpectedQualitativeOutcome(
            phase_trend="Hover in light sleep with occasional drift away from deep while leakage persists.",
            wake_risk_movement="Modest upward pressure; not as steep as repeated loud-noise spikes.",
            intervention_behavior="Prefer gentle-to-moderate intervention; avoid maximum intensity unless additional disturbances emerge.",
        ),
    )


def high_movement_elevated_hr_compound_fixture() -> ScenarioFixture:
    """Compound physiological disturbance with movement and elevated heart rate."""

    return ScenarioFixture(
        key="high_movement_elevated_hr_compound",
        title="High movement + elevated HR compound disturbance",
        description="Movement and heart-rate strain co-occur, representing restlessness/arousal.",
        state=SharedState(
            user_id="user_team_c_compound",
            preferences=UserPreferences(
                goals=["reduce awakenings", "feel rested"],
                preferred_audio=["breathing_pace", "brown_noise", "rain"],
                disliked_audio=[],
                target_wake_time="06:30",
                intervention_aggressiveness="high",
            ),
            latest_sensor=SensorSnapshot(
                timestamp="2026-03-15T04:08:00+00:00",
                heart_rate=84.0,
                hrv=28.0,
                movement=0.77,
                noise_db=49.0,
                light_level=0.07,
                breathing_rate=19.4,
            ),
            sleep_state=SleepState(phase="light", confidence=0.58, wake_risk=0.55),
            active_intervention=ActiveIntervention(
                type="brown_noise",
                intensity=0.52,
                started_at="2026-03-15T04:03:00+00:00",
                rationale="Existing masking while compound disturbance emerges.",
            ),
        ),
        expected=ExpectedQualitativeOutcome(
            phase_trend="Likely destabilize toward very light sleep/near-awake without response.",
            wake_risk_movement="Rise sharply or remain high under sustained compound disturbance.",
            intervention_behavior="Escalate intervention intensity and consider switching to breathing_pace if preferred/order allows.",
        ),
    )


def sparse_sensor_fields_fixture() -> ScenarioFixture:
    """Sparse-data scenario where most sensor fields are missing."""

    return ScenarioFixture(
        key="missing_sensor_fields_sparse_data",
        title="Missing sensor fields / sparse data",
        description="Only timestamp and one signal are present; rule pipeline should remain robust.",
        state=SharedState(
            user_id="user_team_c_sparse",
            preferences=UserPreferences(
                goals=["reduce awakenings"],
                preferred_audio=["brown_noise"],
                disliked_audio=[],
                target_wake_time="07:00",
                intervention_aggressiveness="low",
            ),
            latest_sensor=SensorSnapshot(
                timestamp="2026-03-15T00:47:00+00:00",
                noise_db=47.0,
            ),
            sleep_state=SleepState(phase="unknown", confidence=0.2, wake_risk=0.22),
        ),
        expected=ExpectedQualitativeOutcome(
            phase_trend="Stay conservative (unknown/light) rather than overconfident phase shifts.",
            wake_risk_movement="Small bounded changes only; no extreme jumps from missing fields.",
            intervention_behavior="Use low-intensity or no intervention unless available evidence crosses thresholds.",
        ),
    )


def journal_trend_improvement_vs_degradation_fixture() -> ScenarioFixture:
    """Morning-reflection fixture showing diverging trends across recent journal entries."""

    return ScenarioFixture(
        key="journal_trend_improvement_vs_degradation",
        title="Journal trend improvement vs degradation",
        description="Recent nights include both worsening and improving outcomes for trend-sensitive planning.",
        state=SharedState(
            user_id="user_team_c_journal_trends",
            preferences=UserPreferences(
                goals=["feel rested", "reduce awakenings"],
                preferred_audio=["brown_noise", "rain", "breathing_pace"],
                disliked_audio=["white_noise"],
                target_wake_time="07:00",
                intervention_aggressiveness="medium",
            ),
            latest_sensor=SensorSnapshot(
                timestamp="2026-03-15T06:40:00+00:00",
                heart_rate=60.0,
                hrv=50.0,
                movement=0.09,
                noise_db=40.0,
                light_level=0.03,
                breathing_rate=14.1,
            ),
            sleep_state=SleepState(phase="light", confidence=0.64, wake_risk=0.28),
            nightly_plan=NightlyPlan(
                target_bedtime="22:45",
                target_wake_time="07:00",
                sleep_goal="stability_focus",
                notes="Previous plan before applying latest journal trend interpretation.",
                preferred_intervention_order=["rain", "brown_noise", "breathing_pace"],
            ),
            journal_history=[
                JournalEntry(
                    date="2026-03-11",
                    rested_score=4,
                    awakenings_reported=4,
                    notes="Poor night, frequent awakenings and long return-to-sleep latency.",
                ),
                JournalEntry(
                    date="2026-03-12",
                    rested_score=5,
                    awakenings_reported=3,
                    notes="Slightly better but still fragmented.",
                ),
                JournalEntry(
                    date="2026-03-13",
                    rested_score=7,
                    awakenings_reported=1,
                    notes="Clear improvement with consistent brown noise.",
                ),
                JournalEntry(
                    date="2026-03-14",
                    rested_score=8,
                    awakenings_reported=1,
                    notes="Another solid night, woke briefly once.",
                ),
            ],
        ),
        expected=ExpectedQualitativeOutcome(
            phase_trend="Not a night-tick phase fixture; used to verify journal trend interpretation quality.",
            wake_risk_movement="Expect downstream planning to target lower baseline risk after recent improvement.",
            intervention_behavior="Maintain effective interventions while avoiding over-aggressive escalation tied to older degraded nights.",
        ),
    )


def team_c_regression_scenarios() -> dict[str, ScenarioFixture]:
    """Return all Team C scenario fixtures keyed by stable machine-readable IDs."""

    fixtures = [
        stable_deep_sleep_baseline_fixture(),
        repeated_noise_spikes_fixture(),
        intermittent_light_leakage_fixture(),
        high_movement_elevated_hr_compound_fixture(),
        sparse_sensor_fields_fixture(),
        journal_trend_improvement_vs_degradation_fixture(),
    ]
    return {fixture.key: fixture for fixture in fixtures}
