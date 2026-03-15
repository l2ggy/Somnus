"""Demo/test script for the LLM-powered world model path.

Exercises the full LLM pipeline — prompt building, mock generation, parsing,
critic scoring, and action ranking — without a real API call.

The rule-based simulator (simulation/simulator.py) remains the baseline/fallback.
This script only touches the LLM path:

    SleepState + action
         │
    MockLLMClient (stand-in for a real LLM)
         │
    parse_action_trajectory_bundle()
         │
    score_action_bundle()  ←  trajectory_critic.py
         │
    rank_actions_with_llm()  /  choose_best_action_with_llm()

Run:
    python test_llm_world_model.py
"""

from simulation.sleep_state import SleepState
from simulation.user_profile import UserProfile
from agents.mock_llm_client import MockLLMClient
from agents.llm_planner_agent import rank_actions_with_llm, choose_best_action_with_llm
from agents.llm_world_model_agent import CandidateTrajectory


# ── Formatting helpers ────────────────────────────────────────────────────────

_W = 72  # output width


def _header(title: str) -> None:
    print(f"\n{'═' * _W}")
    print(f"  {title}")
    print(f"{'═' * _W}")


def _section(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(0, _W - len(title) - 6)}")


def _print_state(state: SleepState) -> None:
    print(f"    stage={state.sleep_stage:<5}  "
          f"depth={state.sleep_depth:.2f}  "
          f"movement={state.movement:.2f}  "
          f"noise={state.noise_level:.2f}  "
          f"alarm_in={state.time_until_alarm}min")


def _print_profile(profile: UserProfile | None) -> None:
    if profile is None:
        print("    (no user profile — pure sleep quality scoring)")
        return
    for action, pref in profile.action_preferences.items():
        bar = "+" * int(abs(pref) * 4) if pref > 0 else "-" * int(abs(pref) * 4)
        print(f"    {action:<24} {pref:+.1f}  {bar}")


def _print_ranking(ranked: list) -> None:
    print(f"    {'action':<24} {'score':>7}  trajectories")
    print(f"    {'─'*24} {'─'*7}  {'─'*12}")
    for action, score, bundle in ranked:
        bar = "█" * max(0, int((score + 8) * 1.8))
        print(f"    {action:<24} {score:>+7.3f}  {bar}")


def _print_trajectory(i: int, traj: CandidateTrajectory) -> None:
    print(f"\n    Trajectory {i}  p={traj.probability:.2f}  —  {traj.summary}")
    print(f"      {'step':<5} {'stage':<6} {'depth':<7} {'noise':<7} {'move':<7} {'wake_risk'}")
    print(f"      {'─'*5} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")
    for step in traj.steps:
        print(
            f"      {step.step_index:<5} "
            f"{step.sleep_stage:<6} "
            f"{step.sleep_depth:<7.2f} "
            f"{step.noise_level:<7.2f} "
            f"{step.movement:<7.2f} "
            f"{step.wake_risk:.2f}"
        )


def _run_scenario(
    label: str,
    state: SleepState,
    profile: UserProfile | None,
    client: MockLLMClient,
) -> None:
    """Run the full LLM planning pipeline for one scenario and print results."""
    _header(label)

    _section("Input state")
    _print_state(state)

    _section("User profile")
    _print_profile(profile)

    _section("Action ranking  (LLM path + critic)")
    ranked = rank_actions_with_llm(state, user_profile=profile, horizon=5, llm_client=client)
    _print_ranking(ranked)

    best_action, best_score, best_bundle = ranked[0]

    _section(f"Best action → {best_action}  (score {best_score:+.3f})")
    print(f"    Rationale: {best_bundle.rationale}")

    _section(f"Candidate trajectories for '{best_action}'")
    for i, traj in enumerate(best_bundle.trajectories, 1):
        _print_trajectory(i, traj)


# ── Scenarios ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Run both demo scenarios back to back."""

    client = MockLLMClient(seed=42)

    profile = UserProfile(
        action_preferences={
            "play_brown_noise":  0.8,
            "play_rain":        -0.5,
            "breathing_pacing":  0.3,
            "gradual_wake":     -1.0,
            "do_nothing":        0.0,
        },
        sensitivity_to_noise=1.2,
        intervention_tolerance=1.0,
    )

    # ── Scenario 1: noisy, unstable state mid-night ───────────────────────────
    mid_night = SleepState(
        sleep_stage="light",
        sleep_depth=0.38,
        movement=0.35,
        noise_level=0.60,
        time_until_alarm=210,
    )
    _run_scenario("Scenario 1 — Noisy mid-night  (alarm in 210 min)", mid_night, profile, client)

    # ── Scenario 2: near-alarm, shallow sleep ─────────────────────────────────
    near_alarm = SleepState(
        sleep_stage="light",
        sleep_depth=0.32,
        movement=0.25,
        noise_level=0.20,
        time_until_alarm=22,
    )
    _run_scenario("Scenario 2 — Near alarm  (alarm in 22 min)", near_alarm, profile, client)

    print(f"\n{'═' * _W}\n  Done.\n{'═' * _W}\n")


if __name__ == "__main__":
    main()
