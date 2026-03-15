"""LLM planner agent: ranks interventions using LLM-generated trajectories.

Role in the system
------------------
This module is the LLM-path equivalent of agents/world_model_agent.py.
The rule-based planner (choose_best_action) remains the baseline/fallback.
This planner replaces Monte Carlo simulation with LLM-generated futures:

  Rule-based path                   LLM path
  ─────────────────────────         ──────────────────────────────────
  simulate() × n_rollouts    →      generate_action_trajectories()
  cumulative sleep_reward()  →      score_action_bundle() via critic
  mean score per action      →      probability-weighted expected score
  rank + pick best           →      rank + pick best

Planning loop (per action)
--------------------------
  1. build_world_model_prompt(state, action, ...)
  2. llm_client(prompt)                 ← vendor-agnostic callable
  3. parse_action_trajectory_bundle()   ← validate + structure
  4. score_action_bundle()              ← critic scores via reward.py
  5. rank all actions by score
  6. return best
"""

from __future__ import annotations

from typing import Any

from simulation.actions import ALL_ACTIONS
from simulation.sleep_state import SleepState
from simulation.user_profile import UserProfile
from agents.llm_world_model_agent import ActionTrajectoryBundle, generate_action_trajectories
from agents.trajectory_critic import score_action_bundle


# ── Single-action evaluation ──────────────────────────────────────────────────

def evaluate_action_with_llm(
    state: SleepState,
    action: str,
    user_profile: UserProfile | None = None,
    horizon: int = 5,
    llm_client: Any | None = None,
) -> tuple[ActionTrajectoryBundle, float]:
    """Generate LLM trajectories for one action and score them.

    Args:
        state:        Current sleep state.
        action:       The intervention to evaluate (must be in ALL_ACTIONS).
        user_profile: Optional user preferences applied at scoring time.
        horizon:      Number of future steps the LLM should imagine.
        llm_client:   Callable (prompt: str) -> str. Required.

    Returns:
        (bundle, score) — the raw trajectory bundle and its expected score.

    Raises:
        NotImplementedError: If llm_client is None.
        ValueError:          If the LLM response is malformed.
    """
    if llm_client is None:
        raise NotImplementedError(
            "llm_client is required for the LLM planner path. "
            "Wrap your LLM SDK into a callable (prompt: str) -> str and pass it as llm_client=. "
            "To use the rule-based fallback, call choose_best_action() from "
            "agents/world_model_agent.py instead."
        )

    bundle = generate_action_trajectories(
        state,
        action,
        user_profile=user_profile,
        horizon=horizon,
        llm_client=llm_client,
    )
    score = score_action_bundle(
        bundle,
        user_profile=user_profile,
        starting_time_until_alarm=state.time_until_alarm,
    )
    return bundle, score


# ── Full ranking ──────────────────────────────────────────────────────────────

def rank_actions_with_llm(
    state: SleepState,
    user_profile: UserProfile | None = None,
    horizon: int = 5,
    llm_client: Any | None = None,
) -> list[tuple[str, float, ActionTrajectoryBundle]]:
    """Evaluate every available action with the LLM and return them ranked best-first.

    Each action gets its own LLM call. The critic converts the resulting
    trajectory bundles into scalar scores (probability-weighted expected reward
    + optional preference adjustment). Actions are then sorted descending.

    Args:
        state:        Current sleep state — the starting point for all futures.
        user_profile: Optional user preferences mixed into scoring.
        horizon:      Steps the LLM imagines per trajectory.
        llm_client:   Callable (prompt: str) -> str. Required.

    Returns:
        List of (action, score, bundle) tuples, sorted descending by score.

    Raises:
        NotImplementedError: If llm_client is None.
        ValueError:          If any LLM response is malformed.
    """
    results: list[tuple[str, float, ActionTrajectoryBundle]] = []

    for action in ALL_ACTIONS:
        bundle, score = evaluate_action_with_llm(
            state,
            action,
            user_profile=user_profile,
            horizon=horizon,
            llm_client=llm_client,
        )
        results.append((action, score, bundle))

    results.sort(key=lambda row: row[1], reverse=True)
    return results


# ── Best-action selection ─────────────────────────────────────────────────────

def choose_best_action_with_llm(
    state: SleepState,
    user_profile: UserProfile | None = None,
    horizon: int = 5,
    llm_client: Any | None = None,
) -> tuple[str, float, ActionTrajectoryBundle]:
    """Return the highest-scoring action according to the LLM world model.

    Runs rank_actions_with_llm(...) and returns the top result.
    The full ranked list is discarded here; call rank_actions_with_llm()
    directly if you need the complete ordering for a demo or debug session.

    Args:
        state:        Current sleep state.
        user_profile: Optional user preferences.
        horizon:      Steps the LLM imagines per trajectory.
        llm_client:   Callable (prompt: str) -> str. Required.

    Returns:
        (best_action, best_score, best_bundle)

    Raises:
        NotImplementedError: If llm_client is None.
        ValueError:          If any LLM response is malformed.
    """
    ranked = rank_actions_with_llm(
        state,
        user_profile=user_profile,
        horizon=horizon,
        llm_client=llm_client,
    )
    return ranked[0]


# ── Example usage (run directly) ─────────────────────────────────────────────

def _example() -> None:
    """Illustrates the intended usage pattern.

    Run directly:  python -m agents.llm_planner_agent

    The llm_client below uses the Anthropic SDK. Swap in any (prompt)->str
    callable to use a different backend.
    """
    from simulation.sleep_state import SleepState
    from simulation.user_profile import UserProfile

    state = SleepState(
        sleep_stage="light",
        sleep_depth=0.42,
        movement=0.30,
        noise_level=0.50,
        time_until_alarm=180,
    )

    profile = UserProfile(
        action_preferences={
            "play_brown_noise":  0.8,
            "play_rain":        -0.5,
            "breathing_pacing":  0.3,
            "gradual_wake":     -1.0,
            "do_nothing":        0.0,
        }
    )

    # Wire up a real LLM client — uncomment whichever backend you have:

    # --- Anthropic ---
    # import anthropic
    # _client = anthropic.Anthropic()
    # def llm_client(prompt: str) -> str:
    #     msg = _client.messages.create(
    #         model="claude-opus-4-6",
    #         max_tokens=2048,
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     return msg.content[0].text

    # --- OpenAI-compatible ---
    # from openai import OpenAI
    # _client = OpenAI()
    # def llm_client(prompt: str) -> str:
    #     resp = _client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     return resp.choices[0].message.content

    # Replace this stub with one of the clients above:
    llm_client = None  # → will raise NotImplementedError, showing the helpful message

    try:
        print("=== rank_actions_with_llm ===")
        ranked = rank_actions_with_llm(state, user_profile=profile, horizon=5, llm_client=llm_client)
        for action, score, bundle in ranked:
            n_traj = len(bundle.trajectories)
            print(f"  {action:<24} score={score:+.3f}  ({n_traj} trajectories)")
            print(f"    rationale: {bundle.rationale[:80]}...")

        print()
        print("=== choose_best_action_with_llm ===")
        best_action, best_score, best_bundle = choose_best_action_with_llm(
            state, user_profile=profile, horizon=5, llm_client=llm_client
        )
        print(f"  Best action : {best_action}")
        print(f"  Score       : {best_score:+.3f}")
        print(f"  Rationale   : {best_bundle.rationale}")

    except NotImplementedError as e:
        print(f"[Expected in stub mode] {e}")


if __name__ == "__main__":
    _example()
