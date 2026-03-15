"""Hybrid planner: LLM-first with automatic rule-based fallback.

Preferred path:   LLM world model  (agents/llm_planner_agent.py)
Fallback path:    Rule-based Monte Carlo planner (agents/world_model_agent.py)

The rule-based simulator is never removed — it exists as a reliable, always-
available baseline. This module protects demo/runtime code from fragile model
failures (network errors, malformed JSON, timeout, empty responses) by
catching any exception from the LLM path and transparently re-running the
rule-based planner instead.

Failures are never silently hidden: the returned result always carries
planner_type so callers can log or display which path was actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulation.sleep_state import SleepState
from simulation.user_profile import UserProfile
from agents.llm_planner_agent import choose_best_action_with_llm
from agents.world_model_agent import choose_best_action


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PlannerResult:
    """Structured result returned by choose_best_action_hybrid.

    Attributes:
        selected_action:  The action the planner recommends.
        score:            Numeric score for that action.
        planner_type:     "llm" if the LLM path succeeded,
                          "rule_based_fallback" if it did not.
        rationale:        LLM rationale string if available, else None.
        bundle:           Full ActionTrajectoryBundle if available, else None.
        fallback_reason:  String representation of the exception that caused
                          the fallback, or None when the LLM path succeeded.
    """

    selected_action: str
    score: float
    planner_type: str                   # "llm" | "rule_based_fallback"
    rationale: str | None = None
    bundle: Any | None = None           # ActionTrajectoryBundle | None
    fallback_reason: str | None = None

    def __str__(self) -> str:
        lines = [
            f"selected_action : {self.selected_action}",
            f"score           : {self.score:+.3f}",
            f"planner_type    : {self.planner_type}",
        ]
        if self.rationale:
            lines.append(f"rationale       : {self.rationale}")
        if self.fallback_reason:
            lines.append(f"fallback_reason : {self.fallback_reason}")
        return "\n".join(lines)


# ── Hybrid planner ─────────────────────────────────────────────────────────────

def choose_best_action_hybrid(
    state: SleepState,
    user_profile: UserProfile | None = None,
    horizon: int = 5,
    llm_client: Any | None = None,
    fallback_steps: int = 10,
    fallback_rollouts: int = 20,
) -> PlannerResult:
    """Try the LLM planner; fall back to the rule-based planner on any failure.

    The LLM path is always attempted first when llm_client is provided.
    Any exception — network error, timeout, malformed JSON, bad stage value,
    empty response — is caught and the rule-based planner is run instead.

    The returned PlannerResult always carries planner_type so callers can
    log or surface which path was used, and fallback_reason preserves the
    original error message for debugging.

    Args:
        state:            Current sleep state.
        user_profile:     Optional user preferences passed to both planners.
        horizon:          Trajectory horizon for the LLM planner (steps).
        llm_client:       Callable (prompt: str) -> str. If None, skips the
                          LLM path and goes directly to the rule-based planner.
        fallback_steps:   Rollout horizon for the rule-based planner.
        fallback_rollouts: Monte Carlo samples for the rule-based planner.

    Returns:
        A PlannerResult with selected_action, score, planner_type, and
        optional rationale / fallback_reason.
    """
    # ── Attempt LLM path ──────────────────────────────────────────────────────
    if llm_client is not None:
        try:
            action, score, bundle = choose_best_action_with_llm(
                state,
                user_profile=user_profile,
                horizon=horizon,
                llm_client=llm_client,
            )
            return PlannerResult(
                selected_action=action,
                score=score,
                planner_type="llm",
                rationale=bundle.rationale,
                bundle=bundle,
            )
        except Exception as exc:  # noqa: BLE001 — intentional broad catch for fallback
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        fallback_reason = "llm_client was not provided; using rule-based planner directly."

    # ── Fallback: rule-based Monte Carlo planner ──────────────────────────────
    action, score = choose_best_action(
        state,
        steps=fallback_steps,
        n_rollouts=fallback_rollouts,
        user_profile=user_profile,
    )
    return PlannerResult(
        selected_action=action,
        score=score,
        planner_type="rule_based_fallback",
        fallback_reason=fallback_reason,
    )


# ── Example (run directly) ────────────────────────────────────────────────────

def _example() -> None:
    """Demonstrates the hybrid planner with both a live and a failing client.

    Run:  python -m agents.hybrid_planner_agent
    """
    from simulation.sleep_state import SleepState
    from simulation.user_profile import UserProfile

    state = SleepState(
        sleep_stage="light",
        sleep_depth=0.42,
        movement=0.30,
        noise_level=0.55,
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

    # ── Case 1: mock client succeeds → planner_type = "llm" ──────────────────
    from agents.mock_llm_client import MockLLMClient
    print("=== Case 1: MockLLMClient (should succeed) ===")
    result = choose_best_action_hybrid(state, user_profile=profile, llm_client=MockLLMClient(seed=42))
    print(result)

    # ── Case 2: broken client → planner_type = "rule_based_fallback" ─────────
    def bad_client(prompt: str) -> str:
        raise ConnectionError("Simulated network failure")

    print("\n=== Case 2: broken client (should fall back) ===")
    result = choose_best_action_hybrid(state, user_profile=profile, llm_client=bad_client)
    print(result)

    # ── Case 3: no client → goes straight to rule-based planner ──────────────
    print("\n=== Case 3: no client (direct fallback) ===")
    result = choose_best_action_hybrid(state, user_profile=profile, llm_client=None)
    print(result)

    # ── Case 4: real endpoint (reads from .env) ───────────────────────────────
    # from agents.openai_compatible_llm_client import OpenAICompatibleLLMClient
    # print("\n=== Case 4: OpenAICompatibleLLMClient (real endpoint) ===")
    # result = choose_best_action_hybrid(
    #     state, user_profile=profile,
    #     llm_client=OpenAICompatibleLLMClient(),
    # )
    # print(result)
    # print(f"\nUsed: {result.planner_type}")
    # if result.planner_type == "rule_based_fallback":
    #     print(f"Reason: {result.fallback_reason}")


if __name__ == "__main__":
    _example()
