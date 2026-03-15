#!/usr/bin/env python3
"""
Outer loop multi-night simulation demo.

Simulates 4 consecutive nights without a running server or database.
Shows how the UserPolicy evolves as one intervention becomes discouraged
(white_noise correlates with poor sleep) and another becomes preferred
(rain correlates with good sleep).

Run from the project root:
    python scripts/demo_outer_loop.py

No .env or API key needed — uses the deterministic outer loop path only.
"""

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.outer_loop import (
    InterventionReview,
    InterventionVerdict,
    MorningFeedback,
    NightSummary,
    InterventionUsage,
    UserPolicy,
)
from app.models.userstate import UserPreferences
from app.outer_loop import policy_update_agent
from app.outer_loop.store import default_policy
from app.outer_loop.run import (
    _build_decision_trace,
    _BLOCK_THRESHOLD,
    _DEPRIORITIZE_THRESHOLD,
    _PREFER_THRESHOLD,
)
from app.outer_loop.next_night_planner_agent import run as plan_next_night


# ---------------------------------------------------------------------------
# Scenario definition — 4 nights
# ---------------------------------------------------------------------------

_PREFS = UserPreferences(
    goals=["reduce awakenings"],
    preferred_audio=["brown_noise", "white_noise", "rain"],
    disliked_audio=[],
    target_wake_time="07:00",
    intervention_aggressiveness="medium",
)

_NIGHTS = [
    # (date, interventions_used, quality, awakenings, rested_score, intervention_helpful, notes)
    (
        "2026-03-11",
        [("white_noise", 12, 0.55), ("rain", 4, 0.42)],
        "fair", 2, 5, None,
        "Night was ok, woke up once",
    ),
    (
        "2026-03-12",
        [("white_noise", 18, 0.60)],
        "poor", 4, 3, False,
        "Kept waking up when white noise shifted volume, felt grating",
    ),
    (
        "2026-03-13",
        [("rain", 20, 0.48), ("white_noise", 3, 0.50)],
        "good", 1, 8, True,
        "Rain sounds helped a lot, slept deeply in middle third",
    ),
    (
        "2026-03-14",
        [("white_noise", 14, 0.58)],
        "poor", 5, 2, False,
        "White noise still bothersome, many awakenings, very tired",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary(date, night_data, policy):
    iv_list_raw, quality, awakenings, rested, helpful, notes = night_data
    iv_usages = [
        InterventionUsage(type=t, ticks_active=ticks, avg_intensity=intensity)
        for t, ticks, intensity in iv_list_raw
    ]
    return NightSummary(
        user_id="demo_user",
        date=date,
        total_ticks=sum(t for _, t, _ in iv_list_raw),
        interventions_used=iv_usages,
        phases_observed=["light", "deep"],
        disturbances_detected=["noise_spike:60dB"] if awakenings > 2 else [],
        peak_wake_risk=0.65 if awakenings > 2 else 0.35,
        avg_wake_risk=0.45 if awakenings > 2 else 0.22,
    )


def _make_review(date, night_data):
    iv_list_raw, quality, awakenings, rested, helpful, notes = night_data

    verdicts = []
    for t, ticks, _ in iv_list_raw:
        total = sum(t2 for _, t2, _ in iv_list_raw)
        prominent = ticks > total // 4
        if helpful is not None:
            v, conf, reason = (
                ("helpful", 0.80, "User reported intervention was helpful.")
                if helpful else
                ("harmful", 0.75, "User reported intervention was not helpful.")
            )
        elif quality == "good":
            v = "helpful" if prominent else "neutral"
            conf = 0.55 if prominent else 0.40
            reason = f"Night was good and {t} was {'prominent' if prominent else 'brief'}."
        elif quality == "poor":
            v = "harmful" if prominent else "neutral"
            conf = 0.45 if prominent else 0.35
            reason = f"Night was poor and {t} was {'prominent' if prominent else 'brief'}."
        else:
            v, conf, reason = "neutral", 0.40, f"Fair night; insufficient signal for {t}."
        verdicts.append(InterventionVerdict(type=t, verdict=v, confidence=conf, reason=reason))

    return InterventionReview(
        date=date,
        overall_night_quality=quality,
        verdicts=verdicts,
        primary_disturbance="noise_spike" if awakenings > 2 else None,
    )


def _make_feedback(night_data):
    _, quality, awakenings, rested, helpful, notes = night_data
    return MorningFeedback(
        rested_score=rested,
        awakenings_reported=awakenings,
        intervention_helpful=helpful,
        notes=notes,
    )


def _print_score_bar(score: float, width: int = 20) -> str:
    """Visual bar: negative left, zero center, positive right."""
    norm = (score + 2.0) / 4.0   # map [-2, 2] → [0, 1]
    filled = int(norm * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:+.3f}"


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation():
    print("=" * 70)
    print("SOMNUS OUTER LOOP — MULTI-NIGHT POLICY EVOLUTION DEMO")
    print("=" * 70)
    print()
    print("Scenario: 4 nights. white_noise correlates with poor sleep.")
    print("          rain consistently follows better outcomes.")
    print("          Watch the policy adapt across nights.")
    print()

    policy = default_policy("demo_user")

    for i, (date, iv_list_raw, quality, awakenings, rested, helpful, notes) in enumerate(_NIGHTS, 1):
        night_data = (iv_list_raw, quality, awakenings, rested, helpful, notes)
        summary  = _make_summary(date, night_data, policy)
        review   = _make_review(date, night_data)
        feedback = _make_feedback(night_data)

        pre_policy = policy
        policy = policy_update_agent.run(pre_policy, review, feedback)

        trace = _build_decision_trace(
            pre_policy=pre_policy,
            post_policy=policy,
            review=review,
            next_plan_dict={"sleep_goal": "balanced_sleep", "preferred_intervention_order": [], "notes": ""},
            step_modes={"summary": "det", "review": "det", "policy": "det", "plan": "det"},
        )

        print(f"── Night {i}: {date} ───────────────────────────────────────────────")
        print(f"   Rested: {rested}/10  |  Awakenings: {awakenings}  |  Quality: {quality.upper()}")
        print(f"   Feedback notes: \"{notes}\"")
        print(f"   Interventions reviewed:")
        for v in review.verdicts:
            marker = "✓" if v.verdict == "helpful" else ("✗" if v.verdict == "harmful" else "~")
            print(f"     {marker} {v.type}: {v.verdict} (conf={v.confidence:.2f})")

        print(f"   Score changes:")
        for iv, sc in sorted(trace.score_changes.items(), key=lambda x: abs(x[1].delta), reverse=True):
            if sc.source != "unchanged":
                print(f"     {iv:16s} {_print_score_bar(sc.before)} → {sc.after:+.3f}")

        # Show policy state after this night
        print(f"   Policy after night {i}:")
        for iv, s in sorted(policy.intervention_scores.items(), key=lambda x: -x[1]):
            flag = ""
            if s <= _BLOCK_THRESHOLD:
                flag = " ← BLOCKED"
            elif s <= _DEPRIORITIZE_THRESHOLD:
                flag = " ← discouraged"
            elif s >= _PREFER_THRESHOLD:
                flag = " ← preferred"
            print(f"     {iv:16s} {_print_score_bar(s)}{flag}")

        print(f"   Risk posture: {policy.risk_posture}  |  Avg rested: {policy.avg_rested_score}")
        print()

    # Final state
    print("=" * 70)
    print("FINAL POLICY STATE (after 4 nights)")
    print("=" * 70)
    for iv, s in sorted(policy.intervention_scores.items(), key=lambda x: -x[1]):
        flag = ""
        if s <= _BLOCK_THRESHOLD:
            flag = " ← BLOCKED NEXT NIGHT"
        elif s <= _DEPRIORITIZE_THRESHOLD:
            flag = " ← discouraged"
        elif s >= _PREFER_THRESHOLD:
            flag = " ← preferred"
        print(f"  {iv:16s} {_print_score_bar(s)}{flag}")

    print(f"\n  Risk posture:      {policy.risk_posture}")
    print(f"  Avg rested score:  {policy.avg_rested_score}/10")
    print(f"  Nights tracked:    {policy.nights_tracked}")

    # Next-night plan
    stub = NightSummary(user_id="demo_user", date="2026-03-15", total_ticks=0)
    plan, _ = plan_next_night(policy, stub, _PREFS)
    print()
    print("NEXT-NIGHT PLAN (generated from policy):")
    print(f"  Sleep goal:           {plan.sleep_goal}")
    print(f"  Intervention order:   {plan.preferred_intervention_order}")
    print(f"  Blocked from use:     {[iv for iv, s in policy.intervention_scores.items() if s <= _BLOCK_THRESHOLD] or 'none yet'}")
    print(f"  Plan notes:           {plan.notes}")
    print()

    # Verify the expected outcome
    wn_score = policy.intervention_scores.get("white_noise", 0.0)
    rain_score = policy.intervention_scores.get("rain", 0.0)
    print("OUTCOME VERIFICATION:")
    print(f"  white_noise score: {wn_score:+.3f}  (expected: clearly negative)")
    print(f"  rain score:        {rain_score:+.3f}  (expected: positive)")
    assert wn_score < -0.10, f"white_noise should be discouraged, got {wn_score}"
    assert rain_score > 0.0, f"rain should be positive, got {rain_score}"
    assert "white_noise" not in plan.preferred_intervention_order[:2], \
        "white_noise should not be in top-2 of next-night plan"
    print("  ✓ All assertions passed.")
    print()
    print("Run POST /session/{user_id}/end to see this in the live API.")


if __name__ == "__main__":
    run_simulation()
