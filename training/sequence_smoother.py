"""
sequence_smoother.py
--------------------
Post-processes predicted sleep stage sequences using simple transition
smoothing rules based on known sleep-cycle physiology.

Rules applied in order:
  1. Isolated window correction — a single window surrounded by the same
     stage on both sides is replaced by that surrounding stage.
     e.g.  light, awake, light  →  light, light, light

  2. Unlikely one-step jump suppression — direct transitions between
     physiologically implausible stage pairs are corrected when they
     last only one window:
       deep  → rem   (and vice-versa)
       deep  → awake → deep  (spurious brief awakening between deep blocks)

     A jump is only suppressed when it lasts a single window; if it
     persists for two or more windows it is left unchanged.

This module is an offline/post-processing utility only.
It has no dependency on app/ and must never be imported from there.
"""

from __future__ import annotations

# Pairs of direct transitions considered physiologically unlikely.
# Each entry is (from_stage, to_stage); the reverse is also registered.
_UNLIKELY_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("deep", "rem"),
        ("rem", "deep"),
    }
)


def _is_unlikely_jump(a: str, b: str) -> bool:
    return (a, b) in _UNLIKELY_PAIRS


def smooth_stage_sequence(stages: list[str]) -> list[str]:
    """Apply transition smoothing to a predicted sleep stage sequence.

    Operates in two passes so that each rule is applied cleanly to the
    full sequence rather than having earlier corrections interfere with
    later ones in the same pass.

    Args:
        stages: Ordered list of predicted sleep stage strings.
                Valid values: "awake", "light", "deep", "rem".

    Returns:
        New list of the same length with smoothing applied.
        The input list is not modified.
    """
    if len(stages) < 3:
        return list(stages)

    seq = list(stages)

    # ------------------------------------------------------------------
    # Pass 1: isolated window correction
    # If seq[i] differs from seq[i-1] and seq[i+1], and seq[i-1] == seq[i+1],
    # replace seq[i] with the surrounding stage.
    # ------------------------------------------------------------------
    for i in range(1, len(seq) - 1):
        if seq[i - 1] == seq[i + 1] and seq[i] != seq[i - 1]:
            seq[i] = seq[i - 1]

    # ------------------------------------------------------------------
    # Pass 2: unlikely one-step jump suppression
    # For each window i, if the transition from seq[i-1] to seq[i] is an
    # unlikely pair AND seq[i] differs from seq[i+1] (i.e. lasts only one
    # window), replace seq[i] with seq[i-1].
    #
    # Special case — deep → awake → deep:
    # Detect a single "awake" sandwiched between two "deep" windows and
    # replace it with "deep".  This is handled separately because the
    # deep→awake transition is not itself in _UNLIKELY_PAIRS, but the
    # pattern as a whole (deep→awake→deep) is physiologically implausible
    # for a single 30-second window.
    # ------------------------------------------------------------------
    for i in range(1, len(seq) - 1):
        prev_stage = seq[i - 1]
        curr_stage = seq[i]
        next_stage = seq[i + 1]

        # deep → awake → deep: spurious single-window awakening in deep sleep
        if prev_stage == "deep" and curr_stage == "awake" and next_stage == "deep":
            seq[i] = "deep"
            continue

        # Generic unlikely pair that lasts only one window
        if _is_unlikely_jump(prev_stage, curr_stage) and curr_stage != next_stage:
            seq[i] = prev_stage

    return seq


# ---------------------------------------------------------------------------
# CLI demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        # Rule 1: isolated window
        (
            "Isolated awake between light",
            ["light", "awake", "light", "light", "rem"],
        ),
        # Rule 1: isolated deep between rem
        (
            "Isolated deep between rem",
            ["rem", "rem", "deep", "rem", "rem"],
        ),
        # Rule 2: direct deep → rem jump (one window)
        (
            "Single-window deep → rem jump",
            ["deep", "deep", "rem", "light", "light"],
        ),
        # Rule 2: direct rem → deep jump (one window)
        (
            "Single-window rem → deep jump",
            ["rem", "rem", "deep", "light", "light"],
        ),
        # Rule 2: deep → awake → deep
        (
            "Spurious awake between deep blocks",
            ["deep", "deep", "awake", "deep", "deep"],
        ),
        # Rule 2: persistent jump (≥2 windows) — should NOT be changed
        (
            "Persistent deep → rem (2 windows, keep as-is)",
            ["deep", "deep", "rem", "rem", "light"],
        ),
        # No change needed
        (
            "Clean sequence (no smoothing needed)",
            ["awake", "light", "light", "deep", "deep", "rem", "rem"],
        ),
    ]

    col_w = 42
    print(f"\n{'Example':<{col_w}}  {'Before':<35}  After")
    print("-" * (col_w + 75))
    for label, seq in examples:
        smoothed = smooth_stage_sequence(seq)
        before_str = ", ".join(seq)
        after_str = ", ".join(smoothed)
        changed = " *" if smoothed != seq else ""
        print(f"{label:<{col_w}}  {before_str:<35}  {after_str}{changed}")
    print("\n* = sequence was modified")
