"""Available interventions the agent can apply during a sleep session."""

DO_NOTHING = "do_nothing"
PLAY_BROWN_NOISE = "play_brown_noise"
PLAY_RAIN = "play_rain"
BREATHING_PACING = "breathing_pacing"
GRADUAL_WAKE = "gradual_wake"

ALL_ACTIONS: list[str] = [
    DO_NOTHING,
    PLAY_BROWN_NOISE,
    PLAY_RAIN,
    BREATHING_PACING,
    GRADUAL_WAKE,
]
