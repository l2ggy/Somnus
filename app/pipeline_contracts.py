"""Pipeline contracts and ownership boundaries.

This module is the single source of truth for:
- agent metadata exposed by /agents
- lifecycle grouping used by the orchestrator
- ownership boundaries for concurrent work across 3 contributors

Keeping contracts centralized avoids merge-time drift between docs, API metadata,
and execution order.
"""

from dataclasses import asdict, dataclass
from typing import Literal

Lifecycle = Literal["on_sensor_arrival", "tick", "pre_sleep", "morning"]
Owner = Literal["team_a", "team_b", "team_c"]


@dataclass(frozen=True)
class AgentContract:
    name: str
    module: str
    lifecycle: Lifecycle
    owner: Owner
    purpose: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    def as_registry_item(self) -> dict:
        item = asdict(self)
        item["reads"] = list(self.reads)
        item["writes"] = list(self.writes)
        return item


# Ownership split to reduce conflicts during simultaneous work.
# - team_a: product/frontend/demo surfaces (outside agent runtime)
# - team_b: backend ingestion + orchestration infrastructure
# - team_c: sleep intelligence + personalization logic
AGENT_CONTRACTS: tuple[AgentContract, ...] = (
    AgentContract(
        name="intake",
        module="app.agents.backend.intake",
        lifecycle="on_sensor_arrival",
        owner="team_b",
        purpose="Validates and normalises raw sensor dicts into a SensorSnapshot.",
        reads=("raw_payload",),
        writes=("latest_sensor",),
    ),
    AgentContract(
        name="sensor_interpreter",
        module="app.agents.intelligence.sensor_interpreter",
        lifecycle="tick",
        owner="team_c",
        purpose="Translates raw sensor values into labelled signals (hr_status, noise_alert, etc.).",
        reads=("latest_sensor",),
        writes=("hypotheses",),
    ),
    AgentContract(
        name="sleep_state",
        module="app.agents.intelligence.sleep_state",
        lifecycle="tick",
        owner="team_c",
        purpose="Classifies sleep phase (awake/light/deep/REM) and estimates wake_risk.",
        reads=("latest_sensor", "hypotheses"),
        writes=("sleep_state",),
    ),
    AgentContract(
        name="disturbance",
        module="app.agents.intelligence.disturbance",
        lifecycle="tick",
        owner="team_c",
        purpose="Detects noise/light/movement/HR threats and annotates sleep_state.",
        reads=("latest_sensor", "sleep_state"),
        writes=("sleep_state.disturbance_reason", "sleep_state.wake_risk"),
    ),
    AgentContract(
        name="intervention",
        module="app.agents.intelligence.intervention",
        lifecycle="tick",
        owner="team_c",
        purpose="Selects and activates an intervention when wake_risk is elevated.",
        reads=("sleep_state", "nightly_plan", "preferences"),
        writes=("active_intervention",),
    ),
    AgentContract(
        name="strategist",
        module="app.agents.intelligence.strategist",
        lifecycle="pre_sleep",
        owner="team_c",
        purpose="Builds the NightlyPlan from user goals and journal history.",
        reads=("preferences", "journal_history"),
        writes=("nightly_plan",),
    ),
    AgentContract(
        name="journal_reflection",
        module="app.agents.intelligence.journal_reflection",
        lifecycle="morning",
        owner="team_c",
        purpose="Processes the morning journal entry and generates a reflection.",
        reads=("journal_history", "sleep_state", "hypotheses"),
        writes=("journal_history", "hypotheses"),
    ),
)


def validate_contracts() -> None:
    """Fail fast on merge-introduced incompatibilities in agent contracts."""
    names = [contract.name for contract in AGENT_CONTRACTS]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate agent name found in AGENT_CONTRACTS")


def agent_registry() -> list[dict]:
    validate_contracts()
    return [contract.as_registry_item() for contract in AGENT_CONTRACTS]


def tick_contracts() -> tuple[AgentContract, ...]:
    return tuple(c for c in AGENT_CONTRACTS if c.lifecycle == "tick")
