# Somnus Project Proposal (Agentic Shared-State Architecture)

## 1) Project Vision
Somnus is a real-time sleep companion that **observes**, **interprets**, and **responds** during the night, then **learns** from morning feedback to improve future nights. The core intelligence is not a single model pipeline; it is a coordinated set of specialized agents operating over one shared sleep-state object.

## 2) Problem Statement
Most sleep tools are either passive trackers (insight after the fact) or static sound apps (intervention without personalization). Somnus targets the missing middle:
- detect likely disturbances while sleep is happening,
- apply low-friction interventions in real time,
- adapt nightly strategy based on what has worked for the user.

## 3) Architecture Direction (Current)
Somnus now uses an **agent-based, shared-state backend**. Each agent reads the same `SharedState`, writes scoped updates, and hands off to the next agent via the orchestrator.

### Core lifecycle
1. **Pre-sleep planning** (once): strategy generation for tonight.
2. **Night tick loop** (every sensor update): observe → infer phase/risk → detect disturbance → intervene.
3. **Morning reflection** (once): capture user-reported outcomes and feed learning back into future planning.

This structure keeps the system simple to reason about, testable, and pitchable for hackathon judging.

## 4) Implemented Agent Set (Repository-Aligned)
Somnus currently implements the following cooperating agents:

- **Intake Agent**: validates/clamps incoming raw sensor payloads and writes `latest_sensor`.
- **Sensor Interpreter Agent**: transforms sensor values into structured signal labels.
- **Sleep State Agent**: infers current phase (`awake/light/deep/rem`) and baseline wake risk.
- **Disturbance Agent**: detects likely disruption sources (light/noise/movement/HR) and updates risk/reason.
- **Intervention Agent**: chooses and scales an intervention (`brown_noise`, `rain`, etc.) using plan + preferences + risk.
- **Strategist Agent**: produces the nightly plan pre-sleep (deterministic with optional GPT path + fallback).
- **Journal Reflection Agent**: processes morning entry, records reflection, and supports optional GPT enrichment + fallback.

Together these cover the roles described in the new direction (observer/pattern/risk/intervention/memory/morning review), but in repository-specific modules and naming.

## 5) Shared State + Learning Loop
The central `SharedState` is the product backbone. It includes:
- user preferences,
- latest sensor snapshot,
- inferred sleep state,
- active intervention,
- nightly plan,
- hypothesis/reflection history,
- journal history.

This enables:
- clear inter-agent handoffs,
- deterministic fallbacks for reliability,
- personalization over multiple nights through accumulated history.

## 6) Product Scope for Hackathon Demo
### Demo story
“Somnus monitors sleep continuously, reacts when risk rises, and improves recommendations based on the user’s own outcomes.”

### Demo flow
1. Start session with user preferences + history.
2. Stream sensor events and show per-tick agent trace.
3. Show disturbance detection + intervention decision in real time.
4. Submit morning journal and show reflection-driven memory update.
5. Start next-night planning to demonstrate adaptation.

## 7) 3-Person Team Split (Balanced)

### A. Product + Frontend + Presentation
- UX flow for session start/night monitoring/morning review.
- Judge-facing dashboard: state timeline, agent trace, intervention timeline.
- Pitch deck, demo script, and narrative framing.

### B. Backend + Agent Orchestration
- FastAPI endpoints, session lifecycle, and in-memory/store abstraction.
- Orchestrator sequencing and traceability.
- Reliability features: validation, fallbacks, structured API responses.

### C. Sleep Intelligence + Personalization Logic
- Sensor interpretation heuristics and phase/disturbance/risk logic.
- Intervention selection/tuning rules.
- Data schemas, synthetic scenarios, and feedback-based behavior updates.

## 8) Immediate Next Milestones
1. Add lightweight persistence (Redis/DB) for session continuity.
2. Expand synthetic overnight scenarios for judge demos and regression checks.
3. Improve personalization policy using rolling journal/intervention outcomes.
4. Add frontend timeline visualization for agent handoffs and state transitions.
5. Harden evaluation rubric (latency, intervention appropriateness, personalization gain).

## 9) Success Criteria
- **Real-time responsiveness**: state updates and intervention decisions are fast and stable.
- **Explainability**: each tick exposes interpretable trace outputs.
- **Personalization evidence**: next-night planning changes based on prior outcomes.
- **Demo clarity**: architecture and value proposition are easy for judges to understand in minutes.
