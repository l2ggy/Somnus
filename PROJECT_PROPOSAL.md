# Somnus Project Proposal (Reworked Agent-Based Plan)

## 1) Project Positioning
**Somnus is an agentic sleep assistant that monitors overnight signals, predicts wake-up risk, triggers calming interventions in real time, and learns from morning feedback to personalize future nights.**

Rather than a long, linear ML stack, Somnus now uses a **shared-state multi-agent architecture** where specialized agents coordinate through one central session state.

---

## 2) Problem and Opportunity
Most sleep apps are passive: they summarize sleep after the fact. Somnus is designed to be active:
- Observe wearable/environmental signals while the user sleeps.
- Detect rising disturbance risk during the night.
- Apply interventions (e.g., brown noise, rain, breathing pace, wake ramp) before full wake-up.
- Learn user-specific preferences and outcomes via morning journal reflection.

This supports a stronger hackathon narrative: **monitor → decide → act → learn**.

---

## 3) Current Product Scope (Aligned to Repo)
Somnus is currently implemented as a FastAPI backend with session-based state and three lifecycle phases:
1. **Pre-sleep planning** (build nightly plan)
2. **Night tick loop** (ingest sensor snapshot, assess risk, intervene)
3. **Morning reflection** (journal feedback and hypothesis updates)

### Core user-facing outcomes
- Personalized nightly plan based on goals/preferences/history.
- Real-time disturbance detection and intervention selection.
- Morning reflection with rolling context that informs next-night behavior.

---

## 4) Agent Architecture (Shared State)
All agents read/write a single `SharedState` object (session store). This is the coordination layer.

### Agent set in current implementation
- **Intake Agent**: validates/normalizes incoming sensor payloads.
- **Sensor Interpreter Agent**: converts raw readings into interpretable signal hypotheses.
- **Sleep State Agent**: infers sleep phase + wake risk.
- **Disturbance Agent**: detects threats (noise/light/movement/physiology) and updates risk/reason.
- **Intervention Agent**: selects active intervention and intensity.
- **Strategist Agent** *(pre-sleep)*: creates a nightly plan from preferences + history.
- **Journal Reflection Agent** *(morning)*: processes user feedback and logs insights/hypotheses.

This maps cleanly to the new direction:
- Observer/Intake → Intake + Sensor Interpreter
- Pattern/Risk → Sleep State + Disturbance
- Intervention → Intervention
- Memory/Morning Review → Journal Reflection + persistent session history

---

## 5) Data and State Design
The shared state includes:
- User preferences/goals and intervention constraints
- Latest sensor snapshot
- Inferred sleep state (phase, confidence, wake risk, disturbance reason)
- Active intervention
- Nightly plan
- Hypothesis log
- Journal history

This enables deterministic orchestration, easy testing, and safe incremental LLM integration.

---

## 6) Intelligence Strategy
Somnus uses a **hybrid approach**:
- **Deterministic core** for reliability in real-time ticks.
- **Optional GPT mode** for richer pre-sleep planning and morning reflection, with fallback to deterministic logic.

This preserves responsiveness/safety while still showcasing AI reasoning in key moments.

---

## 7) 3-Person Team Split (Balanced)
### A. Product + Frontend + Pitch/Presentation
- User flow design (start session → live night view → morning review)
- UI for plan, live risk/intervention status, and journal
- Demo script, visuals, and judge-facing story

### B. Backend + Agent Orchestration
- FastAPI endpoints, session lifecycle, state persistence abstraction
- Orchestrator sequencing and traceability
- Reliability, validation, and fallback handling

### C. Sleep Intelligence + Personalization
- Sensor interpretation rules and disturbance heuristics
- Intervention policy tuning and safety constraints
- Journal-to-hypothesis logic, synthetic scenarios, and behavior updates

This keeps responsibilities independent while converging on one shared state contract.

---

## 8) MVP Deliverables
1. **Session API** for start/sensor/journal/state
2. **Agent orchestration loop** with trace output per tick
3. **Adaptive intervention selection** based on wake risk and preferences
4. **Morning feedback loop** that stores insights for personalization
5. **Frontend demo** showing:
   - user goals/preferences input
   - generated nightly plan
   - simulated disturbance + intervention response
   - morning journal update effect

---

## 9) Demo Story (3 Minutes)
1. User starts a session with sleep goals/preferences.
2. Somnus generates tonight’s plan.
3. Simulated overnight sensor stream enters the tick loop.
4. Disturbance risk rises (e.g., high noise/light + movement).
5. Somnus activates intervention and updates shared state in real time.
6. Morning journal is submitted; reflection updates hypotheses for future nights.

**Key message:** Somnus is not just tracking sleep—it is **coordinating specialized agents to protect and improve sleep in real time, then learning what works per user**.

---

## 10) Immediate Next Steps
- Finalize frontend views around current API contract.
- Add synthetic overnight scenarios for robust demo reliability.
- Expand personalization rules (preferred/disliked interventions, aggressiveness profiles).
- Add lightweight persistence (Redis/Postgres) if needed for multi-worker stability.
- Add evaluation metrics for demo (wake-risk reduction proxy, intervention acceptance, rested score trend).
