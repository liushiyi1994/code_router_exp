# ProbeRoute++ Phase 2 Pack

This pack contains two requested deliverables:

1. a technical design document for the upgraded ProbeRoute++ research direction;
2. a Codex starter pack / long-running goal document for Phase 2 experiments with true local model running and evaluation.

## Files

- `PROBEROUTE_TECHNICAL_DESIGN.md` — full technical design: latent states, observability gap, VOI probing, active calibration, true model running, claims.
- `LITERATURE_AND_POSITIONING.md` — relationship to recent top-conference/arXiv literature and novelty boundary.
- `PHASE2_CODEX_STARTER.md` — detailed phase-two implementation and experiment plan for Codex.
- `CODEX_GOAL_PHASE2.md` — `/goal`-ready long-running Codex task.
- `TRUE_MODEL_RUNNING_PROTOCOL.md` — local model serving, prompts, exact scoring, schemas, and logging.
- `PHASE2_EXPERIMENTS_AND_CLAIM_GATES.md` — experiments and the evidence required to support each claim.
- `STARTING_PROMPT_PHASE2.md` — paste-ready starting prompt for Codex.

## What changed from Phase 1

Phase 1 found:

```text
low-rate oracle route states exist;
query-only predicted states are weak;
D2 improves predictability but loses utility;
new-model calibration is promising but not yet strong;
adaptive refinement is not supported yet.
```

Phase 2 converts that into a stronger problem:

```text
The latent route state exists but is partially observable. How can we cheaply observe it and recalibrate new model pools?
```

Cost and provider note:

```text
Local vLLM is the default execution path.
Closed-source OpenAI GPT-family, Anthropic Claude-family, and Google Gemini-family models remain in the later provider-aware model-pool plan.
Provider runs require explicit API access, refreshed pricing, token/cost logging, and separate accounting for provider cost versus local probe latency/GPU proxy cost.
```

## Recommended first action

Use `CODEX_GOAL_PHASE2.md` with Codex `/goal`, or paste `STARTING_PROMPT_PHASE2.md`.
