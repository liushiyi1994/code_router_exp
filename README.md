# RouteCode Codex Starter Pack

This starter pack contains durable instructions for beginning the RouteCode LLM routing research project with Codex or another coding agent.

## What RouteCode is

RouteCode studies LLM routing as an information bottleneck problem. A normal router maps:

```text
query -> selected model
```

RouteCode maps:

```text
query -> learned route label -> selected model
```

The route label is learned, discrete, utility-aware, and explainable. It is a label for model-selection behavior, not merely a semantic topic label.

## Files

- `START_HERE_FOR_CODEX.md`: first-file roadmap for Codex.
- `PROJECT.md`: full project blueprint and research design.
- `AGENTS.md`: durable Codex/agent working instructions.
- `CODEX_GOAL.md`: long-horizon `/goal` specification.
- `METHOD_SPEC.md`: exact input/output, training artifacts, objective, examples.
- `EXPERIMENTS.md`: concrete experiment matrix.
- `CLAIMS_AND_EVALUATION.md`: claim hierarchy, thresholds, required evidence.
- `DATA_AND_COST_PLAN.md`: what data is needed, how much, expected costs.
- `LOCAL_MODELS_AND_SERVING.md`: RTX 5090, vLLM, llama.cpp, SGLang, local models.
- `REFERENCES.md`: paper, benchmark, repo, and novelty-boundary links.
- `PAPERS_AND_BASELINES.md`: baseline priority and implementation plan.
- `STARTING_PROMPT.md`: paste-ready Codex starting prompt.
- `PHASE3_AGENT_HANDOFF.md`: current Phase 3 handoff with data flow, metrics,
  artifacts, supported claims, and no-tool caveats.

## Recommended first action

1. Create a new repo named `routecode`.
2. Copy all files in this starter pack into the repo root.
3. Open Codex in that repo.
4. Paste the prompt from `STARTING_PROMPT.md`.
5. Let Codex build the synthetic pilot before touching real benchmark data.

## Current Phase 3 State

If you are entering the existing Phase 3 work rather than starting from the
synthetic pilot, read `PHASE3_AGENT_HANDOFF.md` first after `AGENTS.md`.

The current final result package is in `results/phase3_final/`.

Start with:

- `results/phase3_final/README.md`: final-result landing page.
- `results/phase3_final/EXPERIMENT_PROTOCOL.md`: how the experiments were run,
  what benchmarks were used, and the current benchmark-scope caveat.
- `results/phase3_final/FINAL_EVALUATION_REPORT.md`: consolidated numbers.
- `results/phase3_final/TWO_CLAIM_LIVE_COMPLETION_AUDIT.md`: the two supported
  calibration/onboarding claims and caveats.

Current short answer: ProbeCode-StateCal reaches the configured Broad100
oracle-level target on held-out queries from 9 benchmark families. The state is
learned from train/validation data from those same families, so new-benchmark
generalization is not yet proven and should be a next evaluation.

First new-benchmark smoke:

- `results/phase3_new_benchmark_live/README.md`
- Benchmarks: `google/simpleqa-verified`, `livebench/math`, and
  `livebench/reasoning`.
- Result: a local-vs-GPT cost-aware oracle opportunity exists on a tiny
  15-query slice, but the learned state policy has not yet been frozen and
  evaluated on these new benchmark families.

## Reference note

`REFERENCES.md` and `PAPERS_AND_BASELINES.md` are first-class files. The agent must use them before making novelty claims or choosing baselines. If a link is stale, the agent should update it rather than silently proceeding.
