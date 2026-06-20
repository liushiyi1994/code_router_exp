# CODEX_GOAL_PHASE2.md — Long-Running Codex Goal for ProbeRoute++

Use this file with Codex `/goal`.

```text
/goal Build Phase 2 of the RouteCode project: ProbeRoute++ with latent route states, observability-gap measurement, cheap local probes, and active state-level new-model calibration.

Context:
The Phase 1 RouteCode findings show that low-rate oracle route states are very strong, but deployable query-predicted route states are weak. The bottleneck is observability: the useful low-dimensional state exists in hindsight but is not reliably inferable from query text. Phase 2 should not merely polish RouteCode. It should implement ProbeRoute++, a partially observable routing system.

Core invariant:
query/probe -> belief over latent route states -> selected model
Do not bypass latent states by mapping probe features directly to model.

Research goals:
1. Learn latent route states from query-model utility matrices, not human labels.
2. Measure the observability gap between oracle states and predicted states.
3. Test whether strong encoders close the gap.
4. Collect cheap local probe signals using true local model runs.
5. Implement ProbeRoute++ policies: never-probe, always-probe, threshold probe, and VOI probe.
6. Implement active state-level calibration for new model pools.
7. Evaluate whether ProbeRoute++ closes the observability gap and/or reduces new-model calibration labels.
8. Produce a Phase 2 evidence report with supported and unsupported claims.

Read first:
- PROBEROUTE_TECHNICAL_DESIGN.md
- LITERATURE_AND_POSITIONING.md
- PHASE2_CODEX_STARTER.md
- TRUE_MODEL_RUNNING_PROTOCOL.md
- ROUTECODE_RESEARCH_FINDINGS.md if present in repo

Implementation tasks:
1. Add modules for latent states, probes, belief updates, VOI policies, local evaluation, and active calibration.
2. Reproduce Phase 1 observability-gap numbers from existing results.
3. Run strong-encoder observability audit.
4. Implement local model generation runner with vLLM-compatible OpenAI client and dry-run mode.
5. Run a 20-query local smoke test with one model.
6. Scale to 200--500 queries and 2--4 local models on exact-scored datasets.
7. Collect probe features with short cheap probes.
8. Train query+probe state predictor.
9. Implement VOIProbePolicy and compare with baselines.
10. Implement active new-model calibration and compare with random/dataset/embedding/direct-router baselines.
11. Run ablations and sensitivity.
12. Write results/phase2/PHASE2_EVIDENCE_REPORT.md.

Hard constraints:
- No GPT/Claude/Gemini API calls unless explicitly configured later.
- Closed-source providers must remain in the cost/model-pool plan even while local vLLM work is the default. Future provider-aware runs should include OpenAI GPT-family, Anthropic Claude-family, and Google Gemini-family models when API access and budget are explicitly enabled.
- Do not hard-code stale closed-source pricing. Before any provider-cost result, refresh the price snapshot, record source URLs and checked dates, and separate prompt/model-evaluation cost from local probe latency/GPU cost.
- No human-defined labels as the main method.
- No expensive open-ended judging until exact tasks work.
- All prompts, outputs, parsed answers, scores, token counts, and latencies must be logged.
- Train/val/test leakage is forbidden.
- Probe cost must be included in utility/cost accounting.
- A claim is supported only if backed by tables/figures and confidence intervals.

Minimum deliverables:
- results/phase2/table_observability_strong_encoders.csv
- results/phase2/fig_observability_gap.pdf
- results/phase2/local_model_outcomes.parquet
- results/phase2/probe_features.parquet
- results/phase2/table_probe_signal_analysis.csv
- results/phase2/table_proberoute_policy.csv
- results/phase2/fig_gap_closed_vs_probe_cost.pdf
- results/phase2/table_active_new_model_calibration.csv
- results/phase2/fig_new_model_calibration_curve.pdf
- results/phase2/PHASE2_EVIDENCE_REPORT.md

Definition of done:
Phase 2 is complete when the evidence report clearly states whether:
1. the observability gap persists with strong encoders;
2. cheap probes close a meaningful fraction of the gap;
3. VOI probing beats threshold/always-probe baselines after cost accounting;
4. active route-state calibration reduces new-model evaluations;
5. the upgraded ProbeRoute++ story is strong enough for an ICML/ICLR-style paper.
```
