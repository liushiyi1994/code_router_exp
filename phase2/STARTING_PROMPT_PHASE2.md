# STARTING_PROMPT_PHASE2.md

Paste this into Codex at the root of the existing RouteCode repository.

---

You are continuing a PhD research codebase on LLM routing. Phase 1 found that low-rate oracle route labels/states are very strong, but deployable query-predicted labels are weak. The new Phase 2 project is **ProbeRoute++**.

Read these documents first:

- PROBEROUTE_TECHNICAL_DESIGN.md
- LITERATURE_AND_POSITIONING.md
- PHASE2_CODEX_STARTER.md
- TRUE_MODEL_RUNNING_PROTOCOL.md
- PHASE2_EXPERIMENTS_AND_CLAIM_GATES.md
- ROUTECODE_RESEARCH_FINDINGS.md if present

Core idea:
LLM routing has low-dimensional latent route states, but these states are partially observable from query text. ProbeRoute++ learns latent route states from utility matrices, measures the observability gap, collects cheap local probe signals, uses value-of-information to decide whether to probe, and calibrates new model pools at the route-state level.

Important invariant:

```text
query/probe -> belief over latent route states -> selected model
```

Do not bypass states by directly mapping probe features to models.

Tasks:

1. Inspect the repo and existing results.
2. Add Phase 2 module skeletons:
   - states/
   - probes/
   - belief/
   - policies/
   - calibration/
   - local_eval/
3. Reproduce the Phase 1 observability-gap numbers from existing result tables.
4. Implement strong-encoder observability audit.
5. Implement local model runner with dry-run mode and vLLM-compatible OpenAI client.
6. Implement exact-scored evaluation for math and multiple-choice tasks first.
7. Run a 20-query local smoke test with one model.
8. Scale to 200--500 queries with 2--4 local models.
9. Collect cheap probe features:
   - kNN uncertainty;
   - short local draft;
   - confidence-only prompt;
   - logprob/entropy if supported;
   - cheap model agreement if multiple probes are run.
10. Train query+probe state predictor.
11. Implement ProbeRoute++ policies:
   - NeverProbePolicy;
   - AlwaysProbePolicy;
   - EntropyThresholdPolicy;
   - MarginThresholdPolicy;
   - VOIProbePolicy;
   - OracleProbePolicy upper bound.
12. Implement active new-model calibration:
   - random calibration;
   - dataset-stratified calibration;
   - embedding-cluster calibration;
   - uniform state calibration;
   - active state calibration;
   - direct retraining matched-budget baseline.
13. Run ablations and sensitivity.
14. Write results/phase2/PHASE2_EVIDENCE_REPORT.md.

Hard constraints:

- No GPT/Claude/Gemini API calls unless explicitly configured later with API keys, budget, caching, token logging, and refreshed pricing.
- Keep OpenAI GPT-family, Anthropic Claude-family, and Google Gemini-family models in the future cost/model-pool plan even while local vLLM is the default.
- No human-defined labels as main method.
- No open-ended LLM judging until exact tasks work.
- Log every prompt, output, parsed answer, score, token count, latency, model version, and config.
- Include probe cost in metrics.
- No train/test leakage.
- Do not make claims unless PHASE2_EXPERIMENTS_AND_CLAIM_GATES.md says the evidence is sufficient.

First deliverable:

```text
results/phase2/m0_previous_findings_recap.md
results/phase2/table_observability_strong_encoders.csv
results/phase2/fig_observability_gap.pdf
```

Second deliverable:

```text
results/phase2/local_model_outcomes.parquet
results/phase2/probe_features.parquet
results/phase2/table_probe_signal_analysis.csv
```

Third deliverable:

```text
results/phase2/table_proberoute_policy.csv
results/phase2/fig_gap_closed_vs_probe_cost.pdf
results/phase2/table_active_new_model_calibration.csv
results/phase2/fig_new_model_calibration_curve.pdf
results/phase2/PHASE2_EVIDENCE_REPORT.md
```

When done, summarize:

1. whether the observability gap persists;
2. whether probes help;
3. whether VOI probing beats baselines;
4. whether state-level calibration reduces new-model evaluations;
5. whether the project is now ICML/ICLR-level or still diagnostic.
