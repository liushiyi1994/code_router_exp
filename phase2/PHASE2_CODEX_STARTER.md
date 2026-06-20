# Phase 2 Codex Starter Pack: ProbeRoute++ Experiments

This document is for Codex or another coding agent that will continue from the current RouteCode repo.

The Phase 1 result was diagnostic: low-rate oracle route states exist, but query-only predicted states do not recover the oracle. Phase 2 should build the upgraded ProbeRoute++ method and run true model/probe experiments.

---

## 1. Phase 2 goal

Build and evaluate **ProbeRoute++**, a partially observable latent route-state router.

The system should:

1. learn latent route states from query--model utility matrices;
2. measure the observability gap between oracle states and predicted states;
3. collect cheap probe signals from local models;
4. use probes to update beliefs over route states only when worthwhile;
5. calibrate new models at the route-state level using far fewer examples;
6. evaluate with true local model running and exact scoring where possible.

---

## 2. Do not make these mistakes

Do not:

- use human-defined labels as the main method;
- map probe features directly to selected model;
- call GPT/Claude/Gemini APIs in the first Phase 2 experiments;
- run expensive open-ended judging before exact tasks work;
- claim interpretability before generating state cards;
- claim adaptive probing works before cost-adjusted results prove it;
- bypass route states in new-model calibration.

Keep GPT-family, Claude-family, and Gemini-family models in the later
provider-aware model-pool plan. They matter for the applied routing story, but
they require explicit API budget, refreshed pricing source URLs, token logging,
and cached request/response artifacts before any run is allowed.

The central invariant is:

```text
query/probe -> belief over latent route states -> model
```

not:

```text
query/probe -> model
```

---

## 3. Repo additions to implement

Add modules:

```text
src/routecode/states/
  calibration_aware.py
  observability.py
  state_cards.py

src/routecode/probes/
  base.py
  knn_uncertainty.py
  local_model_probe.py
  short_draft_probe.py
  probe_features.py
  probe_costs.py

src/routecode/belief/
  state_predictor.py
  posterior_update.py
  calibration.py

src/routecode/policies/
  voi_probe_policy.py
  threshold_probe_policy.py
  always_probe.py
  never_probe.py

src/routecode/calibration/
  bayesian_new_model.py
  active_state_sampler.py
  calibration_metrics.py

src/routecode/local_eval/
  serve.py
  generation_runner.py
  evaluators.py
  prompt_templates.py
  parsers.py
```

Add experiments:

```text
experiments/50_observability_gap_strong_encoders.py
experiments/51_true_model_generation_matrix.py
experiments/52_probe_collection.py
experiments/53_probe_signal_analysis.py
experiments/54_proberoute_policy_eval.py
experiments/55_active_new_model_calibration.py
experiments/56_probe_ablation.py
experiments/57_phase2_report.py
```

---

## 4. Phase 2 milestones

### M0 — Load previous evidence

Input:

```text
results/llmrouterbench_pilot/*
results/llmrouterbench_broad20/*
results/llmrouterbench_scale20/*
results/llmrouterbench_32model/*
ROUTECODE_RESEARCH_FINDINGS.md
```

Task:

- write loader for previous outcome/result tables;
- reproduce key Phase 1 observation tables;
- compute observability gap using existing results.

Output:

```text
results/phase2/m0_previous_findings_recap.md
```

---

### M1 — Strong encoder observability audit

Goal:

> Verify that predicted-state failure is not just weak hashing features.

Implement query-to-state prediction with:

- all-MiniLM baseline;
- BGE if available;
- Qwen embedding if available;
- ModernBERT/DeBERTa classifier if practical;
- kNN with strong embeddings;
- MLP on strong embeddings.

Metrics:

- state accuracy;
- routing utility;
- observability gap;
- recovered gap vs oracle;
- calibration/ECE;
- runtime.

Output:

```text
results/phase2/table_observability_strong_encoders.csv
results/phase2/fig_observability_gap.pdf
```

Decision:

- If strong encoders close the gap, focus on improved query predictor.
- If gap remains, proceed to probes.

---

### M2 — True local model generation/evaluation

Goal:

> Build a true local outcome/probe matrix with actual model calls.

Use RTX 5090 local models.

Recommended first model pool:

```text
Qwen3-8B
Qwen2.5-Coder-7B-Instruct
DeepSeek-R1-Distill-Qwen-7B
Llama-3.1-8B-Instruct
MiniCPM4.1-8B
Gemma-3-4B or another small baseline
```

Recommended first datasets:

```text
GSM8K / MATH500 / AIME-style math
MMLU-Pro / GPQA multiple choice
HumanEval / MBPP only after safe code execution works
```

Start small:

```text
200--500 queries
4 local models
```

Scale later:

```text
1K--3K queries
4--8 local models
```

Output schema:

```text
results/phase2/local_model_outcomes.parquet
```

Required columns:

```text
query_id
query_text
dataset
domain
model_id
prompt_template
generation_params
raw_output
parsed_answer
quality
cost_proxy
latency
tokens_input
tokens_output
error
```

---

### M3 — Probe collection

Goal:

> Collect cheap observations that may reveal latent route states.

Probe types:

1. kNN uncertainty probe;
2. local cheap model short draft;
3. confidence-only local model prompt;
4. logprob/entropy if serving backend supports it;
5. cheap verifier/classifier score;
6. agreement among two cheap models.

Probe data schema:

```text
results/phase2/probe_features.parquet
```

Required columns:

```text
query_id
probe_id
probe_type
probe_model_id
prompt_template
max_new_tokens
raw_probe_output
parsed_probe_answer
self_confidence
logprob_mean
entropy_proxy
agreement_features
latency
input_tokens
output_tokens
probe_cost_proxy
error
```

Important:

- Do not run one prompt per route state.
- Use generic probes that update belief over all states.
- Keep probes cheap: max 32--64 tokens for generation probes.

---

### M4 — Probe signal analysis

Goal:

> Determine whether probes predict route states or routing regret.

Train/evaluate:

- query-only state predictor;
- probe-only state predictor;
- query + probe state predictor;
- query + kNN uncertainty;
- query + cheap model confidence.

Metrics:

- state prediction accuracy;
- routing utility;
- observability gap closed;
- probe cost;
- regret prediction AUC;
- risk-coverage curve.

Output:

```text
results/phase2/table_probe_signal_analysis.csv
results/phase2/fig_probe_signal_gain.pdf
```

---

### M5 — ProbeRoute++ policy

Implement policies:

```text
NeverProbePolicy
AlwaysProbePolicy
EntropyThresholdPolicy
MarginThresholdPolicy
VOIProbePolicy
OracleProbePolicy for upper bound
```

VOI approximation:

```text
VOI(a) = predicted_gain(a | q, b0) - probe_cost(a)
```

Train `predicted_gain` on validation data:

```text
features = [state entropy, top margin, kNN entropy, model utility margin, query length, domain, probe type]
target = V(after probe) - V(before probe)
```

Route after update:

```text
m = argmax_m sum_z b_final[z] * mu[z,m]
```

Metrics:

- utility;
- quality;
- cost/probe cost;
- fraction probed;
- observability gap closed;
- regret vs oracle;
- latency.

Output:

```text
results/phase2/table_proberoute_policy.csv
results/phase2/fig_gap_closed_vs_probe_cost.pdf
```

---

### M6 — Active new-model calibration

Goal:

> Show route states reduce new-model calibration burden.

Protocol:

1. Hold out one model as `new_model`.
2. Learn route states without using that model if possible.
3. Calibrate `mu[z,new_model]` with limited examples.
4. Update state-to-model table.
5. Evaluate routing performance.

Calibration methods:

- random query calibration;
- dataset-stratified calibration;
- embedding-cluster calibration;
- uniform route-state calibration;
- active route-state calibration;
- direct query-router retraining under same budget;
- full calibration upper bound.

Budgets:

```text
r = 1, 2, 4, 8, 16, 32, 64 examples per state
```

Metrics:

- utility vs new-model evaluations;
- number of evaluations to reach 90% of full-calibration performance;
- posterior uncertainty;
- state ranking changes;
- routing table stability.

Output:

```text
results/phase2/table_active_new_model_calibration.csv
results/phase2/fig_new_model_calibration_curve.pdf
```

---

### M7 — Ablations and sensitivity

Ablate:

- number of states K;
- state objective terms;
- calibration variance weight;
- observability weight;
- probe type;
- probe cost multiplier;
- VOI vs threshold;
- belief update model;
- local model pool;
- datasets;
- cost weight lambda;
- random seeds.

Output:

```text
results/phase2/table_ablation_summary.csv
results/phase2/table_sensitivity_summary.csv
```

---

### M8 — Phase 2 paper evidence report

Write:

```text
results/phase2/PHASE2_EVIDENCE_REPORT.md
```

It must answer:

1. Does the observability gap persist with strong encoders?
2. Which probe signals help?
3. Does ProbeRoute++ close the gap cost-effectively?
4. Does active state-level calibration reduce new-model labels?
5. Which claims are supported, unsupported, or mixed?
6. What is the next paper story?

---

## 5. Success criteria for Phase 2

Phase 2 is successful if at least two of these are true:

1. The observability gap persists with strong encoders.
2. Cheap probes close at least 20--40% of the observability gap at low probe cost.
3. VOI probing beats entropy threshold and always-probe after cost accounting.
4. Calibration-aware states reduce new-model calibration examples vs random/dataset/embedding baselines.
5. Active state calibration reaches strong performance with far fewer evaluations than direct router retraining.
6. Latent state cards are stable and interpretable enough for analysis.

---

## 6. Failure modes and pivots

### Failure: strong encoders close the observability gap

Pivot:

```text
The original issue was weak features. Focus on robust query-to-state prediction and calibration-aware states.
```

### Failure: probes do not help

Pivot:

```text
Route states are not cheaply observable. Focus on calibration and benchmark diagnosis.
```

### Failure: state-level calibration does not beat baselines

Pivot:

```text
The latent states are good for hindsight routing but not good calibration strata. Strengthen calibration-aware state learning or focus on observability paper.
```

### Failure: local model outputs are too noisy

Pivot:

```text
Use deterministic MC/math tasks first, then code tasks later.
```

---

## 7. First command plan for Codex

1. Inspect existing repo and current results.
2. Implement observability audit with stronger embeddings.
3. Implement local model runner in dry-run/mock mode.
4. Add vLLM local endpoint client.
5. Run 20-query smoke test with one local model.
6. Scale to 200-query local matrix.
7. Collect first probe features.
8. Train query+probe state predictor.
9. Implement VOI policy.
10. Write Phase 2 evidence report.
