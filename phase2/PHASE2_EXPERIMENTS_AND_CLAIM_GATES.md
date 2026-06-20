# Phase 2 Experiments and Claim Gates

This file defines which experiments support which claims. Codex should not make paper claims without satisfying these gates.

---

## Global cost and provider gate

All policy results must account for cost when making applied claims. Local runs should separate model-evaluation latency/GPU proxy cost from probe-acquisition cost.

Closed-source provider runs are later explicit-budget experiments only. If OpenAI GPT-family, Anthropic Claude-family, or Google Gemini-family models are used, the report must include provider, model version/date, request count, input/output token counts, latency, refreshed pricing source URLs, checked date, and cost-adjusted utility. Do not use provider APIs in the default local vLLM pilot.

---

## Claim A: low-dimensional latent route states exist

### Required experiments

1. Oracle state frontier on existing benchmark outcomes.
2. Oracle state frontier on true local model matrix.
3. K sweep:

```text
K = 1, 2, 4, 8, 16, 32, 64
```

### Required baselines

- best single;
- dataset-label lookup;
- embedding-cluster lookup;
- kNN;
- direct learned router;
- oracle query router.

### Support threshold

Claim supported if:

```text
small K states, e.g. K <= 16 or K <= 32, recover a large fraction of oracle gap
and outperform semantic/embedding clusters at same K.
```

### Warning

This is an oracle/diagnostic claim unless the states are predicted without using test utility vectors.

---

## Claim B: observability gap exists

### Required experiments

1. Compare oracle state assignment vs query-predicted state assignment.
2. Repeat with strong encoders, not only hashing features.

Encoders:

```text
hashing baseline
all-MiniLM
BGE / bge-m3
Qwen embeddings
ModernBERT / DeBERTa
kNN strong embeddings
MLP on embeddings
```

### Metrics

- predicted state routing utility;
- oracle state routing utility;
- observability gap;
- state accuracy;
- utility-weighted confusion;
- recovered gap vs oracle;
- CIs.

### Support threshold

Claim supported if:

```text
oracle states remain much stronger than predicted states after strong encoders.
```

### If not supported

If strong encoders close the gap, pivot to:

```text
better query-to-state models + calibration-aware states
```

and downplay observability gap.

---

## Claim C: cheap probes close the observability gap

### Required experiments

1. Collect probe features.
2. Train query+probe state predictor.
3. Compare against query-only predictor.
4. Evaluate cost-adjusted utility.

### Probe types

- kNN uncertainty;
- small model confidence;
- short draft;
- logprob/entropy if available;
- cheap verifier;
- cheap model agreement.

### Baselines

- query-only predictor;
- never probe;
- always probe;
- entropy threshold;
- margin threshold;
- uncertainty-only routing;
- VOI ProbeRoute++;
- oracle probe policy.

### Metrics

- fraction probed;
- probe cost;
- utility;
- observability gap closed;
- cost-adjusted utility;
- latency.

### Support threshold

Claim supported if:

```text
VOI ProbeRoute++ closes a meaningful fraction of the observability gap
and beats threshold/always-probe policies after probe cost accounting.
```

A meaningful first target:

```text
20--40% gap closure with <=30--40% queries probed
```

but final thresholds should be chosen from validation before test.

---

## Claim D: active route-state calibration reduces new-model labels

### Required experiments

1. Hold out each model as new model.
2. Calibrate it using limited examples.
3. Compare state-level calibration against baselines.

### Calibration methods

- random query sampling;
- dataset-stratified sampling;
- embedding-cluster sampling;
- uniform route-state sampling;
- active route-state sampling;
- direct query-router retraining under matched budget;
- full calibration upper bound.

### Budgets

```text
r = 1, 2, 4, 8, 16, 32, 64 examples per state
```

### Metrics

- utility vs number of evaluations;
- evaluations needed to reach 90% of full-calibration performance;
- calibration regret;
- posterior uncertainty;
- state-to-model table stability.

### Support threshold

Claim supported if:

```text
active route-state calibration reaches a target utility with substantially fewer new-model evaluations than direct retraining/random/dataset/embedding calibration.
```

---

## Claim E: learned latent states are interpretable post hoc

### Required experiments

1. Generate state cards.
2. Analyze representative queries.
3. Measure state stability.
4. Report utility profile per state.

### Metrics

- state traffic mass;
- dominant domains/datasets;
- best/second-best model and margin;
- within-state utility variance;
- representative queries;
- stability across seeds;
- adjusted Rand index / state matching across runs.

### Support threshold

Claim supported if:

```text
states are stable enough and have coherent representative queries/utility profiles.
```

Do not claim interpretability from state IDs alone.

---

## Final paper strength rubric

### Strong ICML/ICLR-style paper likely if:

- Claim A supported robustly;
- Claim B supported with strong encoders;
- Claim C or D supported strongly;
- ablations show each technical component matters;
- results hold across at least two model pools or benchmark settings.

### Solid but not top-tier if:

- Claim A and B supported, but C/D weak.

Then the paper is mostly diagnostic.

### Weak if:

- strong encoders eliminate the observability gap;
- probes do not help;
- state calibration does not beat simple stratified baselines.

Then pivot to simpler strong-query-encoder RouteCode or benchmark diagnosis.
