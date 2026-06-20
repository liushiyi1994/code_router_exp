# Literature and Positioning for ProbeRoute++

This document records the current literature boundary for the upgraded RouteCode / ProbeRoute++ project.

## 1. Our new position

Old RouteCode story:

```text
Learn low-dimensional route labels.
```

New ProbeRoute++ story:

```text
LLM routing has low-dimensional latent route states, but they are partially observable from query text. We measure the observability gap, learn calibration-aware latent states, use cheap probes by value of information to infer states when needed, and adapt new model pools by state-level active calibration.
```

This combines four ideas:

1. latent route-state frontier;
2. observability gap;
3. active probing / value of information;
4. active state-level new-model calibration.

The full combination appears to be distinct from existing routing work.

---

## 2. Close LLM routing literature

### 2.1 LLMRouterBench

- Paper: https://arxiv.org/abs/2601.07206
- Role: primary benchmark and motivation.
- Key fact: over 400K instances, 21 datasets, 33 models, and 10 routing baselines.
- Why relevant: it reports that many routers have similar performance under unified evaluation and that a substantial gap remains to the oracle due to model-recall failures.

How we differ/use it:

```text
LLMRouterBench shows the gap and supplies the benchmark.
ProbeRoute++ explains the gap through latent state observability and attempts to close it with probes/calibration.
```

---

### 2.2 RouteLLM

- Paper: https://arxiv.org/abs/2406.18665
- OpenReview: https://openreview.net/forum?id=8sSqNntaMr
- Code: https://github.com/lm-sys/routellm
- Role: canonical learned-router baseline.

How we differ:

```text
RouteLLM learns direct query-to-model routers from preference data.
ProbeRoute++ learns latent route states, models their observability, and recalibrates model pools at the state level.
```

Use as baseline:

- RouteLLM-MF;
- RouteLLM-BERT if checkpoint available;
- RouteLLM random/Elo baselines if available.

---

### 2.3 WebRouter

- Paper: https://arxiv.org/abs/2510.11221
- Role: closest compression/information-bottleneck prior.
- Key idea: cost-aware variational information bottleneck for query-specific routing in web agents.

Novelty warning:

```text
Do not claim first information bottleneck or compressed representation for routing.
```

Our distinction:

```text
WebRouter trains compressed neural representations for routing.
ProbeRoute++ learns latent route states, measures oracle-vs-deployable observability, actively probes hidden states, and uses state-level calibration for new models.
```

---

### 2.4 FineRouter

- Paper: https://arxiv.org/abs/2603.19415
- Role: closest latent-task/latent-label prior.
- Key idea: graph-based latent task discovery + task-aware quality estimation heads.

Novelty warning:

```text
Do not claim first latent task discovery for routing.
```

Our distinction:

```text
FineRouter discovers latent task types to improve routing.
ProbeRoute++ learns latent route states optimized for routing regret and calibration variance, measures observability gaps, and actively probes/calibrates states.
```

---

### 2.5 Select-then-Route

- Paper: https://aclanthology.org/2025.emnlp-industry.28/
- Role: taxonomy-based coarse routing + cascade reference.

How we differ:

```text
Select-then-Route uses taxonomy to narrow candidate models and cascades based on confidence.
ProbeRoute++ learns latent states from utility, not human taxonomy, and uses probes to infer states rather than directly cascade by confidence.
```

---

### 2.6 STEER

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/40413
- Venue: AAAI 2026.
- Role: closest confidence/stepwise routing reference.
- Key idea: use smaller-model confidence/logits for step-level routing between small and large LLMs without external routers.

Novelty warning:

```text
Do not claim first confidence-based or logit-based routing.
```

Our distinction:

```text
STEER routes by small-model confidence.
ProbeRoute++ uses probe signals to update a belief over latent route states, then chooses models using a reusable state-to-model utility table. This preserves model-pool calibration advantages.
```

---

### 2.7 Universal Model Routing / UniRoute

- Paper: https://arxiv.org/abs/2502.08773
- OpenReview: https://openreview.net/forum?id=ka82fvJ5f1
- Role: new/unseen model transfer reference.

How we differ:

```text
UniRoute focuses on model-side representations for unseen LLMs.
ProbeRoute++ focuses on query-side latent route states as reusable calibration strata.
```

Potential complement:

```text
Use UniRoute-style model features as priors for mu[z, m_new].
```

---

### 2.8 Causal LLM Routing

- Paper: https://openreview.net/forum?id=iZC5xoQQkX
- Venue: NeurIPS 2025.
- Role: logged-feedback / reduced-label routing reference.
- Key idea: learn routing policies from observational data using regret-minimizing objectives.

How we differ:

```text
Causal LLM Routing reduces label burden by learning from partial logged feedback.
ProbeRoute++ reduces new-model recalibration burden by estimating model utility at the latent-state level and actively sampling high-value states.
```

These are complementary.

---

### 2.9 BEST-Route

- Paper: https://openreview.net/forum?id=tFBIbCVXkG
- Code: https://github.com/microsoft/best-route-llm
- Venue: ICML 2025.
- Role: recent adaptive-compute routing baseline.
- Key idea: choose both model and number of samples based on difficulty and thresholds.

How we differ:

```text
BEST-Route adapts test-time output compute.
ProbeRoute++ adapts information acquisition before model selection and recalibrates new models at the route-state level.
```

---

### 2.10 Cascade Routing

- Paper: https://openreview.net/forum?id=AAl89VNNy1
- Venue: ICML 2025.
- Role: theory/method complexity reference.
- Key idea: unify routing and cascading, derive optimal strategies, identify quality estimators as critical.

How we differ:

```text
Cascade Routing assumes/learns quality estimators for model selection.
ProbeRoute++ studies the latent-state structure behind quality estimation and the cost of making it observable/calibrated.
```

---

### 2.11 GraphRouter

- Paper: https://openreview.net/forum?id=eU39PDsZtT
- Venue: ICLR 2025.
- Role: strong structured-router baseline.
- Key idea: heterogeneous graph over tasks, queries, and models.

How we differ:

```text
GraphRouter learns structured graph representations for routing.
ProbeRoute++ learns latent route states with explicit observability and calibration objectives.
```

---

## 3. Non-LLM inspirations

### 3.1 Partially observable decision-making

ProbeRoute++ is naturally a partially observable decision problem:

```text
latent state z exists, but query features reveal it imperfectly.
```

The router maintains a belief and can acquire observations before acting.

### 3.2 Active feature acquisition

ProbeRoute++ treats cheap probes as costly features. The router should acquire a probe only when the expected value exceeds its cost.

### 3.3 Bayesian experimental design / value of information

New-model calibration is a value-of-information problem: sample state/model pairs that are most likely to change the routing decision.

### 3.4 Information bottleneck / rate-distortion

Latent route states compress query information, but compression is not the final goal. Compression enables measurement and calibration.

### 3.5 Mixture-of-experts gating

LLM routing is external MoE gating. ProbeRoute++ asks what latent state the gate needs and when that state is observable.

---

## 4. Papers with comparable ICML/ICLR-level method complexity

These papers show that our upgraded design is in the right complexity range if experiments are strong.

### BEST-Route, ICML 2025

Method complexity:

```text
model choice + number of samples + difficulty thresholds + cost-quality objective
```

ProbeRoute++ comparable complexity:

```text
latent state learning + VOI probing + state-level active calibration
```

### Cascade Routing, ICML 2025

Method complexity:

```text
formal optimal strategies for routing/cascading + quality-estimator analysis
```

ProbeRoute++ comparable complexity:

```text
observability gap definition + value-of-information policy + calibration sample efficiency
```

### Causal LLM Routing, NeurIPS 2025

Method complexity:

```text
observational data + regret minimization + surrogate objectives
```

ProbeRoute++ comparable complexity:

```text
latent route states + partial observability + active probing + Bayesian calibration
```

### Universal Model Routing, ICLR 2026

Method complexity:

```text
new unseen models + model-side feature representation + routing transfer
```

ProbeRoute++ comparable complexity:

```text
new model pools + query-side latent state calibration + active sample allocation
```

### STEER, AAAI 2026

Method complexity:

```text
confidence-guided stepwise routing using small-model logits
```

ProbeRoute++ should be more principled:

```text
probe signals update latent-state beliefs; probes selected by VOI; model choice mediated by state utility table.
```

---

## 5. Novelty statement draft

A concise novelty statement:

> Prior routing methods learn direct query-to-model policies, compressed neural routers, or latent task clusters. We instead show that routing has a low-dimensional latent state structure that is strong in hindsight but only partially observable from query text. We formalize this as an observability gap, learn calibration-aware latent route states, and propose a value-of-information probing and calibration framework that uses cheap observations and state-level sampling to adapt routing to new model pools with far fewer evaluations.

---

## 6. What to avoid in writing

Avoid:

```text
We are the first to compress queries for routing.
We are the first to use latent labels for routing.
We are the first to use confidence/probing.
We save router tokens.
```

Use:

```text
We introduce the observability gap between oracle latent route states and deployable route-state inference.
We learn latent states optimized for routing and calibration, not semantic task labels.
We use probes to infer latent state beliefs, not to directly choose a model.
We use latent states as calibration strata for new model pools.
```
