# Expected Results and Success Criteria for ProbeRoute++ Controlled Experiments

This file defines the expected results, claim gates, and target numbers for the controlled ProbeRoute++ run.

Important: these are **targets and decision criteria**, not results we already have. Codex must report the actual numbers honestly. If the target is not reached, the paper claim must be weakened or revised.

The current Phase 1 RouteCode evidence supports only the diagnostic finding that low-rate oracle routing structure exists while deployable query-to-route-label prediction is weak. Phase 2 must prove that cost-aware latent states, cheap probes, and state-level calibration turn this diagnostic into a strong method.

---

## 1. Main SOTA-level success target

For the main controlled setting with 4--5 local models and 2 frontier models, ProbeRoute++ should aim to be on the best quality--cost--latency frontier.

### Primary target

```text
ProbeRoute++ should be within 3 absolute quality points of the cost-aware oracle,
while using much lower remote frontier cost than all-frontier policies.
```

If quality is reported as accuracy/pass@1/exact match:

```text
Quality gap to cost-aware oracle <= 3 absolute points.
```

If utility is reported:

```text
Cost-aware utility >= 95--97% of cost-aware oracle utility.
```

### Cost target

```text
Normalized remote API cost <= 0.15x--0.35x of all-frontier.
```

A stronger target is:

```text
<= 0.20x all-frontier remote cost with <=3 quality-point loss vs cost-aware oracle.
```

This mirrors the type of result AIMS reports: high accuracy/usage tradeoff with normalized remote cost far below All-LLM. Our setting is different, but the reporting style should be similarly clear.

### Latency target

```text
p50 latency <= all-frontier p50 latency.
p95 latency <= all-frontier p95 latency or no more than 1.2x all-frontier p95.
Router + probe overhead <= 10% of total latency on average.
```

If frontier API latency is highly variable, also report:

```text
p90 and p99 latency
SLA success rate at 2s, 5s, 10s, 20s budgets
```

### Frontier usage target

```text
Frontier-call rate <= 25%--40% of queries.
Local-model usage >= 60%--75% of queries.
Probe rate <= 20%--40% of queries.
```

If a benchmark is very hard, allow higher frontier rate, but show per-benchmark breakdown.

### Baseline dominance target

ProbeRoute++ should beat or Pareto-dominate:

```text
best single model
best local model
all-GPT frontier under cost-aware utility
all-Claude frontier under cost-aware utility
dataset/topic lookup
embedding-cluster lookup
kNN router
direct MLP/BERT router
confidence cascade
threshold-probe variant
```

Minimum acceptable main-result claim:

```text
At matched quality, ProbeRoute++ reduces remote cost by >=30% vs the strongest baseline.
At matched cost, ProbeRoute++ improves quality by >=2--4 absolute points vs the strongest baseline.
```

Strong claim:

```text
ProbeRoute++ is the best method on the quality--cost Pareto frontier and is within 3 points of the cost-aware oracle.
```

---

## 2. Experiment-by-experiment expected results

## A1. Cost-aware routability audit

### Purpose

Show that the controlled 4--5 local + 2 frontier model pool has routing headroom.

### Expected result

```text
Query oracle > best single by >= 10 absolute quality points, or
cost-aware oracle improves utility by >= 15% over best single.
```

Expected model behavior:

```text
Frontier models win hard reasoning/code/science cases.
Local models win many easy/general/routine cases.
No single model wins everything.
```

### Success threshold

```text
Oracle gap must be non-trivial: recovered gap denominator should be large enough for routing to matter.
```

If best single is already within 2--3 points of oracle, the model pool is too dominated and should be changed.

---

## A2. Cost-aware latent route-state frontier

### Purpose

Show that low-dimensional latent route states exist under a cost-aware objective, not only quality-only utility.

### Expected result

```text
K=8 or K=16 oracle latent states recover >=85%--95% of the cost-aware oracle gap.
K=32 should be near saturation.
```

Strong target:

```text
K=16 oracle latent states recover >=90% of oracle gap.
```

### Claim supported

```text
LLM routing has a low-dimensional latent decision structure.
```

If K=32 still recovers less than 70%, low-dimensional route-state framing is weak.

---

## A3. Observability gap

### Purpose

Show that the latent state exists in hindsight but is not fully observable from query text.

### Expected result

```text
Oracle latent states recover >=90% of oracle gap.
Query-only predicted states recover much less, ideally <=50%--65%.
```

Strong observability-gap target:

```text
oracle state recovered gap - query-only state recovered gap >= 25 absolute points.
```

Example desired pattern:

```text
Oracle latent state: 0.90 recovered gap
Query-only state:   0.45 recovered gap
Gap:                0.45
```

### Claim supported

```text
Low-dimensional route states are only partially observable from query text.
```

If strong encoders close the gap almost completely, then ProbeRoute++ probing is less important and the paper should emphasize calibration instead.

---

## A4. Cheap probe feasibility

### Purpose

Show that cheap probes help infer latent states for uncertain queries.

### Expected result

```text
Query-only state prediction: 0.35--0.55 recovered gap.
Query + cheap probes:       0.55--0.75 recovered gap.
```

Minimum target:

```text
cheap probes improve recovered gap by >=10 absolute points.
```

Strong target:

```text
cheap probes improve recovered gap by >=20 absolute points, while probing <=40% of queries.
```

### Good probe behavior

```text
Probes help most on high-entropy / low-margin queries.
Probes should not be needed for easy high-confidence cases.
```

---

## B1. Main ProbeRoute++ evaluation

### Purpose

Show that ProbeRoute++ is better than baselines in quality--cost--latency tradeoff.

### Expected result

ProbeRoute++ should achieve:

```text
Quality gap to cost-aware oracle <= 3 absolute points.
Normalized remote cost <= 0.15x--0.35x all-frontier.
Frontier-call rate <= 25%--40%.
Probe rate <= 20%--40%.
p95 latency <= all-frontier p95 or <=1.2x all-frontier p95.
```

Compared with strongest baseline:

```text
>=2--4 point quality gain at matched cost, or
>=30% remote-cost reduction at matched quality.
```

### Expected table pattern

| Method | Quality | Norm. remote cost | Frontier rate | Probe rate | p95 latency | Expected role |
|---|---:|---:|---:|---:|---:|---|
| All frontier GPT | highest/near-highest | 1.00x | 100% | 0% | high | quality upper reference |
| All frontier Claude | high | 1.00x | 100% | 0% | high | quality upper reference |
| Best local | moderate | 0.00x | 0% | 0% | low | cost lower reference |
| Confidence cascade | good | 0.30x--0.50x | 30%--50% | high | medium | strong baseline |
| Direct MLP/BERT router | good | 0.25x--0.50x | 25%--50% | 0% | low/medium | learned baseline |
| ProbeRoute++ | near-oracle | 0.15x--0.35x | 25%--40% | 20%--40% | competitive | target SOTA |
| Cost-aware oracle | best | lowest possible | oracle | oracle | oracle | unreachable upper bound |

---

## B2. New-model calibration

### Purpose

Show that latent route states reduce new-model calibration burden.

### Expected result

```text
State-level calibration reaches direct full-router performance with 3x--5x fewer new-model evaluations.
```

Strong target:

```text
K=16, r=16 or r=32 examples/state gives within 3--5 quality points of full calibration.
```

For K=16:

```text
r=16 -> 256 evaluations per new model
r=32 -> 512 evaluations per new model
```

Target claim:

```text
ProbeRoute++ adds a new model with a few hundred calibration calls, not thousands.
```

### Expected curve

```text
r=4:  partial improvement
r=8:  large improvement
r=16: near saturation begins
r=32: close to full calibration
```

### Baselines to beat

```text
random calibration
dataset-stratified calibration
embedding-cluster calibration
direct router retraining under same budget
```

---

## B3. Cost and latency accounting

### Purpose

Show that improvements are real under remote cost and latency, not just accuracy.

### Expected result

```text
Router state predictor latency: milliseconds to low tens of milliseconds.
Cheap probe latency: much smaller than frontier call or used only on <=40% queries.
Router + probe overhead: <=10% average total latency.
Remote API cost dominates monetary cost.
ProbeRoute++ normalized remote cost: 0.15x--0.35x all-frontier.
```

Report:

```text
$/1K queries
$/1M queries
p50/p95/p99 latency
SLA success rate
component latency breakdown
```

---

## C1. Component ablation

### Purpose

Show every major component matters.

### Expected result

```text
Full ProbeRoute++ is best quality-cost-latency tradeoff.
Without latent states: worse calibration or worse transfer.
Without probes: lower quality / larger observability gap.
Without VOI: more probes for same quality or lower quality at same probe budget.
Without active calibration: more new-model evaluations needed.
```

Target degradation:

```text
Removing probes should reduce recovered gap by >=5--10 points.
Removing active calibration should require >=2x more calibration examples.
Removing VOI should increase probe rate by >=20% relative or reduce utility at matched probe cost.
```

---

## C2. Number of latent states K

### Sweep

```text
K = 4, 8, 16, 32
```

### Expected result

```text
K=4 underfits.
K=8 or K=16 gives best quality-cost-calibration tradeoff.
K=32 gives small quality improvement but worse predictability/calibration cost.
```

Target:

```text
K=8/16 should be within 1--2 points of K=32 while requiring fewer calibration examples.
```

---

## C3. Probe policy ablation

### Compare

```text
never probe
always probe
entropy threshold
margin threshold
VOI probe
oracle probe upper bound
```

### Expected result

```text
VOI probing should achieve >=90% of always-probe utility with <=60% of always-probe probe calls.
VOI should beat entropy/margin thresholds at matched probe budget.
```

---

## C4. Calibration budget ablation

### Sweep

```text
r = 4, 8, 16, 32 examples per latent state
```

### Expected result

```text
utility improves quickly from r=4 to r=16;
r=32 is near saturation.
```

Target:

```text
r=16 or r=32 should reach within 3--5 points of full calibration.
```

---

## D1. Cost weight sensitivity

### Sweep

```text
lambda_cost = 0, low, medium, high
```

### Expected result

```text
As lambda increases, ProbeRoute++ routes more to local models and lowers remote cost.
Quality drops gracefully.
ProbeRoute++ remains on or near the best Pareto frontier.
```

---

## D2. Latency/SLA sensitivity

### Sweep

```text
latency budget = 2s, 5s, 10s, 20s
```

### Expected result

```text
Under tight latency budgets, ProbeRoute++ probes less and avoids slow frontier calls when possible.
Under relaxed budgets, it uses frontier/probes more for quality.
```

Report:

```text
SLA success rate
quality under each budget
remote cost under each budget
```

---

## D3. Frontier price sensitivity

### Sweep

```text
frontier price multiplier = 0.5, 1, 2, 5
```

### Expected result

```text
As frontier price increases, ProbeRoute++ uses frontier less.
It should remain better than all-frontier and confidence cascade across price regimes.
```

---

## D4. Held-out domain sensitivity

### Protocol

Train/calibrate on a subset of benchmarks and test on held-out benchmarks.

Example:

```text
train/calibrate: GSM8K + MBPP + MMLU-Pro
test: MATH500 + AIME + HumanEval + LiveCodeBench + GPQA
```

### Expected result

```text
Performance drops under held-out domains, but ProbeRoute++ should remain better than direct baselines at the same calibration budget.
```

Target:

```text
held-out domain degradation <=5--8 quality points vs random split.
```

---

## 3. Claim gates

## Claim 1: SOTA quality-cost-latency routing

Supported only if:

```text
ProbeRoute++ is within 3 quality points of cost-aware oracle,
remote cost <=0.35x all-frontier,
p95 latency <= all-frontier p95 or <=1.2x all-frontier,
and ProbeRoute++ Pareto-dominates strongest baseline.
```

If these are not all true, weaken to:

```text
ProbeRoute++ improves quality-cost tradeoff over selected baselines.
```

---

## Claim 2: no extensive router training

Supported only if:

```text
route-state predictor is trained with <=5K labeled/routed examples or precomputed benchmark matrix;
no LoRA or large-router fine-tuning is needed for main method;
training time is <=1--3 hours on a single GPU or CPU/GPU workstation;
new-model adaptation uses <=512--1024 evaluations per new model in main setting.
```

If a ModernBERT/DeBERTa classifier is used, report training time and GPU memory.

The claim should be phrased as:

```text
ProbeRoute++ reduces outcome-label and calibration burden, not necessarily neural training FLOPs.
```

---

## Claim 3: easy new-model calibration

Supported only if:

```text
state-level calibration reaches direct-router performance with 3x--5x fewer evaluations,
or reaches within 3--5 quality points of full calibration with <=512 evaluations for K=16.
```

---

## Claim 4: low-dimensional routing structure

Supported only if:

```text
K=8/16 oracle latent states recover >=85%--90% of cost-aware oracle gap across most benchmarks.
```

---

## Claim 5: observability gap and probe benefit

Supported only if:

```text
oracle state recovered gap - query-only recovered gap >=25 points,
and probes improve recovered gap by >=10 points, preferably >=20 points.
```

---

## 4. Red flags and what to do

### Red flag 1: ProbeRoute++ is more than 5 points below oracle

Then do not claim near-oracle. Focus on cost reduction or calibration.

### Red flag 2: ProbeRoute++ cost is not much lower than frontier

If normalized remote cost >0.50x all-frontier, the cost claim is weak. Tighten lambda, improve state thresholds, or reduce probe/frontier fallback.

### Red flag 3: Probes add too much latency

If p95 latency is worse than all-frontier by >20%, move probes offline, use non-generative probes, or make probing rarer.

### Red flag 4: Strong direct router beats ProbeRoute++ everywhere

Then emphasize new-model calibration and interpretability, or add state-level calibration as the main contribution.

### Red flag 5: New-model calibration does not beat random calibration

Then the latent states are not useful calibration strata. Revisit the state objective and add within-state utility variance penalty.

---

## 5. Expected final paper headline if successful

If the targets are met, the final paper can claim:

```text
ProbeRoute++ achieves near-oracle cost-aware routing, within 3 quality points of the cost-aware oracle, while reducing remote frontier cost to 0.15x--0.35x of all-frontier and requiring only hundreds of evaluations to add a new model.
```

A safer version:

```text
ProbeRoute++ improves the quality-cost-latency frontier over direct routers and confidence cascades by learning cost-aware latent route states, probing only uncertain queries, and calibrating new models at the state level.
```
