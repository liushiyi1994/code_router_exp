# Pilot-First Research Flow for RouteCode

This file makes the intended research workflow explicit for Codex and future agents.

The project should **not** start by building a complex method and then searching for results. It should follow the research arc:

```text
Pilot study -> observations -> method design -> method evaluation -> ablation -> sensitivity -> paper claims
```

The first goal is to find the phenomenon. The method should be designed only after the pilot tells us what structure exists.

---

## 0. Core principle

RouteCode is a research project, not just an engineering repo.

Do not implement all method variants immediately. First measure the routing information structure:

1. Is the workload routable at all?
2. Do small route labels recover much of learned-router performance?
3. Are utility-aware route labels better than topic/embedding labels?
4. Are route-label errors concentrated and predictable?
5. Does benchmark split design change router rankings?

Only after answering these should we decide which RouteCode method variant deserves full development.

---

## 1. Phase A — setup and synthetic sanity

### Purpose

Build a local testbed so the code works before using real LLM routing data.

### Data

Synthetic query--model utility matrices with known structure:

```text
U(q, m) = model_skill[m]
          - query_difficulty[q]
          + domain_affinity[domain(q), m]
          + route_code_affinity[true_code(q), m]
          + residual_interaction[q, m]
          + noise
```

### What synthetic data is for

Synthetic data is only for debugging and validating the pipeline.

It should test whether the implementation can recover known structure when that structure is planted.

### What synthetic data is not for

Do not make scientific claims from synthetic data.

### Required outputs

```text
results/demo/table_routability.csv
results/demo/table_recovered_gap.csv
results/demo/table_rate_distortion.csv
results/demo/fig_compression_ladder.pdf
results/demo/fig_rate_distortion.pdf
```

---

## 2. Phase B — real-data pilot study

This is the real start of the research.

### Primary benchmark

Use LLMRouterBench if accessible. RouterBench / RouteLLM are backup sources.

### Pilot B0: routability audit

Question:

```text
Is there enough oracle gap for routing to matter?
```

Compute:

- best single model utility;
- oracle utility;
- oracle gap;
- model-win distribution;
- per-domain oracle gap;
- dominance ratio.

Expected observations:

- Some model pools are dominated by one model and are not good for routing.
- Main experiments should focus on pools with non-trivial oracle gap.

Decision gate:

- If oracle gap is tiny, do not build RouteCode yet. First identify routable subsets or change model pool.

---

### Pilot B1: compression ladder

Question:

```text
How much routing performance is recoverable from compressed query labels?
```

Compare:

1. best single;
2. dataset-label lookup;
3. predicted-topic lookup;
4. embedding-cluster lookup;
5. kNN router;
6. simple learned router;
7. RouteCode oracle labels;
8. query oracle.

Metrics:

- mean utility;
- cost at fixed quality;
- oracle regret;
- recovered gap vs learned router;
- recovered gap vs oracle;
- bootstrap confidence intervals;
- dataset-label leakage gap.

Expected result patterns:

| Observation | Meaning | Next step |
|---|---|---|
| Dataset-label strong, predicted-topic weak | benchmark partition leakage | focus on evaluation artifact diagnosis |
| Predicted-topic strong | coarse explainable labels are enough | high-ceiling route-label claim alive |
| Embedding-cluster strong, topic weak | local geometry matters more than names | build learned route codes and compare to kNN |
| RouteCode oracle strong, predicted RouteCode weak | useful codes exist but are hard to infer | build predictability-constrained method |
| All compressed methods weak | full query info needed | pivot to high-rate routing diagnostic |

Pre-committed offensive threshold:

```text
Only claim “routers barely use fine-grained query information” if predicted-topic or predicted-code recovers >=85% of the best learned-router gain, with lower bootstrap CI >=80%.
```

Do not call 60--70% recovery an offensive result.

---

### Pilot B2: first rate--distortion curve

Question:

```text
How does routing regret change as the number of route labels increases?
```

Run K values:

```text
K = 1, 2, 4, 8, 16, 32, 64, 128
```

Compare:

- random labels;
- dataset/domain labels;
- embedding k-means labels;
- utility-vector clusters;
- regret-optimized RouteCode labels;
- full learned router;
- oracle.

Expected result:

- The project is strongest if RouteCode dominates semantic clusters at fixed K and the curve saturates early.

Decision gate:

- If RouteCode does not beat semantic clusters or kNN at similar K, the method contribution is weak. Shift focus to benchmark diagnosis or high-rate routing analysis.

---

### Pilot B3: residual concentration

Question:

```text
When compressed routing fails, are failures concentrated and predictable?
```

Measure:

- fraction of total regret caused by top 5%, 10%, 20% highest-regret queries;
- per-code regret distribution;
- winner entropy per code;
- model-margin per code;
- distance-to-centroid vs regret;
- kNN disagreement vs regret;
- confidence calibration vs regret.

Expected result:

- If a small fraction of queries accounts for most regret, adaptive refinement becomes promising.

Decision gate:

- Only build adaptive refinement if residual regret is concentrated and confidence/disagreement predicts it.

---

### Pilot B4: benchmark split sensitivity

Question:

```text
Are routing results stable, or do they depend on mixed-domain benchmark splits?
```

Evaluate under:

- random split;
- leave-one-dataset-out;
- leave-one-domain-out;
- domain-homogeneous split;
- cluster-held-out split;
- model-pool holdout.

Measure:

- method ranking correlation across splits;
- absolute utility degradation;
- recovered gap change;
- compression rate needed to reach 80% learned-router gain.

Expected result:

- If rankings reorder under controlled splits, benchmark compressibility/evaluation artifact becomes a major paper contribution.

---

## 3. Phase C — observation synthesis

After Phase B, write a short observation memo before implementing more methods.

The memo should answer:

1. Is the benchmark/model pool routable?
2. How compressible is routing?
3. Which compressed representation works best?
4. Is utility-aware RouteCode better than topic/embedding codes?
5. Are failures concentrated?
6. Are results split-sensitive?
7. Which paper claim is alive?

### Claim decision table

| Pilot result | Paper direction |
|---|---|
| Small predicted labels recover >=85% learned-router gain | high-ceiling “few bits are enough” paper |
| RouteCode beats topic/embedding labels but recovery is moderate | rate--distortion + method paper |
| Utility-code oracle strong, predicted-code weak | predictability-constrained code learning paper |
| Residual failures concentrated | adaptive refinement paper section |
| Benchmark rankings reorder | benchmark diagnosis/evaluation paper section |
| Compression weak everywhere | high-rate routing diagnostic / negative result paper |

Do not proceed to full method implementation without this memo.

---

## 4. Phase D — method design after pilot

The method should be selected based on the pilot.

### D1. Flat RouteCode

Use if pilot shows utility-aware labels beat semantic labels.

Method:

```text
learn K route labels from utility matrix -> train q-to-label predictor -> route via label-to-model table
```

Objective:

```text
minimize routing regret + beta * label rate + gamma * label acquisition cost
```

### D2. Predictability-constrained RouteCode

Use if oracle utility labels are strong but hard to predict from text.

Method:

```text
learn labels that are both routing-useful and predictable from query features
```

Combined assignment loss:

```text
routing_regret(q, z) - alpha * log p_h(z | q) + beta * rate_penalty(z)
```

### D3. Explainable route-label cards

Use for all method variants.

Each learned label should have a code card:

```text
label id
short name
best model
second-best model
model margin
common datasets/domains
representative queries
high-regret failure cases
cost/quality profile
human-readable explanation
```

### D4. New-model calibration

Use if labels are stable and interpretable.

Method:

```text
freeze query-to-label predictor
sample r examples per label for new model
estimate new model utility per label
update label-to-model table
```

Sweep:

```text
r = 1, 2, 4, 8, 16, 32, 64 examples per label
```

### D5. Adaptive refinement

Use only if residual failures are concentrated and predictable.

Method:

```text
coarse label -> route if confident
coarse label uncertain -> finer label / kNN / full router / fallback
```

Refine only if expected value of information is positive.

---

## 5. Phase E — method evaluation

Compare RouteCode against open-source baselines.

### Required baselines

- random;
- best single;
- dataset oracle;
- query oracle;
- dataset-label lookup;
- predicted-topic lookup;
- embedding-cluster lookup;
- kNN;
- MLP/SVM;
- RouteLLM if easy;
- GraphRouter / LLMRouter baselines if available;
- Avengers/Avengers-Pro if included in LLMRouterBench.

### Main metrics

- mean utility;
- accuracy/quality;
- cost at fixed quality;
- quality at fixed cost;
- oracle regret;
- recovered gap vs learned router;
- recovered gap vs oracle;
- code rate: log2(K), H(Z);
- code acquisition cost;
- calibration examples needed for new model;
- bootstrap CIs.

---

## 6. Phase F — ablation study

Ablations should test why the method works.

Required ablations:

1. code count K;
2. code objective: semantic vs utility vs regret vs predictability-constrained;
3. predictor type: linear, MLP, ModernBERT/DeBERTa, optional LoRA;
4. cost weight lambda;
5. model pool size/composition;
6. training data size;
7. new-model calibration examples per label;
8. with/without code-card interpretability;
9. with/without adaptive refinement if implemented;
10. with/without rate penalty.

---

## 7. Phase G — sensitivity analysis

Sensitivity tests should test robustness.

Run sensitivity over:

- random seeds;
- embedding backbone;
- clustering algorithm;
- label noise;
- cost mis-estimation;
- train/val/test split strategy;
- domain granularity;
- model price ratios;
- query length buckets;
- dominated vs complementary model pools;
- bootstrap sampling.

---

## 8. Phase H — final paper claims

Only make claims that the experiments support.

### Claim 1: small route labels recover most routing performance

Requires:

- predicted-code/predicted-topic recovery >= threshold;
- CIs;
- comparison to learned routers and oracle.

### Claim 2: route labels transfer across model pools

Requires:

- holdout/add-new-model experiments;
- direct-router retraining baseline under same budget.

### Claim 3: new models need fewer calibration examples

Requires:

- examples-per-label sweep;
- full calibration/retraining baseline;
- clear cost accounting.

### Claim 4: benchmarks are compressible/evaluation artifacts exist

Requires:

- compression ladder;
- split sensitivity;
- ranking reordering or inflated dataset-label performance.

### Claim 5: adaptive refinement improves cost-quality

Requires:

- residual concentration;
- confidence calibration;
- refinement-rate vs utility curve.

Do not claim all five unless all five are supported. The main paper should focus on Claim 1 + Claim 3, with Claim 2 as secondary and Claim 4 as diagnostic. Claim 5 is optional/follow-up unless results are strong.

---

## 9. Codex instruction summary

When working with Codex:

1. Build the synthetic demo first.
2. Run real-data pilot next.
3. Stop and summarize observations.
4. Choose method variant based on observations.
5. Evaluate method.
6. Run ablation and sensitivity.
7. Only then draft claims.

Do not skip from synthetic demo directly to full method complexity.
