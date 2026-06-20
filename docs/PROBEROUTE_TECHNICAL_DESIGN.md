# ProbeRoute++ Technical Design

**Working title:** *Low-Dimensional but Partially Observable: Latent Route States for Sample-Efficient LLM Routing*

**Short names:** ProbeRoute++, Partially Observable RouteCode, Latent Route-State Routing

**Core claim target:**

> LLM routing has a low-dimensional latent decision structure, but this structure is only partially observable from query text. We can measure this observability gap, learn latent route states, use cheap probes to infer those states when needed, and recalibrate new model pools at the route-state level with far fewer labels/evaluations.

---

## 1. Why we are changing the center of the paper

The first RouteCode project found a strong diagnostic result but a weak deployable method.

The key findings from the current RouteCode status document are:

- Low-rate **oracle** route labels/states are very strong.
- Query-predicted route labels are weak.
- D2/predictability-constrained labels improve predictability but lose much of the oracle utility advantage.
- New-model calibration by labels is promising but not yet a main paper-level result.
- Adaptive refinement is not yet supported as a main claim.

Concrete pilot numbers from the status document:

```text
query_oracle mean utility: 0.8966
best_single mean utility: 0.6672
routecode_oracle_labels K=16 mean utility: 0.8897, recovered gap: 0.9699
regret_routecode_oracle_labels K=8 mean utility: 0.8966, recovered gap: 1.0000
routecode_predicted_labels K=16 mean utility: 0.6138, recovered gap: -0.2331
d2_embedding_centroid alpha=3 mean utility: 0.7466, recovered gap: 0.3459
```

Interpretation:

```text
The right low-dimensional route structure exists in hindsight.
But query-only prediction cannot reliably infer it.
```

Therefore the new technical center is not “learn route labels.”

The new technical center is:

> **Partially observable low-dimensional routing.**

---

## 2. Conceptual model

A normal router assumes:

```text
query q -> selected model m
```

Original RouteCode assumed:

```text
query q -> latent route label z -> selected model m
```

ProbeRoute++ assumes:

```text
query q -> uncertain belief over latent route states b(z)
if needed, acquire cheap observation/probe o
query + probe -> updated belief over z
belief over z -> selected model m
```

The latent state `z` is not human-defined. It is learned from query--model utility structure.

The labels are initially just IDs:

```text
z_1, z_2, ..., z_K
```

Only after training do we interpret them using code cards.

---

## 3. Objects and notation

### Query, model, utility

We have queries:

```text
q_i, i = 1..N
```

Candidate models:

```text
m_j, j = 1..M
```

Quality/correctness:

```text
y_ij = quality(q_i, m_j)
```

Cost:

```text
c_ij = cost(q_i, m_j)
```

Utility:

```text
U(q_i, m_j) = y_ij - lambda_cost * c_ij - lambda_latency * latency_ij
```

Start with `lambda_cost = 0` for quality-only replication of current results. Then sweep cost-aware settings.

### Provider and Cost Scope

The current Phase 2 implementation defaults to local serving, preferably through vLLM, so the system can run without external API keys. This is only the first compute regime. The cost-aware routing story must also keep a closed-source provider regime in scope:

- OpenAI GPT-family models;
- Anthropic Claude-family models;
- Google Gemini-family models;
- local/open-weight models served through vLLM, llama.cpp, SGLang, or equivalent local OpenAI-compatible servers.

Closed-source provider experiments are not enabled by default. Before any GPT/Claude/Gemini run, explicitly configure API access and budget, refresh provider prices, record the checked date and source URL, and save the exact model IDs. Cost accounting must keep these terms separate:

- target-model input/output token cost;
- probe-model input/output token cost;
- local GPU latency/cost proxy;
- end-to-end latency;
- calibration/evaluation examples required to add a new model.

Do not frame this as saving router tokens. The important cost questions are model-selection utility under provider prices, calibration sample efficiency, and whether cheap probes pay for themselves after all target/probe costs are counted.

### Query oracle

```text
m*(q_i) = argmax_j U(q_i, m_j)
```

This is an upper bound and not deployable.

### Latent route state

```text
z_i in {1, ..., K}
```

A route state groups queries with similar model-selection utility behavior.

### State-to-model utility table

```text
mu[z, m] = E[U(q, m) | z(q) = z]
```

The best model for state `z` is:

```text
pi(z) = argmax_m mu[z, m]
```

### Query-to-state belief

At inference time, we usually do not know `z`.

We have a belief:

```text
b0(z) = p(z | q)
```

### Probe observation

A probe action `a` returns an observation `o`:

```text
o = probe_a(q)
```

Examples:

- kNN neighbor disagreement;
- cheap local model confidence;
- short local model draft;
- token logprob or entropy;
- small verifier score;
- agreement among cheap models.

After observing `o`:

```text
b1(z) = p(z | q, o, a)
```

---

## 4. The observability gap

Define two values.

### Oracle-state value

This assumes the correct latent state is known:

```text
V_oracle_state = E_q[ U(q, pi(z_oracle(q))) ]
```

### Predicted-state value

This uses a deployable query-to-state predictor:

```text
V_predicted_state = E_q[ U(q, pi(z_pred(q))) ]
```

### Observability gap

```text
ObservabilityGap = V_oracle_state - V_predicted_state
```

Plain English:

> How much routing utility is lost because the useful low-dimensional state exists but cannot be inferred from query text?

This is the central phenomenon.

Current pilot suggests this gap is large.

---

## 5. Latent route-state learning

We learn states from the utility matrix, not from human labels.

### 5.1 Routing-regret term

For a query assigned to state `z`, the regret is:

```text
regret(i, z) = U_i^* - U[i, pi(z)]
```

where:

```text
U_i^* = max_m U[i, m]
```

The state assignment should minimize average regret.

### 5.2 Rate/complexity term

We do not want too many states.

```text
Rate = log2(K) or empirical entropy H(Z)
```

### 5.3 Calibration-variance term

To make new-model calibration cheap, states should have low within-state utility variance.

For state `z` and model `m`:

```text
Var_zm = Var_{q: z(q)=z}( U(q, m) )
```

Calibration variance loss:

```text
L_calib = sum_z sum_m Var_zm
```

Plain English:

> A state is useful for future calibration if model performance is stable inside the state.

This is a key technical upgrade over simple latent clustering.

### 5.4 Observability/predictability term

A state must be at least partially inferable from cheap query features or cheap probes.

Let `h_theta(q)` predict state probabilities.

```text
L_obs = CE(z_i, h_theta(q_i))
```

or a routing-weighted version:

```text
L_obs_util = U_i^* - sum_z p_theta(z|q_i) * U[i, pi(z)]
```

### 5.5 Full state-learning objective

A practical objective:

```text
min_{z, pi, h}
  L_route
  + beta * Rate
  + eta * L_calib
  + alpha * L_obs
```

where:

```text
L_route = average routing regret by state
Rate = state complexity
L_calib = within-state utility variance
L_obs = predictability/observability loss
```

This makes the state learner solve three things at once:

1. good current routing;
2. cheap future calibration;
3. partial observability.

---

## 6. State learning algorithms

### 6.1 Baseline utility clustering

1. Create utility vector for each query:

```text
U_i = [U(q_i, m_1), ..., U(q_i, m_M)]
```

2. Cluster utility vectors.
3. Choose best model per cluster.
4. Evaluate oracle assignment and predicted assignment.

This is a diagnostic method.

### 6.2 Regret-optimized state learning

Alternating optimization:

1. Initialize states using utility clustering or embedding clustering.
2. Update state-to-model table:

```text
pi(z) = argmax_m sum_{i:z_i=z} U[i,m]
```

3. Reassign each query:

```text
z_i = argmin_z [ U_i^* - U[i, pi(z)] + eta * calibration_penalty(i,z) ]
```

4. Repeat until convergence.

### 6.3 Calibration-aware state learning

Add penalty for high within-state variance.

When assigning query `i` to state `z`, penalize states where adding `i` increases utility variance across models.

Approximate assignment loss:

```text
loss(i,z) = routing_regret(i,z)
          + eta * delta_within_state_variance(i,z)
          + beta * rate_penalty(z)
```

This should improve new-model calibration.

### 6.4 Predictability-aware state learning

After a state assignment exists, train a query-to-state model.

Then reassign labels using:

```text
loss(i,z) = routing_regret(i,z)
          + eta * variance_penalty(i,z)
          - alpha * log p_h(z | q_i)
          + beta * rate_penalty(z)
```

This is more principled than D2 if we keep routing and calibration terms alive.

### 6.5 Latent state interpretation

After learning states, generate state cards.

For each state:

```json
{
  "state_id": 7,
  "posthoc_name": "hard symbolic reasoning / math-heavy",
  "best_model": "frontier_model",
  "second_best_model": "math_7b",
  "mean_utility_by_model": {},
  "within_state_variance_by_model": {},
  "traffic_mass": 0.08,
  "representative_queries": [],
  "dominant_datasets": [],
  "failure_cases": [],
  "probe_sensitivity": {},
  "new_model_calibration_status": {}
}
```

Do not define labels first. Learn states first, explain later.

---

## 7. ProbeRoute++ online routing

### 7.1 Initial state belief

Given a new query:

```text
b0(z) = p(z | q)
```

This can come from:

- embedding + MLP;
- ModernBERT/DeBERTa classifier;
- kNN over state-labeled training examples;
- ensemble of above.

### 7.2 Current routing value

Given belief `b`, expected utility of model `m` is:

```text
EU(m | b) = sum_z b(z) * mu[z,m]
```

Best current value:

```text
V(b) = max_m EU(m | b)
```

Route immediately if belief is confident and expected regret is low.

### 7.3 Probe actions

Possible probe actions:

#### Probe A: kNN uncertainty

No generation.

Features:

- label entropy among nearest neighbors;
- oracle winner entropy among nearest neighbors;
- local margin between best two models;
- distance to centroid;
- distance to boundary.

Cost: nearly zero.

#### Probe B: cheap local model confidence

Run one local small model with a short response or confidence prompt.

Features:

- self-rated confidence;
- output length;
- answer format validity;
- average token logprob if available;
- entropy if available;
- exact correctness if task has answer and this is an offline experiment.

Cost: local GPU time.

#### Probe C: short draft

Generate max 32 or 64 tokens.

Features:

- draft embedding;
- draft confidence;
- answer candidate;
- whether draft is incomplete;
- verifier score.

Cost: local GPU time and small latency.

#### Probe D: cheap verifier

Use a small classifier/verifier to estimate whether the cheap model can solve the query.

Cost: encoder inference.

### 7.4 Value of information

For a probe action `a`:

```text
VOI(a) = E_o[ V(b_after(q,o,a)) ] - V(b0) - C(a)
```

Run probe only if:

```text
max_a VOI(a) > 0
```

Approximate `E_o` in practice using validation data:

1. Bucket queries by current belief features.
2. Estimate average value gain for each probe type.
3. Train a regressor for expected improvement.
4. Subtract measured probe cost.

### 7.5 Posterior update

After probe observation `o`, update belief:

```text
b1(z) = p(z | q, o, a)
```

Implementation options:

- train classifier on `[query features, probe features]`;
- likelihood model `p(o|z,a)` + Bayes update;
- calibration table by bins;
- gradient-boosted tree over belief + probe features.

### 7.6 Final model choice

Choose:

```text
m* = argmax_m sum_z b_final(z) * mu[z,m]
```

The final decision remains mediated by state belief. Do not bypass states and map directly to models, or we lose the calibration benefit.

---

## 8. New-model active calibration

This is the most important deployment benefit.

### 8.1 Problem

A new model enters the pool:

```text
m_new
```

We need to know where it should be used.

A full query-level calibration would evaluate it on many queries.

ProbeRoute++ instead estimates:

```text
mu[z, m_new]
```

for each latent state `z`.

### 8.2 Naive state calibration

Sample `r` examples per state:

```text
K states * r examples per state
```

Example:

```text
K=16, r=16 -> 256 evaluations for new model
K=32, r=16 -> 512 evaluations for new model
```

### 8.3 Bayesian utility posterior

For binary correctness:

```text
theta[z, m_new] ~ Beta(alpha_z, beta_z)
```

For continuous utility:

```text
mu[z, m_new] ~ Normal(mean_z, variance_z)
```

Maintain posterior uncertainty per state.

### 8.4 Value of calibration

A calibration query is valuable if it can change the routing decision in a high-traffic state.

For state `z`:

```text
VOC(z) = traffic_mass(z)
       * P(new model could become best for z)
       * expected utility improvement if ranking changes
       * uncertainty(z)
```

Sample more from states with high `VOC(z)`.

### 8.5 Active calibration algorithm

1. Initialize with 1--2 examples per state.
2. Estimate posterior over `mu[z, m_new]`.
3. Compute `VOC(z)` for each state.
4. Sample next query from highest-value state.
5. Update posterior.
6. Stop when budget is exhausted or routing table is stable.

### 8.6 Comparison baselines

Compare against:

- random calibration examples;
- dataset-stratified calibration;
- embedding-cluster calibration;
- direct query-to-model router retraining;
- model-feature transfer baseline if implemented;
- full calibration upper bound.

### 8.7 Main metric

```text
routing utility vs number of new-model evaluations
```

This is stronger than saying routing inference is cheap.

### 8.8 Current implementation status: does ASC work?

Short answer:

```text
ASC works as an implemented cached new-model onboarding mechanism.
ASC is not yet proven as a broad benchmark-level research claim.
```

The implemented module is **Active State Calibration (ASC)**. It is not a
query router. It is the model-pool update layer used when a new model enters
the pool.

Current data flow:

```text
frozen query -> state labels
old state -> model utility table
held-out new model m_new
candidate calibration queries per state
    -> ASC selects calibration queries
    -> reveal or run m_new only on selected queries
    -> update posterior U(state, m_new)
    -> conservatively update state -> selected model table
```

The implemented ASC code lives in:

```text
src/routecode/eval/new_model_calibration.py
```

Main implemented functions:

- `active_state_calibration_priority`: builds a state-level posterior and
  ranks states by value of calibration.
- `sample_active_state_calibration_queries`: selects scout examples first,
  then allocates remaining budget to high-value states.
- `conservative_state_model_update`: switches a state to the new model only
  when expected gain and confidence clear thresholds.
- `calibrate_new_model_by_active_state`: cached end-to-end onboarding loop for
  a held-out new model.

The active branch of the Phase 2 calibration script now uses ASC:

```text
experiments/55_active_new_model_calibration.py
method = active_route_state_calibration
```

What has been tested:

```text
pytest -q tests/test_new_model_calibration.py
```

Result:

```text
14 passed
```

This covers:

- ASC prioritizes common, uncertain states where the new model could improve
  utility.
- Query selection uses scout examples plus the proposed weighted score:

```text
0.5 * representativeness
+ 0.3 * uncertainty
+ 0.2 * routing impact
```

- Conservative table update switches only when both margin and confidence pass.
- The cached end-to-end ASC loop can onboard a synthetic held-out new model and
  update only the state where sampled evidence supports switching.

Script smoke test:

```text
python experiments/55_active_new_model_calibration.py \
  --config configs/synthetic.yaml \
  --output-dir /tmp/routecode_active_state_calibration_smoke \
  --max-holdout-models 1 \
  --r-values 1,2
```

Result:

```text
table_active_new_model_calibration.csv
fig_new_model_calibration_curve.pdf
m6_active_new_model_calibration_memo.md
README.md
```

Observed synthetic smoke rows for `active_route_state_calibration`:

| examples_per_label | new-model evaluations | mean utility | labels switched to new model |
|---:|---:|---:|---:|
| 1 | 16 | 0.586741 | 0 |
| 2 | 32 | 0.624229 | 1 |

Interpretation:

- The implementation behaves as intended: with very little evidence it can
  refuse to switch, and with more evidence it can switch a state to the new
  model.
- The smoke run is not enough to claim ASC beats all baselines. In this tiny
  synthetic setting, uniform/random/dataset calibration can still be stronger
  for the selected held-out model.
- Therefore the correct current claim is:

```text
ASC is implemented and mechanically validated.
Broad benchmark evidence is still required before claiming ASC is the best
new-model calibration strategy.
```

Required next evaluation before a paper claim:

- run ASC on Broad100 / LLMRouterBench cached outcome matrices;
- compare against random, dataset-stratified, embedding-cluster, uniform
  route-state calibration, and budget-matched direct retraining;
- report utility vs new-model evaluations for multiple held-out models;
- include sensitivity over `K`, calibration budget, `delta`, `tau`, and prior
  strength;
- keep the claim diagnostic unless ASC consistently beats baselines under
  matched budget.

---

## 9. True model running plan

This phase introduces real local model calls rather than relying only on released outcome matrices.

### 9.1 Why run true models?

Released matrices gave the first observations. True local model runs let us:

1. collect probe signals not available in the benchmark;
2. validate the method with current local models;
3. measure probe latency/cost;
4. test new-model calibration with an actual held-out model;
5. control scoring and prompts.

### 9.2 Recommended local model pool

Use 4--6 local models first, not 20.

Core candidates:

```text
Qwen3-8B
Qwen2.5-Coder-7B-Instruct
DeepSeek-R1-Distill-Qwen-7B
Llama-3.1-8B-Instruct
MiniCPM4.1-8B
Gemma-3-4B or similar smaller model
```

Optional stronger/larger model if memory allows:

```text
Qwen3-14B or coder 14B quantized
```

### 9.3 Recommended datasets for true evaluation

Prioritize exact scoring to avoid LLM judge costs.

Start with:

```text
GSM8K / MATH500 / AIME-style numeric math
MMLU-Pro / GPQA multiple choice
HumanEval / MBPP code generation, if sandboxed evaluation is ready
```

Avoid open-ended writing tasks until evaluation infrastructure is reliable.

### 9.4 Prompt and generation settings

Use deterministic settings first:

```text
temperature = 0
max_new_tokens = task-specific
seed fixed if engine supports it
```

Store every prompt, output, parsed answer, score, latency, token counts, and model version.

### 9.5 Probe collection settings

For each query, collect cheap probes from one or two local models:

```text
max_new_tokens = 32 or 64
confidence-only prompt if supported
short answer/draft prompt
logprob/entropy if supported by serving backend
```

Do not run one probe per state. Use generic probes that update belief over all states.

---

## 10. Main experiments for the new paper

### Experiment 1: latent route-state frontier

Show oracle-state frontier over K:

```text
K = 1, 2, 4, 8, 16, 32, 64
```

Across:

- pilot 6-model;
- Broad20;
- Scale20;
- 32-model run;
- true local model run.

### Experiment 2: observability gap with strong encoders

Compare query-only state prediction using:

- hashing features;
- all-MiniLM;
- BGE;
- Qwen embeddings;
- ModernBERT/DeBERTa;
- kNN;
- MLP.

Show whether the gap persists.

### Experiment 3: calibration-aware state learning

Compare states learned by:

- semantic clustering;
- utility clustering;
- regret-only RouteCode;
- regret + calibration variance;
- regret + calibration variance + observability.

Metric:

```text
routing utility
new-model calibration sample efficiency
within-state variance
state interpretability
```

### Experiment 4: ProbeRoute++

Compare:

- query-only predicted state;
- D2;
- kNN;
- uncertainty-only routing;
- always probe;
- entropy-threshold probe;
- VOI ProbeRoute++;
- oracle state;
- query oracle.

Metric:

```text
observability gap closed per unit probe cost
```

### Experiment 5: active new-model calibration

For held-out/new models, compare:

- random examples;
- dataset-stratified examples;
- embedding-cluster examples;
- route-state uniform examples;
- active route-state examples;
- direct router retraining.

Metric:

```text
utility vs number of new-model evaluations
```

### Experiment 6: ablations and sensitivity

Ablate:

- K;
- state objective terms;
- probe type;
- probe cost;
- VOI vs threshold;
- posterior update method;
- calibration budget;
- local model pool;
- dataset split;
- cost weight lambda.

---

## 11. Claims allowed after successful experiments

### Strong claim

> LLM routing has a low-dimensional latent decision structure, but this structure is partially observable from query text.

Requires:

- oracle-state frontier strong;
- predicted-state frontier much weaker;
- gap persists with strong encoders.

### Method claim

> ProbeRoute++ closes a meaningful fraction of the observability gap using cheap probes.

Requires:

- ProbeRoute++ beats query-only and threshold baselines;
- cost-adjusted gains remain positive;
- ablations show VOI/probes are responsible.

### Calibration claim

> Latent route states enable new models to be integrated with fewer evaluations.

Requires:

- active route-state calibration beats random/dataset/embedding/direct retraining under matched budget;
- result holds across multiple held-out/new models.

### Interpretability claim

> Learned latent route states can be post-hoc interpreted.

Requires:

- code cards;
- representative examples;
- utility profiles;
- state stability across seeds/splits.

---

## 12. What not to claim

Do not claim:

```text
few inferred bits are enough
```

Current evidence does not support this.

Do not claim:

```text
we are the first to compress routing inputs
```

WebRouter is close.

Do not claim:

```text
we are the first to discover latent tasks/states for routing
```

FineRouter is close.

Do not claim:

```text
routing inference is expensive, and we save router tokens
```

The meaningful cost is calibration and supervision, not router-token inference.

---

## 13. Relationship to the previous RouteCode findings

The old RouteCode conclusion was conservative:

```text
low-rate utility structure exists, but query-to-label prediction is the bottleneck.
```

ProbeRoute++ turns that bottleneck into the new problem:

```text
How can a router cheaply observe the hidden route state?
```

This is the main upgrade from diagnostic framework to method paper.
