# Exact Routing System Spec: ProbeRoute++

This document defines the **exact routing system** we are building for the Phase 3 controlled experiments.

The system is named **ProbeRoute++**.

It is not simply a classifier, not simply a prompt-compression system, and not simply a confidence cascade. It is a cost-aware, latency-aware LLM routing system based on **latent route states** and **optional cheap probing**.

---

## 1. One-sentence system definition

> ProbeRoute++ maps each query to a belief over learned latent route states; if the belief is confident, it routes immediately through a state-to-model utility table; if the belief is uncertain, it runs a cheap probe, updates the state belief, and then selects the model with the best expected quality-cost-latency utility.

The key invariant is:

```text
query/probe -> latent route-state belief -> model
```

not:

```text
query/probe -> model directly
```

This invariant preserves the main benefit: **new models can be calibrated at the route-state level instead of retraining a full query-to-model router.**

---

## 1.1 Implementation contract for the Stage 1 pilot

The Stage 1 pilot must build and evaluate the system in this document, not a simpler proxy router.

The live generation/cache runner is only a data-collection layer. Its job is to produce cached model outputs with quality, cost, and latency. A live routing summary from that runner is useful for debugging, but it is not the main ProbeRoute++ method unless the routing decision flows through latent state beliefs.

After a live pilot cache is complete, the required method path is:

```text
model_outputs/scored_outputs
  -> train-only latent route-state learner
  -> p(z | q) before-probe belief
  -> state-to-model utility table U(z,m)
  -> optional cheap probe and updated p(z | q, probe)
  -> expected-utility model choice
  -> routing_decisions + table_main_eval
```

Concrete implementation entrypoint:

```text
src/routecode/proberoutepp.py
```

Required Stage 1 pilot artifacts from the actual method:

```text
state_model_utility_table.parquet
routing_decisions.parquet
table_main_eval.csv
table_calibration.csv
RUN_REPORT.md
proberoutepp_metadata.json
fig_quality_cost_frontier.pdf
fig_latency_breakdown.pdf
```

Simple policies such as all-frontier, best-local, dataset/domain lookup, answer-agreement rescue, or direct query/probe-to-model gates are baselines or diagnostics. They may appear in reports, but they cannot be labeled as ProbeRoute++ unless they make decisions through:

```text
query/probe -> latent route-state belief -> state-to-model utility -> selected model
```

Stage 1 is not complete until the live cached pilot outputs have been passed through this path and the resulting `table_main_eval.csv` reports the no-probe, threshold-probe, and VOI-probe variants against the cost-aware oracle.

---

## 2. High-level system diagram

```text
                                      Offline phase
                         ┌─────────────────────────────────┐
                         │ query-model outcome matrix       │
                         │ quality, cost, latency           │
                         └───────────────┬─────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │ learn latent route states z=1..K │
                         │ objective: regret + calibration  │
                         │ variance + rate/predictability   │
                         └───────────────┬─────────────────┘
                                         │
                         ┌───────────────┴─────────────────┐
                         ▼                                 ▼
        ┌─────────────────────────────┐    ┌─────────────────────────────┐
        │ train query-to-state model   │    │ build state-to-model table   │
        │ p(z | q)                     │    │ U(z, m)                      │
        └───────────────┬─────────────┘    └───────────────┬─────────────┘
                        │                                  │
                        └────────────────┬─────────────────┘
                                         ▼
                                  Online routing

User query q
    │
    ▼
┌───────────────────────────────┐
│ Step 1: predict b0(z)=p(z|q)   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Step 2: compute route value    │
│ V(b0)=max_m Σ_z b0(z)U(z,m)   │
└───────────────┬───────────────┘
                │
                ▼
     Is state belief confident / is probe VOI <= 0?
             │                         │
             │ yes                     │ no
             ▼                         ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│ route immediately        │   │ Step 3: run cheap probe a*   │
│ m*=argmax_m E[U(z,m)]    │   │ local small model / kNN etc. │
└─────────────┬───────────┘   └──────────────┬──────────────┘
              │                              │
              │                              ▼
              │                ┌─────────────────────────────┐
              │                │ Step 4: update belief        │
              │                │ b1(z)=p(z|q,probe output)    │
              │                └──────────────┬──────────────┘
              │                              │
              │                              ▼
              │                ┌─────────────────────────────┐
              │                │ Step 5: choose best model    │
              │                │ m*=argmax_m Σ_z b1(z)U(z,m) │
              │                └──────────────┬──────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
                    selected model generates answer
```

---

## 3. What the router input is

At inference time, the router receives:

```json
{
  "query_id": "example_001",
  "query_text": "Write a Python function to find the longest palindromic substring.",
  "candidate_models": [
    "qwen3.5-0.8b",
    "qwen3.5-9b",
    "qwen3-coder-30b-a3b",
    "qwen3.6-35b-a3b",
    "gemma-3-12b-it",
    "gpt-5.5",
    "gemini-3.5-flash"
  ],
  "lambda_cost": 0.5,
  "lambda_latency": 0.1,
  "sla_latency_ms": 10000,
  "max_frontier_cost_usd": 0.01
}
```

The query is not answered by the router. The router only chooses which model should answer.

---

## 4. What the router output is

The router outputs:

```json
{
  "query_id": "example_001",
  "selected_model": "qwen3-coder-30b-a3b",
  "state_distribution": {
    "z03": 0.82,
    "z11": 0.10,
    "z04": 0.08
  },
  "selected_state": "z03",
  "state_confidence": 0.82,
  "probe_used": false,
  "probe_type": null,
  "expected_quality": 0.86,
  "expected_remote_cost_usd": 0.0,
  "expected_latency_ms": 2500,
  "decision_reason": "High confidence in code-oriented route state; local code model has best cost-aware utility."
}
```

If probing is used:

```json
{
  "query_id": "example_002",
  "selected_model": "gpt-5.5",
  "state_distribution_before_probe": {
    "z04": 0.34,
    "z06": 0.31,
    "z13": 0.20,
    "z02": 0.15
  },
  "probe_used": true,
  "probe_type": "qwen3.5-0.8b_short_confidence_probe",
  "probe_cost_usd": 0.0,
  "probe_latency_ms": 420,
  "state_distribution_after_probe": {
    "z06": 0.67,
    "z04": 0.18,
    "z13": 0.10,
    "z02": 0.05
  },
  "selected_state": "z06",
  "state_confidence": 0.67,
  "expected_remote_cost_usd": 0.0042,
  "expected_latency_ms": 6400,
  "decision_reason": "Cheap probe indicated hard reasoning; frontier model has best expected utility."
}
```

---

## 5. Candidate model pool for the controlled run

The controlled Phase 2 run uses **4-5 modern local models plus 2 frontier models**.

### Local models

Preferred local models:

```text
1. Qwen/Qwen3.5-0.8B
   Role: cheap probe model / tiny local baseline.

2. Qwen/Qwen3.5-9B
   Role: general local answer model.

3. Qwen/Qwen3-Coder-30B-A3B-Instruct
   Role: code specialist.

4. Qwen/Qwen3.6-35B-A3B
   Role: strong local general/reasoning model.

5. Gemma-3-12B-it OR Mistral-Small-3.2-24B-Instruct-2506
   Role: diverse non-Qwen local model.
```

If hardware is tight, use 4 local models:

```text
Qwen3.5-0.8B
Qwen3.5-9B
Qwen3-Coder-30B-A3B-Instruct
Gemma-3-12B-it or Mistral-Small-3.2-24B
```

### Frontier models

```text
1. GPT-5.5 or the latest available GPT API model.
2. Gemini 3.5 flash or the latest available gemini API model.
```

### Important serving note

The offline evaluation matrix can run models sequentially. We do **not** need to host all local models simultaneously on one GPU.

The live demo can keep only:

```text
1 cheap probe model + 1 strong local answer model + frontier APIs
```

---

## 6. Benchmarks for the controlled run

Use exact-scored benchmarks first, avoiding LLM judges when possible.

Recommended benchmark set:

```text
1. GSM8K                     math, exact answer
2. MATH500                   harder math, exact answer
3. AIME                      competition math, exact answer
4. HumanEval                 code generation, pass@1
5. MBPP                      code generation, pass@1
6. LiveCodeBench subset      harder modern coding
7. GPQA                      science reasoning / multiple choice
8. MMLU-Pro                  broad knowledge / multiple choice
9. BBH or logical reasoning  optional reasoning benchmark
```

Use smaller subsets first:

```text
50-100 examples per benchmark for dry run.
200-500 examples per benchmark for pilot.
Full or larger subsets for final run if cost allows.
```

---

## 7. Core mathematical objective

For a query `q` and model `m`, define:

```text
U(q,m) = quality(q,m)
         - lambda_cost * normalized_remote_cost(q,m)
         - lambda_latency * normalized_latency(q,m)
```

For local models:

```text
remote cost = 0
local latency is measured separately
```

For frontier models:

```text
remote cost = input_tokens * input_price + output_tokens * output_price
latency = measured API latency
```

Latent route states:

```text
z in {1, 2, ..., K}
```

State-to-model utility:

```text
U(z,m) = average utility of model m on training queries assigned to state z
```

Initial state belief:

```text
b0(z) = p(z | q)
```

Route value without probe:

```text
V(b0) = max_m sum_z b0(z) * U(z,m)
```

Probe value:

```text
VOI(a) = E_o[V(b_after_probe)] - V(b0) - C(a)
```

Probe only if:

```text
max_a VOI(a) > 0
```

Final model selection:

```text
m* = argmax_m sum_z b(z) * U(z,m)
```

---

## 8. Offline stage in detail

### Stage 1: build query-model outcome matrix

For each benchmark query and each model, collect:

```text
query_id
benchmark
query_text
model_id
raw_output_path
parsed_answer
quality_score
correctness
input_tokens
output_tokens
remote_cost_usd
latency_ms
is_frontier
is_local
cache_key
```

Output file:

```text
results/controlled/model_outputs.parquet
results/controlled/scored_outputs.parquet
```

### Stage 2: compute utility matrix

Build matrices:

```text
Y[N, M] = quality/correctness
C[N, M] = remote cost
T[N, M] = latency
U[N, M] = quality - lambda_cost*C - lambda_latency*T
```

### Stage 3: learn latent route states

Learn `K` route states from the train utility matrix.

Candidate methods:

```text
1. utility-vector clustering
2. regret-optimized state assignment
3. calibration-aware state assignment
```

The final state objective should balance:

```text
routing regret
+ route-state rate / K penalty
+ within-state utility variance
+ observability/predictability regularization
```

Plain English:

```text
States should preserve routing utility, be few in number, have stable model behavior inside each state, and be inferable from query/probe features.
```

### Stage 4: train query-to-state belief model

Train:

```text
p(z | q)
```

Recommended implementations:

```text
embedding + MLP
kNN over embeddings
ModernBERT / DeBERTa if time allows
```

Do not start with LoRA.

### Stage 5: learn probe update model

For each probe feature set, train an updater:

```text
p(z | q, probe_features)
```

Simple first version:

```text
concatenate query embedding + probe features -> MLP -> p(z)
```

More principled version:

```text
Bayesian / calibrated logistic update over state probabilities
```

### Stage 6: build state-to-model utility table

For each state `z` and model `m`:

```text
mean_quality(z,m)
mean_remote_cost(z,m)
mean_latency(z,m)
mean_utility(z,m)
confidence interval
number of training examples
```

Output:

```text
results/controlled/state_model_utility_table.parquet
```

---

## 9. Online routing algorithm

Algorithm: **ProbeRoute++ Online Routing**

```python
def route(query, candidate_models, lambda_cost, lambda_latency, budget):
    # 1. Query-only state belief
    b0 = state_predictor.predict_proba(query)

    # 2. Current best model without probing
    value_no_probe, model_no_probe = expected_best_model(
        belief=b0,
        state_model_table=U_state_model,
        lambda_cost=lambda_cost,
        lambda_latency=lambda_latency,
    )

    # 3. Decide whether to probe
    probe_candidates = get_available_probes(query, budget)
    voi_scores = []
    for probe in probe_candidates:
        voi = estimate_value_of_information(
            query=query,
            belief=b0,
            probe=probe,
            current_value=value_no_probe,
        )
        voi_scores.append((probe, voi))

    best_probe, best_voi = max(voi_scores, key=lambda x: x[1])

    if best_voi <= 0:
        return RoutingDecision(
            selected_model=model_no_probe,
            state_belief=b0,
            probe_used=False,
        )

    # 4. Run cheap probe
    probe_output = run_probe(best_probe, query)

    # 5. Update state belief
    b1 = probe_state_updater.predict_proba(query, probe_output)

    # 6. Select final model
    final_value, final_model = expected_best_model(
        belief=b1,
        state_model_table=U_state_model,
        lambda_cost=lambda_cost,
        lambda_latency=lambda_latency,
    )

    return RoutingDecision(
        selected_model=final_model,
        state_belief=b1,
        probe_used=True,
        probe_type=best_probe.name,
        expected_utility=final_value,
    )
```

---

## 10. Cheap probes

ProbeRoute++ should start with a small number of cheap probes.

### Probe 1: non-generative uncertainty probe

No model generation.

Features:

```text
state predictor entropy
top-1 minus top-2 state margin
kNN state entropy
kNN oracle-winner entropy
distance to nearest state centroid
embedding cluster margin
```

Cost:

```text
near zero
```

### Probe 2: cheap local short-draft probe

Use the cheapest local model, e.g. Qwen3.5-0.8B.

Prompt:

```text
Answer briefly. If unsure, still answer.
```

Settings:

```text
max_new_tokens: 16 or 32
temperature: 0
logprobs: true if backend supports
```

Features:

```text
mean token logprob
token entropy
output length
answer format validity
self-rated confidence if prompted
```

### Probe 3: domain-specific local probe, optional

For code-like queries, use Qwen3-Coder short probe.

Features:

```text
syntax validity
unit-test quick result if cheap
code output confidence
```

Only run this probe when the query appears code-like and the VOI is positive.

---

## 11. New-model calibration algorithm

When a new model arrives, we do **not** retrain the full router.

We keep:

```text
query -> state belief
```

Then calibrate:

```text
state -> new model utility
```

### Calibration protocol

For each latent state `z`:

```text
sample r calibration examples
run new model
score outputs
estimate U(z, new_model)
update state-model table
```

Default:

```text
K = 8 or 16
r = 4, 8, 16, 32 examples per state
```

So:

```text
K=16, r=16 -> 256 evaluations
K=16, r=32 -> 512 evaluations
```

### Active calibration

Do not sample uniformly forever.

Sample more from states where:

```text
traffic mass is high
new model may beat current best
posterior uncertainty is high
expected utility gain is high
```

Value of calibration for state `z`:

```text
VOC(z) = traffic_mass(z)
         * probability_new_model_changes_decision(z)
         * expected_utility_gain(z)
         / expected_calibration_cost(z)
```

Sample states with highest VOC.

---

## 12. Cost and latency accounting

Every model/probe call must log:

```text
request_id
query_id
model_id
is_frontier
is_probe
backend
input_tokens
output_tokens
remote_cost_usd
start_time
end_time
latency_ms
cache_hit
error_type
```

### Main cost metrics

```text
remote_cost_per_query
remote_cost_per_1k_queries
remote_cost_per_1m_queries
normalized_cost_vs_all_gpt
normalized_cost_vs_all_claude
frontier_call_rate
probe_call_rate
local_model_usage_rate
```

### Main latency metrics

```text
mean latency
p50 latency
p90 latency
p95 latency
p99 latency
router_latency
probe_latency
answer_model_latency
end_to_end_latency
```

### Expected cost target

```text
ProbeRoute++ normalized remote cost <= 0.15x--0.35x all-frontier.
```

### Expected latency target

```text
ProbeRoute++ p95 latency <= all-frontier p95
or <= 1.2x all-frontier p95 if quality/cost are much better.
```

---

## 13. Evaluation methods and baselines

### Reference policies

```text
All local best
All GPT frontier
All Claude frontier
Best single model overall
Query oracle
Cost-aware oracle
```

### Simple routing baselines

```text
dataset/domain lookup
embedding cluster lookup
kNN router
direct MLP/BERT router
```

### Cascade / confidence baselines

```text
cheap-local confidence cascade
always-probe cascade
entropy-threshold cascade
```

### ProbeRoute++ variants

```text
ProbeRoute++ no-probe
ProbeRoute++ threshold-probe
ProbeRoute++ VOI-probe
ProbeRoute++ active calibration
```

---

## 14. Expected main results

Strong target:

```text
1. ProbeRoute++ is within 3 absolute quality points of the cost-aware oracle.
2. ProbeRoute++ uses <=0.15x--0.35x all-frontier remote API cost.
3. ProbeRoute++ frontier-call rate <=25%--40%.
4. ProbeRoute++ probe rate <=20%--40%.
5. ProbeRoute++ p95 latency <= all-frontier p95 or <=1.2x all-frontier p95.
6. ProbeRoute++ reaches direct-router/new-model performance with 3x--5x fewer calibration evaluations.
7. Main method does not require LoRA or extensive router training.
```

Minimum acceptable method claim:

```text
At matched quality, ProbeRoute++ reduces remote cost by >=30% vs strongest baseline.
OR at matched cost, ProbeRoute++ improves quality by >=2--4 absolute points vs strongest baseline.
```

Do not claim SOTA unless the method is on the best quality-cost-latency Pareto frontier.

---

## 15. Exact example walkthroughs

### Example 1: easy general query

Query:

```text
What is photosynthesis?
```

Initial state belief:

```text
z01 easy factual/simple explanation: 0.88
z04 science hard reasoning: 0.07
others: 0.05
```

Action:

```text
No probe.
```

Selected model:

```text
Qwen3.5-9B or Qwen3.5-0.8B depending on quality-cost table.
```

Why:

```text
High confidence and local model has best cost-aware utility.
```

---

### Example 2: routine code query

Query:

```text
Write a Python function to merge two sorted linked lists.
```

Initial belief:

```text
z03 routine code generation: 0.81
z07 hard code/debugging: 0.11
others: 0.08
```

Action:

```text
No probe.
```

Selected model:

```text
Qwen3-Coder-30B-A3B-Instruct
```

Why:

```text
The code route state maps to the local code specialist under cost-aware utility.
```

---

### Example 3: hard ambiguous reasoning

Query:

```text
Explain why this proof of the algorithm's correctness is wrong.
```

Initial belief:

```text
z03 routine code: 0.27
z06 proof reasoning: 0.26
z07 hard debugging: 0.24
z12 ambiguous boundary: 0.23
```

Action:

```text
Probe used.
```

Probe:

```text
Qwen3.5-0.8B short-draft + confidence/logprob.
```

Probe output features:

```text
low confidence
high entropy
self-rating: hard reasoning
```

Updated belief:

```text
z06 proof reasoning: 0.63
z07 hard debugging: 0.18
z12 ambiguous boundary: 0.13
z03 routine code: 0.06
```

Selected model:

```text
GPT-5.5 or Claude Sonnet 4.6
```

Why:

```text
After probing, the expected utility of frontier model is highest.
```

---

### Example 4: new model calibration

New model:

```text
Qwen3-Coder-Next
```

Existing states:

```text
K = 16
```

Calibration:

```text
sample 16 examples per state
16 * 16 = 256 model calls
```

After scoring:

```text
state z03 routine code: new model beats old code model
state z07 hard debugging: frontier still best
state z01 easy general: cheap local still best
```

Update table:

```text
z03 -> Qwen3-Coder-Next
other states unchanged
```

No full router retraining required.

---

## 16. What the system is not

ProbeRoute++ is not:

```text
1. a prompt compression method;
2. a manually defined topic taxonomy;
3. a direct query-to-model classifier;
4. a confidence cascade only;
5. a system that probes every label separately;
6. a system that always calls a small LLM before routing;
7. a system that requires LoRA or large router training in the main method.
```

ProbeRoute++ is:

```text
latent state learning + partial observability + VOI probing + state-level model calibration.
```

---

## 17. Implementation deliverables

Codex should implement the system as modules:

```text
src/proberoute/data/
  benchmark_loader.py
  output_schema.py
  scoring.py

src/proberoute/models/
  local_server.py
  frontier_client.py
  token_cost.py

src/proberoute/states/
  learn_states.py
  state_table.py
  calibration_variance.py

src/proberoute/router/
  query_state_predictor.py
  probe_policy.py
  voi.py
  online_router.py

src/proberoute/calibration/
  new_model_calibration.py
  active_calibration.py

src/proberoute/eval/
  metrics.py
  cost_latency.py
  plots.py
```

Required outputs:

```text
results/controlled/state_model_utility_table.parquet
results/controlled/routing_decisions.parquet
results/controlled/cost_latency_summary.csv
results/controlled/table_main_eval.csv
results/controlled/table_calibration.csv
results/controlled/fig_quality_cost_frontier.pdf
results/controlled/fig_latency_breakdown.pdf
results/controlled/RUN_REPORT.md
```

---

## 18. Claim gates

Only claim near-SOTA / SOTA if:

```text
1. quality gap to cost-aware oracle <= 3 absolute points;
2. normalized remote cost <= 0.35x all-frontier;
3. ProbeRoute++ Pareto-dominates strongest baseline;
4. p95 latency <= all-frontier or <=1.2x all-frontier;
5. new-model calibration uses 3x--5x fewer evaluations than direct retraining;
6. method does not rely on LoRA or extensive router training.
```

If these are not met, use weaker wording:

```text
ProbeRoute++ improves the quality-cost-latency tradeoff and reduces calibration burden in selected settings.
```
