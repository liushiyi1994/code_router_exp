# ProbeRoute++ Controlled Phase 2 Experiment Plan

**Purpose:** This document is the concrete experiment specification for the next stage of the ProbeRoute++ / RouteCode research project.

**Core paper direction:**

> LLM routing has a low-dimensional latent decision structure, but that structure is only partially observable from query text. ProbeRoute++ learns cost-aware latent route states, uses cheap probes only when state uncertainty is high, and adapts new model pools through state-level calibration instead of retraining a full query-to-model router.

This document is written for a long-running Codex experiment agent. It should be treated as the source of truth for the actual run.

---

## 0. Why this controlled plan exists

The previous broad experiment plan had too many models, too many baselines, and too many sensitivity axes. The actual paper should be closer to the AIMS-style evaluation design:

- controlled deployment-style model pool;
- many benchmarks;
- cost and latency as first-class metrics;
- focused observations before method;
- focused ablations and threshold sensitivity.

AIMS used a small number of model pairs, evaluated on many benchmarks, and reported accuracy, SLM/local usage, latency breakdown, normalized remote cost, ablations, training-data sensitivity, estimator latency, and threshold sensitivity. We follow the same **research rhythm**, but our technical problem is single-query model routing rather than agent subtask routing.

---

## 1. Main research questions

### RQ1 — Routability

Is the controlled 5-local + 2-frontier model pool actually routable under quality, monetary cost, and latency?

### RQ2 — Low-dimensional latent route states

Can a small number of latent route states, learned from model-utility patterns, preserve most of the cost-aware routing oracle?

### RQ3 — Observability gap

Are these latent route states inferable from query text alone? If not, how large is the gap between oracle state assignment and predicted state assignment?

### RQ4 — ProbeRoute++ method

Can cheap probes close part of the observability gap while preserving the cost/latency benefits of routing?

### RQ5 — New-model calibration

Can a new model be integrated by calibrating its utility per latent route state with far fewer evaluations than direct router retraining?

### RQ6 — Cost/latency deployment value

Does ProbeRoute++ reduce remote frontier API cost and frontier-call rate while preserving quality and keeping local/probe latency acceptable?

---

## 2. Key claim gates

Do not claim more than the experiments support.

### Claim A: low-dimensional latent structure exists

Supported if:

- oracle latent route states with `K = 8` or `K = 16` recover a large fraction of the cost-aware oracle gap;
- this holds across multiple benchmark domains, not only one dataset.

Target:

```text
K = 8 or K = 16 recovers >= 80% of cost-aware oracle improvement over best single.
```

### Claim B: observability gap exists

Supported if:

- oracle route states are much better than query-predicted route states;
- the gap remains after using strong query features/encoders.

Target:

```text
oracle state recovered gap - query-predicted state recovered gap >= 0.25
```

### Claim C: cheap probes help

Supported if:

- ProbeRoute++ improves over query-only state routing and confidence cascade;
- it probes only a minority of queries;
- it improves quality-cost utility, not just accuracy.

Target:

```text
ProbeRoute++ closes >= 25% of the observability gap
while probing <= 40% of queries.
```

### Claim D: state-level calibration reduces new-model burden

Supported if:

- state-level calibration beats random/dataset/embedding calibration under the same number of new-model evaluations;
- it reaches direct-router performance with fewer evaluations.

Target:

```text
3x to 5x fewer new-model evaluations than direct retraining/calibration
for the same cost-aware utility.
```

### Claim E: cost/latency value

Supported if:

- remote frontier cost drops substantially compared with all-frontier baselines;
- probe/router overhead is small relative to model generation;
- p50/p95 latency is acceptable under target deployment setting.

Target:

```text
normalized remote cost <= 0.35x all-frontier
with quality within 2–5 points of all-frontier or substantially above best-local.
```

---

## 3. Controlled model pool

The real run should use **4–5 latest local models + 2 frontier models**. Do not use dozens of models in the main paper.

### 3.1 Main local model pool

The exact set can be adjusted if a model fails to run on RTX 5090, but keep the roles fixed.

| Role | Preferred model | Why this model |
|---|---|---|
| Cheap probe / tiny local baseline | `Qwen/Qwen3.5-0.8B` | Small, modern, useful for cheap confidence/probe signals. HF model card says Qwen3.5-0.8B is intended for prototyping, task-specific fine-tuning, and research/development, and is compatible with Transformers, vLLM, and SGLang. |
| Strong local general | `Qwen/Qwen3.5-9B` | Modern general local model; HF model card says it is compatible with Transformers, vLLM, SGLang, and KTransformers. |
| Code specialist | `Qwen/Qwen3-Coder-30B-A3B-Instruct` or a GGUF/quantized version | Newer code-specialized model; use for HumanEval, MBPP, LiveCodeBench. |
| Strong local reasoning/general | `Qwen/Qwen3.6-35B-A3B` quantized/FP8 if feasible | Newer strong MoE-style local candidate; HF model card says it is compatible with Transformers, vLLM, SGLang, KTransformers. |
| Diverse non-Qwen local model | `google/gemma-3-12b-it` or `mistralai/Mistral-Small-3.2-24B-Instruct-2506` quantized | Non-Qwen diversity. Gemma 3 12B is suitable for text generation, question answering, summarization, and reasoning. Mistral Small 3.2 is a 24B instruction model and a stronger but heavier option. |

### 3.2 Fallback local pool

If VRAM or runtime becomes a problem, use:

```text
Qwen/Qwen3.5-0.8B
Qwen/Qwen3.5-9B
Qwen/Qwen3-30B-A3B-Instruct-2507 or Qwen3-Coder-30B-A3B quantized
Gemma-3-12B-it GGUF
Mistral-Small-3.2-24B quantized, optional
```

If only four local models are feasible, drop Mistral or Gemma, but keep:

```text
cheap probe model
strong general model
code specialist
reasoning/general strong model
```

### 3.3 Frontier models

Use two frontier models:

| Role | Model | Cost notes |
|---|---|---|
| Frontier GPT | `gpt-5.5` or latest GPT API model available | OpenAI pricing should be read from the official pricing page/API docs at run time. The current docs list GPT-5.5 pricing separately for short and long context. |
| Frontier Claude | `claude-sonnet-4-6` or latest Claude Sonnet API model available | Anthropic lists Claude Sonnet 4.6 at $3 / 1M input tokens and $15 / 1M output tokens, with prompt caching/batch discounts available. |

### 3.4 Important serving constraint

Do not try to keep all local models loaded at once on one RTX 5090.

Run local models sequentially and cache outputs:

```text
for model in local_models:
    start server
    run benchmark slice
    save raw outputs + metadata
    stop server
```

For online latency experiments, run a realistic live configuration:

```text
cheap probe model loaded continuously
one strong local answer model loaded continuously
frontier APIs available
```

The full 5-local + 2-frontier matrix is for offline evaluation and cached measurement.

---

## 4. Benchmarks

Use many benchmarks, but keep scoring exact or nearly exact to avoid expensive LLM judging.

### 4.1 Main benchmark set

| Domain | Benchmark | Main metric | Output cap |
|---|---|---:|---:|
| easy math | GSM8K | exact final answer | 512 tokens |
| hard math | MATH500 | exact final answer | 1024 tokens |
| competition math | AIME | exact final answer | 1024 tokens |
| code generation | HumanEval | pass@1 | 1024 tokens |
| code generation | MBPP | pass@1 | 1024 tokens |
| live/modern code | LiveCodeBench subset | pass@1 | 1536 tokens |
| science reasoning | GPQA | multiple choice accuracy | 256 tokens |
| broad knowledge/reasoning | MMLU-Pro | multiple choice accuracy | 256 tokens |
| optional logic | BBH/logical reasoning subset | exact or multiple choice | 512 tokens |

### 4.2 Dataset sizes

Use staged sizes to control cost.

#### Stage 0: dry run

```text
5 examples per benchmark
~40–45 total queries
```

Purpose:

```text
verify prompts, parsing, scoring, cache, cost logging, latency logging
```

#### Stage 1: pilot

```text
100 examples per benchmark
~800–900 total queries
```

Purpose:

```text
first real observation and method results
```

#### Stage 2: main run

```text
200–300 examples per benchmark
~1,600–2,700 total queries
```

Purpose:

```text
paper-quality results if budget allows
```

#### Stage 3: full appendix / optional

```text
larger available subsets for exact-scored local models only
frontier subset may remain capped
```

### 4.3 Sampling rules

Use deterministic sampling with seed.

Stratify by:

```text
benchmark
estimated difficulty if available
question length bucket
```

Keep the same query set for all models.

---

## 5. Cost and latency accounting

Cost and latency are first-class outcomes. Every model call must be logged with tokens, cost, and latency.

### 5.1 Per-call schema

Each row in `model_outputs.parquet` should include:

```text
run_id
query_id
benchmark
domain
model_id
provider
is_local
is_frontier
is_probe
prompt_text_hash
prompt_template_version
input_tokens
output_tokens
max_output_tokens
start_time
end_time
latency_s
queue_time_s, optional
status
error_type, optional
raw_output_path
parsed_answer
quality_score
cost_input_usd
cost_output_usd
cost_total_usd
cache_hit
hardware_id
server_backend
server_config_json
```

### 5.2 Cost formulas

For frontier models:

```text
cost_total = input_tokens * input_price_per_token
           + output_tokens * output_price_per_token
```

For local models:

```text
remote_cost = 0
local_latency and GPU-time are reported separately
```

Also report optional local compute proxy:

```text
local_gpu_seconds
local_energy_proxy = gpu_seconds * assumed_wattage / 3600
```

Do not mix local GPU cost into the main monetary cost unless clearly labeled.

### 5.3 Main cost metrics

Report:

```text
remote API cost per query
remote API cost per 1K queries
remote API cost per 1M queries
normalized remote cost vs all-GPT
normalized remote cost vs all-Claude
frontier-call rate
local-call rate
probe-call rate
```

### 5.4 Latency metrics

Report:

```text
mean latency
p50 latency
p90 latency
p95 latency
p99 latency
router time
probe time
selected model generation time
frontier network/API time
end-to-end time
```

### 5.5 Utility definitions

Run all three utility settings:

#### Quality-only

```text
U_quality(q,m) = quality(q,m)
```

Use only for diagnosis.

#### Cost-aware

```text
U_cost(q,m) = quality(q,m) - lambda_cost * normalized_remote_cost(q,m)
```

Main metric.

#### Cost + latency-aware

```text
U_cost_latency(q,m) = quality(q,m)
                   - lambda_cost * normalized_remote_cost(q,m)
                   - lambda_latency * normalized_latency(q,m)
```

Use for SLA-style evaluation and sensitivity.

Do not rely only on scalar utility. Also report Pareto curves:

```text
quality vs remote cost
quality vs latency
quality vs frontier-call rate
```

---

## 6. Prompt and scoring protocol

### 6.1 General model prompt rules

For exact-scored tasks, force concise answer formats.

Example multiple choice:

```text
Answer with only one letter: A, B, C, or D.
```

Example math:

```text
Solve the problem. Put the final numeric answer in the last line as: Final answer: <answer>
```

Example code:

```text
Return only the Python code inside one code block. Do not include explanations.
```

### 6.2 Probe prompt rules

Probes must be cheap.

Default cheap probes:

1. kNN disagreement / embedding margin: no generation.
2. Qwen3.5-0.8B short confidence probe: 16–32 output tokens.
3. Code-only optional probe: Qwen3-Coder short 32-token dry-run or confidence prompt.

Probe output should not be a full answer unless needed.

Example confidence probe:

```text
You are a routing probe. Do not solve the problem fully.
Given the query, estimate whether a small local model can answer it reliably.
Return JSON only:
{"confidence_small_model_can_solve": <0-1>, "reason_type": "math|code|knowledge|ambiguous", "needs_frontier": true|false}
```

For logprob-based probes, record token logprobs if backend supports it. If logprobs are unavailable, use self-rated confidence + output format validity.

---

## 7. Experiment group A — Pilot observations

These experiments justify the method. Run these before advanced method work.

### A0. Dry-run infrastructure test

**Question:** Does the full local/frontier/caching/scoring pipeline work?

**Data:** 5 examples per benchmark.

**Methods/models:** all 5 local + 2 frontier on tiny subset.

**Outputs:**

```text
model_outputs.parquet
scored_outputs.parquet
cost_latency_summary.csv
errors.csv
```

**Expected result:**

```text
all models run or failures are logged cleanly
scoring works
cost/latency estimates are non-null
cache/resume works
```

**Stop condition:** Fix infrastructure before larger runs.

---

### A1. Cost-aware routability audit

**Question:** Is routing useful in the controlled model pool?

**Models:** 5 local + 2 frontier.

**Benchmarks:** all 8–9 benchmarks, pilot size first.

**Baselines:**

```text
all each individual model
best local model
best frontier model
best single model overall
query oracle quality-only
query oracle cost-aware
query oracle cost+latency-aware
dataset oracle
```

**Metrics:**

```text
quality
remote cost
latency
frontier-call rate
oracle gap
winner distribution
per-benchmark oracle gap
```

**Expected result:**

```text
query oracle > best single
cost-aware oracle uses local models for many queries
frontier models are not always needed
```

**Supports:** RQ1.

---

### A2. Low-dimensional cost-aware latent state frontier

**Question:** Do small latent states preserve cost-aware oracle routing?

**Methods:**

```text
best single
dataset/domain lookup
embedding cluster lookup
utility-vector clusters
regret-optimized latent route states
query oracle
```

**K values:**

```text
K = 4, 8, 16, 32
```

**Metrics:**

```text
cost-aware utility
quality
remote cost
latency
oracle regret
recovered oracle gap
state entropy H(Z)
frontier-call rate by state
```

**Expected result:**

```text
K=8 or K=16 states recover most cost-aware oracle gap.
```

**Supports:** Claim A.

---

### A3. Observability gap under strong features

**Question:** Can query text alone infer the latent states?

**Feature/predictor set:**

```text
BGE or Qwen embeddings + MLP
ModernBERT or DeBERTa classifier
kNN over embeddings
```

Use only 2–3 strong predictors. Do not run 10 encoders.

**Compare:**

```text
oracle latent states
query-predicted latent states
dataset/topic lookup
direct query-to-model router
```

**Metrics:**

```text
state prediction accuracy
utility-weighted confusion
cost-aware utility
observability gap
ECE / calibration
confidence vs regret
```

**Expected result:**

```text
strong encoders improve query-only prediction,
but a meaningful observability gap remains.
```

**Supports:** Claim B.

---

### A4. Cheap probe feasibility

**Question:** Do cheap probes improve state inference on uncertain queries?

**Probe models:**

```text
Qwen3.5-0.8B as default cheap probe
Qwen3-Coder probe only for code-like queries, optional
```

**Probe features:**

```text
embedding margin
kNN disagreement
short 16/32-token probe output
self-rated confidence
mean logprob/entropy if available
format validity
cheap model agreement, optional
```

**Metrics:**

```text
state inference improvement
cost-aware utility improvement
observability gap closed
probe cost
probe latency
probe usefulness by benchmark/domain
```

**Expected result:**

```text
probes help mostly on high-uncertainty queries.
```

**Supports:** Claim C feasibility.

---

## 8. Experiment group B — Main method evaluation

### B1. Main ProbeRoute++ routing evaluation

**Question:** Does ProbeRoute++ improve quality-cost-latency tradeoff?

**Methods:**

```text
All GPT frontier
All Claude frontier
Best local
Best single overall
Query oracle
Cost-aware oracle
Dataset/domain lookup
Embedding-cluster lookup
kNN router
Direct MLP/BERT router
Confidence cascade
ProbeRoute++ no probe
ProbeRoute++ threshold probe
ProbeRoute++ VOI probe
```

**Metrics:**

```text
quality / exact match / pass@1
cost-aware utility
remote API cost per 1K queries
normalized remote cost
frontier-call rate
local-call rate
probe rate
mean latency
p50/p95 latency
quality at fixed cost
cost at fixed quality
```

**Expected result:**

```text
ProbeRoute++ beats query-only state router and confidence cascade;
ProbeRoute++ reduces frontier usage and remote cost;
ProbeRoute++ keeps p50/p95 latency acceptable.
```

**Supports:** Claims C and E.

---

### B2. VOI probe policy evaluation

**Question:** Is VOI probing better than simple thresholds?

**Methods:**

```text
never probe
always probe
entropy threshold
margin threshold
confidence cascade
VOI ProbeRoute++
oracle probe policy upper bound
```

**Metrics:**

```text
cost-aware utility
probe rate
probe cost
probe latency
observability gap closed per probe cost
quality-cost frontier
```

**Expected result:**

```text
VOI reaches most of always-probe quality with much fewer probes.
```

Target:

```text
VOI reaches 90–95% of always-probe utility with 40–60% fewer probes.
```

---

### B3. Active new-model calibration

**Question:** Can a new model be added cheaply by calibrating per latent state?

**Protocol:** Hold out one model at a time as the new model.

Suggested held-out models:

```text
Qwen3-Coder-30B-A3B-Instruct
Qwen3.6-35B-A3B
GPT-5.5
Claude Sonnet 4.6
```

**K values:**

```text
K = 8 and K = 16
```

**Calibration budgets:**

```text
r = 4, 8, 16, 32 examples per state
```

**Methods:**

```text
random calibration
dataset-stratified calibration
embedding-cluster calibration
uniform latent-state calibration
active latent-state calibration
direct query-to-model router retraining under same total budget
```

**Metrics:**

```text
cost-aware utility vs number of new-model evaluations
quality vs calibration budget
remote cost after calibration
latency after calibration
calibration dollars spent
regret to full-data calibration
```

**Expected result:**

```text
active latent-state calibration reaches the same utility with 3–5x fewer new-model evaluations.
```

**Supports:** Claim D.

---

### B4. Cost and latency breakdown

**Question:** Where does time and money go?

**Methods:**

```text
All frontier
Best local
Confidence cascade
ProbeRoute++ threshold
ProbeRoute++ VOI
Cost-aware oracle
```

**Metrics:**

```text
router time
probe time
local generation time
frontier API time
network/API latency
remote API cost
normalized remote cost
frontier-call rate
```

**Expected result:**

```text
router/probe overhead is small relative to full generation;
ProbeRoute++ reduces remote frontier spend;
ProbeRoute++ does not create unacceptable p95 latency.
```

This is the AIMS-style system table/figure.

---

## 9. Experiment group C — Ablation studies

Keep ablations focused.

### C1. Component ablation

Compare:

```text
Full ProbeRoute++
w/o latent states: direct query-to-model router
w/o probe: query-only state router
w/o VOI: threshold probe
w/o active calibration: uniform state calibration
w/o calibration-aware state objective, if implemented
```

Metrics:

```text
quality
remote cost
latency
frontier-call rate
probe rate
calibration evaluations required
```

Expected:

```text
removing probe hurts utility;
removing VOI raises probe cost/latency;
removing active calibration increases calibration evaluations;
direct router may be competitive fixed-pool but worse for new-model transfer.
```

---

### C2. Number of latent states

Sweep:

```text
K = 4, 8, 16, 32
```

Metrics:

```text
oracle state utility
predicted state utility
ProbeRoute++ utility
calibration sample efficiency
state interpretability
```

Expected:

```text
K=8 or K=16 best tradeoff;
K=4 underfits;
K=32 harder to predict/calibrate.
```

---

### C3. Probe policy ablation

Compare:

```text
never probe
always probe
entropy threshold
margin threshold
VOI
oracle probe upper bound
```

Expected:

```text
VOI dominates threshold/random at matched probe budget.
```

---

### C4. Calibration budget ablation

Sweep:

```text
r = 4, 8, 16, 32 examples per state
```

Expected:

```text
state-level calibration improves quickly and saturates.
```

---

## 10. Experiment group D — Sensitivity analysis

### D1. Cost weight sensitivity

Sweep:

```text
lambda_cost = 0, low, medium, high
```

Report:

```text
quality-cost frontier
remote cost
frontier-call rate
selected model distribution
```

Expected:

```text
higher cost weight routes more to local models;
ProbeRoute++ remains on a better frontier than baselines.
```

---

### D2. Latency/SLA sensitivity

Sweep:

```text
latency budget = 2s, 5s, 10s, 20s
```

Report:

```text
SLA success rate
p95 latency
quality
remote cost
```

Expected:

```text
stricter SLA reduces probing/frontier use;
ProbeRoute++ adapts better than fixed cascade.
```

---

### D3. Frontier price sensitivity

Simulate:

```text
frontier price x0.5
frontier price x1
frontier price x2
frontier price x5
```

Expected:

```text
ProbeRoute++ routes to frontier less as frontier prices increase;
relative method ordering remains stable.
```

---

### D4. Local speed sensitivity

Simulate or measure:

```text
RTX 5090 normal speed
slower local speed x2
slower local speed x4
```

Expected:

```text
when local/probe latency rises, VOI should probe less;
quality-cost-latency tradeoff should degrade gracefully.
```

---

### D5. Held-out benchmark/domain sensitivity

Train/calibrate on a subset and test on held-out benchmarks.

Example:

```text
calibrate on GSM8K + MBPP + MMLU-Pro
test on MATH500 + AIME + HumanEval + LiveCodeBench + GPQA + BBH
```

Expected:

```text
OOD benchmarks increase observability gap;
state-level calibration remains more sample-efficient than direct retraining.
```

---

## 11. Baselines to include

### Required compact baseline set

```text
All each individual model
Best local
Best frontier
Best single overall
Query oracle quality-only
Cost-aware oracle
Dataset/domain lookup
Embedding-cluster lookup
kNN router
Direct MLP/BERT router
Confidence cascade
ProbeRoute++ threshold
ProbeRoute++ VOI
```

### Optional if easy

```text
RouteLLM-MF/BERT
GraphRouter / LLMRouter baseline
FrugalGPT-style cascade
Select-then-Route-style taxonomy baseline
STEER-style confidence baseline
```

Do not block the actual run on complex checkpoint-heavy baselines.

---

## 12. Expected final result package

The detailed expected results and claim thresholds are specified in:

```text
EXPECTED_RESULTS_AND_SUCCESS_CRITERIA.md
```

The short target package is:

```text
1. ProbeRoute++ is within 3 absolute quality points of the cost-aware oracle.
2. ProbeRoute++ reaches >=95--97% of cost-aware oracle utility.
3. Normalized remote API cost is 0.15x--0.35x of all-frontier.
4. Frontier-call rate is <=25%--40%; local-model usage is >=60%--75%.
5. Probe rate is <=20%--40%; VOI probing beats always/threshold probing at matched cost.
6. p95 latency is <= all-frontier p95 or no more than 1.2x all-frontier p95.
7. Router + probe overhead is <=10% average total latency.
8. Active state-level calibration adds a new model with 3x--5x fewer evaluations than direct retraining.
9. K=8 or K=16 latent states recover >=85%--90% of the cost-aware oracle gap.
10. Ablations show latent states, probes, VOI, and active calibration each matter.
```

If the results do not meet these thresholds, Codex must weaken the claims in `RUN_REPORT.md` rather than overclaiming SOTA.

---

## 13. Figure and table plan

### Main paper figures

1. **Figure 1:** ProbeRoute++ workflow: query → latent state belief → optional probe → updated belief → cost-aware model choice.
2. **Figure 2:** Cost-aware routability: best local/frontier/single vs oracle across benchmarks.
3. **Figure 3:** Latent state rate-distortion curve for `K = 4,8,16,32`.
4. **Figure 4:** Observability gap: oracle states vs query-predicted vs query+probe.
5. **Figure 5:** Quality vs normalized remote cost frontier.
6. **Figure 6:** Active new-model calibration curve.

### Main paper tables

1. **Table 1:** Model pool with roles, local/frontier, serving backend, pricing/latency notes.
2. **Table 2:** Benchmark suite and scoring metrics.
3. **Table 3:** Main method comparison: quality, cost, latency, frontier-call rate.
4. **Table 4:** Component ablation.
5. **Table 5:** Cost/latency breakdown.

### Appendix figures/tables

```text
per-benchmark detailed results
price sensitivity
latency/SLA sensitivity
K sweep details
calibration budget details
local serving logs
error analysis
```

---

## 14. Implementation outputs expected from Codex

Codex should produce these directories:

```text
configs/
  proberoute_controlled.yaml
  model_prices.yaml
  model_servers.yaml
  benchmark_sampling.yaml

results/controlled/
  raw_outputs/
  parsed_outputs/
  scored_outputs.parquet
  model_outputs.parquet
  cost_latency_summary.csv
  table_routability.csv
  table_rate_distortion.csv
  table_observability_gap.csv
  table_main_eval.csv
  table_calibration.csv
  table_ablation.csv
  table_sensitivity.csv
  fig_quality_cost_frontier.pdf
  fig_latency_breakdown.pdf
  fig_rate_distortion.pdf
  fig_observability_gap.pdf
  fig_calibration_curve.pdf
  RUN_REPORT.md
```

Every experiment must be resumable and cache model outputs.

---

## 15. Stop/go gates

### Gate 1 — after dry run

Proceed only if:

```text
all selected models can run or have documented fallback
scoring works
cost and latency logging works
```

### Gate 2 — after pilot observation

Proceed to method only if:

```text
cost-aware oracle gap exists
latent states recover meaningful oracle gap
query-only predicted states leave observability gap
```

### Gate 3 — after probe feasibility

Proceed to full ProbeRoute++ only if:

```text
cheap probes improve state inference or utility on uncertain examples
```

If probes fail, pivot to calibration-centric paper.

### Gate 4 — after calibration pilot

Make new-model calibration a main claim only if:

```text
state-level calibration beats random/dataset/direct baselines under matched budget
```

---

## 16. Final paper framing if results work

Main claim:

> ProbeRoute++ reduces frontier API usage, remote cost, and new-model calibration burden while preserving quality, by learning cost-aware latent route states and probing only uncertain queries.

More precise abstract sentence:

> We show that model routing exhibits a low-dimensional cost-aware structure, but that this structure is only partially observable from query text. ProbeRoute++ treats routing as latent-state inference under cost and latency constraints: it predicts a belief over route states, acquires cheap probe signals only when their value exceeds cost, and selects models using calibrated state-level utility estimates. Because the route states are reusable, new models can be integrated by calibrating a small number of examples per state rather than retraining a query-to-model router.

