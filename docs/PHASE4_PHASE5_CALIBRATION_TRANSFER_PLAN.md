# Phase 4 and Phase 5 RouteCode Plan

**Purpose:** define the next method and evaluation plan for RouteCode after the
Phase 3 compression ladder.

**Short answer:** Phase 4 is where RouteCode becomes the method. Phase 5 is
where we test whether those learned labels make new-model calibration and
model-pool transfer cheaper than direct router retraining.

---

## 1. Phase Relationship

### Phase 3: compression ladder

Phase 3 answers:

```text
How much routing utility can be recovered from simple query-visible structure?
```

It compares:

```text
best single
dataset-label lookup
predicted-topic lookup
embedding-cluster lookup
kNN
simple learned router
RouteCode oracle labels
query oracle
```

This phase is the evidence bridge between "an oracle exists" and "a deployable
query router works." It must keep oracle labels separate from predicted labels.

### Phase 4: RouteCode method

Phase 4 answers:

```text
Can we learn compact, utility-aware, explainable route labels that preserve
model-selection utility and are predictable enough from query text?
```

This is the core RouteCode method phase. It includes:

1. utility-vector clustering;
2. regret-optimized codebook learning;
3. predictability-constrained RouteCode;
4. query-to-label prediction;
5. code-to-model tables;
6. code cards;
7. rate-distortion curves.

Phase 4 produces the frozen label space used by Phase 5.

### Phase 5: new-model calibration and model-pool transfer

Phase 5 answers:

```text
Once a label space exists, can we add or swap models by estimating utility per
route label instead of retraining a full query-to-model router?
```

This phase tests two paper claims:

1. New models can be integrated with fewer calibration examples.
2. Route labels transfer across model pools better than direct learned routers
   under the same calibration budget.

---

## 2. Phase 4 Detailed Plan: RouteCode

### 2.1 Inputs

Phase 4 uses the canonical outcome table:

```text
query_id
query_text
dataset
domain
task_family
task_subtype
model_id
quality
cost_total
latency
tokens_input
tokens_output
judge
metadata_json
```

From this table, build matrices:

```text
Y[N, M] = quality
C[N, M] = normalized cost
L[N, M] = latency, optional
U[N, M] = Y - lambda_cost * C - lambda_latency * L
```

The default utility for the main paper should be:

```text
U(q, m) = quality(q, m) - lambda_cost * normalized_cost(q, m)
```

Latency-aware utility is a robustness setting:

```text
U(q, m) = quality(q, m)
          - lambda_cost * normalized_cost(q, m)
          - lambda_latency * normalized_latency(q, m)
```

### 2.2 Leakage Rules

Every Phase 4 artifact must be fit using train data only:

| Artifact | Fit on | Applied to |
|---|---|---|
| embeddings normalizer, if any | train queries | val/test |
| semantic clusters | train embeddings | val/test by nearest centroid |
| utility-vector codebook | train utility matrix | val/test by predictor or held-out assignment rule |
| regret-optimized codebook | train utility matrix | val/test by predictor |
| predictability-constrained codebook | train utility + train embeddings | val/test by predictor |
| code-to-model table | train utility grouped by label | val/test labels |
| K, alpha, thresholds | validation | one final test report |
| code cards | train, with test examples reported only as failures if explicitly marked | report |

No codebook, cluster, lookup table, or threshold may be selected using test
utility.

### 2.3 Methods

#### Method A: utility-vector clustering

Cluster query utility vectors:

```text
u_i = [U(q_i, m_1), ..., U(q_i, m_M)]
```

For each K:

```text
K in {1, 2, 4, 8, 16, 32, 64, 128}
```

Fit k-means or k-medoids on train utility vectors only. For each cluster label
z, compute:

```text
pi(z) = argmax_m mean_train[U(q, m) | z(q) = z]
```

This method is an oracle-style diagnostic because it clusters using model
outcomes. It measures whether routing structure is low-dimensional.

#### Method B: regret-optimized codebook

Learn labels to minimize routing regret directly:

```text
D(g, pi) = E_q [ max_m U(q, m) - U(q, pi(g(q))) ]
```

The objective should prefer label assignments that keep the same best or
near-best model together, even when their raw utility vectors are not identical.

Report:

```text
mean utility
oracle regret
recovered gap vs oracle
effective K
empirical H(Z)
label sizes
model selected per label
```

#### Method C: predictability-constrained RouteCode

This is the deployable RouteCode method.

The codebook should trade off utility preservation and query predictability:

```text
loss = regret_loss + alpha * prediction_loss + beta * label_complexity
```

Where:

```text
regret_loss: lost utility from using pi(z)
prediction_loss: how hard z is to infer from query embeddings
label_complexity: optional entropy or balance regularizer
```

The practical implementation already has the relevant entrypoint:

```text
routecode.codes.predictability_constrained.PredictabilityConstrainedRouteCode
```

The sweep should include:

```text
K = 4, 8, 16, 32, 64
alpha = 0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0
beta = 0.0, 0.1, 1.0 when label balance is unstable
lambda_cost = 0.0, 0.05, 0.1, 0.2 for cost sensitivity
```

Select K and alpha on validation. Report test once.

### 2.4 Query-to-Label Predictors

The minimum Phase 4 predictor set:

| Predictor | Input | Purpose |
|---|---|---|
| logistic regression | embeddings | cheap linear baseline |
| MLP | embeddings | stronger nonlinear cheap baseline |
| kNN label predictor | embeddings | local geometry baseline |
| gradient boosting | hashed/text features or embeddings | tabular-feature baseline |
| ModernBERT or DeBERTa classifier | query text | stronger text encoder baseline |

Do not start with LoRA. LoRA is only a later robustness baseline if cheap
predictors fail and the paper needs a stronger deployable predictor.

### 2.5 Phase 4 Benchmarks

Use precomputed routing benchmarks first:

| Priority | Benchmark | Purpose |
|---|---|---|
| primary | LLMRouterBench | largest target benchmark for main compression and RouteCode curves |
| secondary | RouterBench | external validation and comparison with routing literature |
| secondary | RouteLLM data/eval | alignment with a canonical router baseline |
| optional | LLMRouter library examples | implementation baseline coverage |

Use exact-scored controlled benchmarks for live/local validation:

| Domain | Benchmark | Metric |
|---|---|---|
| easy math | GSM8K | exact final answer |
| hard math | MATH500 | exact final answer |
| competition math | AIME | exact final answer |
| code | HumanEval | pass@1 |
| code | MBPP | pass@1 |
| live code | LiveCodeBench subset | pass@1 |
| science | GPQA | multiple choice accuracy |
| broad knowledge | MMLU-Pro | multiple choice accuracy |
| logic, optional | BBH subset | exact or multiple choice |

Avoid LLM-judge open-ended tasks until the exact-scored pipeline is stable.

### 2.6 Phase 4 Outputs

Expected outputs:

```text
table_rate_distortion.csv
table_predictor_comparison.csv
table_codebook_sweep.csv
table_label_predictability.csv
table_code_card_summary.csv
fig_rate_distortion_regret.pdf
fig_rate_distortion_recovered_gap.pdf
fig_utility_weighted_confusion.pdf
fig_code_label_heatmap.pdf
code_cards.md
code_cards.json
phase4_routecode_method_memo.md
```

### 2.7 Phase 4 Success Criteria

A strong Phase 4 result:

```text
K = 8 or K = 16 oracle route labels recover >= 80% of oracle improvement over
best single.
```

A strong deployable result:

```text
predicted RouteCode labels recover >= 85% of the best learned-router gain,
lower bootstrap CI >= 80%, and >= 50% to 60% of oracle gain.
```

If predicted labels recover only 50% to 80%, the paper should be framed as an
information-frontier and observability-gap paper, not as "routers barely need
query information."

---

## 3. Phase 5 Detailed Plan: New-Model Calibration

### 3.1 Core Hypothesis

RouteCode should integrate a new model with fewer evaluations because it
estimates the new model's utility at the route-label level:

```text
new model utility per label
```

instead of learning a full mapping:

```text
query text -> selected model
```

The expected sample-efficiency advantage comes from replacing N query-level
decisions with K label-level estimates.

### 3.2 Main Protocol

For each benchmark, model pool, seed, K, and lambda_cost:

1. Split queries by query_id into train, validation, and test.
2. Select a base model pool M_base.
3. Select one held-out model m_new.
4. Remove m_new from all Phase 4 codebook training.
5. Fit the RouteCode label space on train queries using only M_base outcomes.
6. Fit the query-to-label predictor on train queries.
7. Choose K and alpha on validation using only M_base.
8. Freeze the label predictor and label space.
9. Sample r calibration queries per label from train or a dedicated calibration split.
10. Evaluate or reveal m_new outcomes only for those calibration queries.
11. Estimate mean utility for m_new in each label.
12. Update the label-to-model table:

```text
pi_new(z) = argmax_m mean_calibrated[U(q, m) | z(q) = z]
```

13. Evaluate on test queries using the full model pool M_base union {m_new}.
14. Compare against same-budget direct router retraining.

### 3.3 Calibration Budgets

Use:

```text
r in {1, 2, 4, 8, 16, 32, 64}
```

where r is examples per label.

Total new-model evaluations:

```text
budget = K * r
```

Example budgets:

| K | r | max new-model evaluations |
|---:|---:|---:|
| 8 | 1 | 8 |
| 8 | 8 | 64 |
| 16 | 8 | 128 |
| 16 | 16 | 256 |
| 32 | 8 | 256 |
| 32 | 16 | 512 |
| 64 | 16 | 1024 |

If a label has fewer than r available train queries, use all train queries in
that label and report the actual evaluation count.

### 3.4 Calibration Sampling Strategies

Run these strategies:

| Strategy | Description | Why it matters |
|---|---|---|
| uniform per label | sample r queries from every label | main RouteCode claim |
| label-size proportional | sample proportional to train label frequency | fair average-case estimator |
| utility-margin weighted | sample more from labels where old best and second best are close | targets labels likely to switch |
| uncertainty weighted | sample labels with high query-to-label entropy | targets observability risk |
| dataset-stratified per label | preserve benchmark/domain mix inside each label | guards against label-domain leakage |
| random same-budget | random K*r queries without labels | baseline for label-aware calibration |

The first paper can use uniform per label as the main method and report the
others as ablations.

### 3.5 Estimator for New-Model Utility

For each label z, estimate:

```text
mu_new(z) = mean U(q, m_new) over calibration queries assigned to z
```

Use shrinkage when r is small:

```text
mu_hat_new(z) =
    w_z * mean_label_calibration(z)
    + (1 - w_z) * global_mean_calibration
```

with:

```text
w_z = n_z / (n_z + tau)
tau in {2, 4, 8}
```

Select tau on validation. Report the no-shrinkage estimator as an ablation.

For cost-aware runs, estimate quality and cost separately when live calls are
available:

```text
quality_hat_new(z)
cost_hat_new(z)
latency_hat_new(z)
utility_hat_new(z) =
    quality_hat_new(z)
    - lambda_cost * cost_hat_new(z)
    - lambda_latency * latency_hat_new(z)
```

### 3.6 Held-Out New Models

#### Precomputed LLMRouterBench setting

Use each reliable model as m_new, one at a time, when compute allows. The
current broad config already lists these holdout candidates:

```text
DeepHermes-3-Llama-3-8B-Preview
DeepSeek-R1-0528-Qwen3-8B
DeepSeek-R1-Distill-Qwen-7B
Fin-R1
GLM-Z1-9B-0414
Intern-S1-mini
Llama-3.1-8B-Instruct
Llama-3.1-8B-UltraMedical
Llama-3.1-Nemotron-Nano-8B-v1
MiMo-7B-RL-0530
MiniCPM4.1-8B
NVIDIA-Nemotron-Nano-9B-v2
OpenThinker3-7B
Qwen2.5-Coder-7B-Instruct
Qwen3-8B
cogito-v1-preview-llama-8B
gemma-2-9b-it
glm-4-9b-chat
granite-3.3-8b-instruct
internlm3-8b-instruct
```

Group results by role:

```text
general local
reasoning local
code local
medical or finance specialist
small/cheap model
```

#### Controlled local vLLM setting

Use sequential vLLM serving. Do not load all local models at once.

Target local pool:

| Role | Model id |
|---|---|
| cheap probe | `Qwen/Qwen3-0.6B` or `Qwen/Qwen3.5-0.8B` |
| general local | `Qwen/Qwen3-4B` |
| strong general local | `Qwen/Qwen3.5-9B` |
| code specialist | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| strong local reasoning | `Qwen/Qwen3.6-35B-A3B` |
| diverse non-Qwen local | `google/gemma-3-12b-it` or `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |

Recommended local holdouts:

```text
qwen3-4b-local
qwen3.5-9b-local
qwen3-coder-30b-a3b
qwen3.6-35b-a3b
gemma-3-12b-it
```

Do not treat the tiny probe model as the only new-model result. It is useful for
pipeline validation, but the paper needs at least one strong general model, one
code specialist, and one diverse non-Qwen local model.

#### Closed-source provider setting

Closed-source provider models are important for the applied routing story, but
they must be explicit-budget experiments with cached outputs and fresh pricing
documentation.

Provider families to include in the plan:

| Provider family | Planned role | Notes |
|---|---|---|
| OpenAI GPT-family | frontier general or frontier reasoning | example configured id: `gpt-5.5` |
| Anthropic Claude-family | frontier reasoning/writing/coding | add before final provider comparison |
| Google Gemini-family | efficient frontier or broad frontier | example configured id: `gemini-3.5-flash` |

Before any provider run:

1. require explicit API enablement;
2. refresh official pricing source URLs and checked date;
3. set max spend in config;
4. log request count, input tokens, output tokens, latency, and model version;
5. cache raw outputs and scoring records;
6. run calibration budgets first, not full matrices.

### 3.7 New-Model Calibration Baselines

Every Phase 5 calibration result should compare against:

| Baseline | Description |
|---|---|
| no-new-model RouteCode | frozen old label-to-model table without m_new |
| always-new-model | route every query to m_new |
| best single with new model | best one model on train including m_new calibration data |
| full query oracle with new model | diagnostic upper bound using all test utilities |
| random same-budget calibration | reveal K*r random m_new outcomes, not per-label balanced |
| dataset-label calibration | estimate m_new utility per dataset label |
| embedding-cluster calibration | estimate m_new utility per embedding cluster |
| kNN calibration | nearest calibrated examples choose whether to use m_new |
| direct logistic router | train q -> model labels under same K*r budget |
| direct SVM router | same budget as RouteCode calibration |
| direct kNN router | same budget as RouteCode calibration |
| direct MLP router | stronger same-budget direct baseline when stable |
| full-data direct router | upper bound when all m_new train outcomes are available |

For external positioning, compare with RouteLLM, RouterBench/FrugalGPT-style
cascades, and LLMRouter baselines where adapters are available. Do not block the
first Phase 5 run on complex external baselines.

### 3.8 New-Model Calibration Metrics

Report all metrics over test queries:

| Metric | Definition |
|---|---|
| mean quality | average task accuracy/pass@1/exact score of selected model |
| mean utility | average selected U(q, m) |
| normalized cost | selected model cost normalized to all-frontier or max-cost baseline |
| p50/p95 latency | selected model latency, with router/probe latency separated |
| oracle regret | mean max_m U(q, m) - U(q, selected_model) |
| recovered gap vs oracle | (method - best_single) / (oracle - best_single) |
| recovered gap vs learned | (method - best_single) / (learned_router - best_single) |
| new-model usage rate | share of test queries routed to m_new |
| labels switching to new model | number and fraction of labels whose pi(z) becomes m_new |
| calibration evaluations | actual number of revealed/evaluated m_new outcomes |
| calibration spend | API spend or local GPU proxy cost for calibration |
| examples to target | evaluations needed to reach 90%, 95%, 97% of full-data reference |
| bootstrap CI | confidence interval over query-level utility |

Cost must be part of the main utility result, not only a secondary table.

### 3.9 New-Model Calibration Acceptance Targets

Strong claim target:

```text
RouteCode label calibration reaches the same utility as direct router
retraining with 3x to 5x fewer new-model evaluations.
```

Applied target:

```text
RouteCode with calibrated new model is within 3 absolute quality points of the
cost-aware oracle, or reaches >= 95% to 97% of oracle utility, while using much
lower remote frontier cost than all-frontier routing.
```

Minimum diagnostic target:

```text
RouteCode label calibration beats random same-budget calibration and
dataset-label calibration on most held-out models.
```

If RouteCode only wins for some model roles, report the conditional finding:

```text
Label-level calibration helps when the new model has label-specific strengths,
but not when the new model is uniformly better or uniformly worse.
```

### 3.10 Existing Entrypoint

The current implementation already has a calibration script:

```bash
PYTHONPATH=src python experiments/07_new_model_calibration.py \
  --config configs/llmrouterbench_broad20.yaml
```

Expected outputs:

```text
results/<run_name>/table_new_model_integration.csv
results/<run_name>/fig_transfer_calibration_curve.pdf
results/<run_name>/phase_e5_new_model_calibration_memo.md
```

For a fast smoke run, use a pilot config and only a small holdout set. For the
paper run, use the broad LLMRouterBench config or the full accessible benchmark
pool.

---

## 4. Phase 5 Detailed Plan: Model-Pool Transfer

### 4.1 Core Hypothesis

Route labels should transfer better than direct q-to-model routers because the
label describes model-selection behavior at a higher level than a model id.

The transfer pattern is:

```text
source pool A:
    learn query -> route label
    learn route label -> source model

target pool B:
    keep query -> route label
    recalibrate route label -> target model
```

Direct routers have to relearn:

```text
query -> target model
```

under the same limited target-pool calibration budget.

### 4.2 Transfer Scenarios

Run at least these scenarios:

| Scenario | Source pool | Target pool | Question |
|---|---|---|---|
| add-one-model | M_base | M_base + m_new | does label calibration add a model cheaply? |
| local-to-local | 3 to 4 local models | different 3 to 4 local models | do labels survive local pool changes? |
| local-to-frontier | local-only | local + GPT/Gemini/Claude | can labels identify when frontier is worth cost? |
| remove-dominated | full pool | pool with strongest model removed | does the label predictor remain useful? |
| specialist-addition | general local pool | add code/math/specialist model | do labels capture specialist wins? |
| family-shift | Qwen-heavy pool | non-Qwen or mixed pool | does RouteCode avoid overfitting to a model family? |
| cost-regime shift | same models, low lambda | same models, high lambda | do labels transfer across cost preferences? |
| benchmark-pool shift | broad10 | broad20 or 32-model pool | do labels survive larger candidate pools? |

### 4.3 Source and Target Pool Definitions

#### Offline LLMRouterBench pools

Use these from existing configs:

```text
configs/llmrouterbench_broad10.yaml
configs/llmrouterbench_broad20.yaml
configs/llmrouterbench_32model.yaml
configs/llmrouterbench_scale20.yaml
```

Recommended pool sweeps:

```text
source_size in {4, 8, 12}
target_size in {4, 8}
K = 16
direct_router_methods = logistic, svm, knn, mlp, gradient_boosting
```

The current broad20 config already contains:

```text
model_pool_transfer:
  source_sizes: [4, 8, 12]
  target_sizes: [4, 8]
  k: 16
  direct_router_methods: [logistic, svm, knn, mlp, gradient_boosting]
```

#### Controlled live/local pools

Use roles, not only names:

```text
source local pool:
  cheap probe
  general local
  code specialist
  diverse non-Qwen local

target local + frontier pool:
  cheap probe
  general local
  code specialist
  diverse non-Qwen local
  GPT-family frontier
  Claude-family frontier
  Gemini-family frontier
```

The provider target pool should be used only after API budget, caching, and
pricing are configured.

### 4.4 Model-Pool Transfer Protocol

For each source-target scenario:

1. Split queries by query_id.
2. Fit source RouteCode labels using train queries and source-pool utility.
3. Fit source query-to-label predictor.
4. Freeze the source query-to-label predictor.
5. On target train or target calibration split, estimate target-pool utility per label.
6. Build target label-to-model table:

```text
pi_target(z) = argmax_m mean_target_train[U(q, m) | source_label(q) = z]
```

7. Predict labels for target test queries using only query-visible features.
8. Select target models using pi_target.
9. Compare against native target RouteCode trained on target pool.
10. Compare against direct target routers under the same calibration budget.

### 4.5 Model-Pool Transfer Baselines

Report:

| Method | Role |
|---|---|
| target best single | minimum useful target baseline |
| target query oracle | target upper bound |
| source router without recalibration | negative-control transfer |
| source RouteCode label transfer | main method |
| target native RouteCode | upper bound for label method when target full train is available |
| target kNN | nonparametric target router |
| target logistic/SVM/MLP | direct q-to-model target baselines |
| dataset-label target lookup | diagnostic coarse-label baseline |
| embedding-cluster target lookup | semantic cluster baseline |
| RouteLLM/RouterBench/LLMRouter baselines | external comparison when adapters are ready |

### 4.6 Model-Pool Transfer Metrics

| Metric | Definition |
|---|---|
| target mean utility | utility on target test pool |
| transfer utility retention | transferred RouteCode utility / native target RouteCode utility |
| transfer recovered gap | recovered gap vs target oracle |
| target oracle regret | regret under target pool |
| negative transfer rate | share of scenarios where transfer is worse than target best single |
| calibration evaluations | target-pool examples used to fit pi_target |
| source-target overlap | number of shared models between pools |
| label coverage | fraction of target test labels observed in target calibration |
| label remap entropy | entropy of target selected model distribution over labels |
| model-rank correlation | rank correlation of label-level model utilities across source and target |
| code-card drift | change in best model, second-best model, and dominant datasets per label |
| bootstrap CI | CI over query-level utility |

### 4.7 Model-Pool Transfer Acceptance Targets

Strong transfer target:

```text
source RouteCode label transfer reaches >= 90% of native target RouteCode
utility with <= 25% of the target calibration examples needed by direct routers.
```

Sample-efficiency target:

```text
source RouteCode label transfer matches direct target router utility with 3x to
5x fewer target-pool calibration examples.
```

Robustness target:

```text
negative transfer rate <= 10% across source-target scenarios, and failures are
detectable by validation utility or high label uncertainty.
```

If transfer fails under family shift, report:

```text
Route labels transfer within related model pools but need recalibration or
relearning under large model-family shifts.
```

### 4.8 Existing Entrypoint

The current implementation already has a model-pool transfer script:

```bash
PYTHONPATH=src python experiments/19_model_pool_transfer.py \
  --config configs/llmrouterbench_broad20.yaml
```

Expected outputs:

```text
results/<run_name>/table_model_pool_transfer.csv
results/<run_name>/phase_f_g_model_pool_transfer_memo.md
```

---

## 5. Benchmark Plan

### 5.1 Benchmark Order

Use this order:

1. Synthetic only for pipeline sanity.
2. LLMRouterBench pilot.
3. LLMRouterBench broad20.
4. LLMRouterBench 32-model if complete and reliable.
5. RouterBench secondary validation.
6. RouteLLM data/eval alignment.
7. Controlled exact-scored local/frontier validation.

Do not use provider API calls as the first proof. Provider API calls should
validate the applied story after offline evidence is already positive.

### 5.2 Main Benchmark Fields to Report

For each benchmark family:

```text
benchmark name
query count
model count
model list
quality metric
cost source
latency source, if available
train/val/test split seed
lambda_cost values
K values
baseline list
external repo/source used
```

### 5.3 Required Split Variants

Run:

| Split | Purpose |
|---|---|
| random query split | standard in-distribution estimate |
| leave-one-dataset-out | checks benchmark shortcut dependence |
| leave-one-domain-out | checks domain transfer |
| domain-homogeneous | checks fine-grained routing without coarse domain labels |
| model holdout | new-model calibration |
| model-pool holdout | transfer across candidate pools |
| cluster-held-out | checks embedding-neighborhood generalization |

The main Phase 5 claims should not rely only on random mixed splits.

---

## 6. Model Plan

### 6.1 Offline Benchmark Models

Use all reliable precomputed models from the selected benchmark when:

```text
all selected models have complete outcomes for the selected query set
model ids are stable
quality metrics are comparable
cost metadata can be assigned or normalized
```

For LLMRouterBench broad20, use the configured 20-model pool as the main
offline sample-efficiency test.

### 6.2 Local vLLM Models

Primary serving backend:

```text
vLLM OpenAI-compatible server
```

Local models should be run sequentially and cached:

```text
start vLLM for one model
run benchmark slice
save raw outputs, scores, tokens, latency
stop server
repeat for next model
```

Minimum useful local paper pool:

```text
cheap probe model
strong general local
code specialist
strong reasoning/general local
diverse non-Qwen local
```

### 6.3 Closed-Source Provider Models

Provider families to keep in the final applied model-pool plan:

```text
OpenAI GPT-family
Anthropic Claude-family
Google Gemini-family
```

Use them for:

1. calibration-only validation;
2. local-plus-frontier transfer;
3. cost-quality frontier plots.

Do not use them for:

1. full brute-force matrix generation before offline results are promising;
2. any run without explicit budget and token logging;
3. any claim that lacks refreshed provider pricing documentation.

---

## 7. Main Result Tables and Figures

### 7.1 New-Model Calibration

Tables:

```text
table_new_model_integration.csv
table_new_model_calibration_by_model.csv
table_new_model_calibration_by_domain.csv
table_calibration_budget_efficiency.csv
table_calibration_ablation.csv
```

Figures:

```text
fig_transfer_calibration_curve.pdf
fig_calibration_examples_to_target.pdf
fig_new_model_usage_by_label.pdf
fig_calibration_domain_breakdown.pdf
```

### 7.2 Model-Pool Transfer

Tables:

```text
table_model_pool_transfer.csv
table_transfer_by_pool_pair.csv
table_transfer_negative_cases.csv
table_label_remap_drift.csv
```

Figures:

```text
fig_model_pool_transfer_utility.pdf
fig_transfer_vs_direct_budget.pdf
fig_label_remap_heatmap.pdf
fig_source_target_rank_correlation.pdf
```

### 7.3 Cost-Quality Reporting

Tables:

```text
table_cost_quality_frontier.csv
table_provider_cost_summary.csv
table_latency_summary.csv
```

Figures:

```text
fig_cost_quality_frontier.pdf
fig_frontier_call_rate_by_method.pdf
fig_latency_breakdown.pdf
```

---

## 8. Claim Gates

### Gate 1: Phase 4 labels are useful

Required before Phase 5 can be a main paper claim:

```text
RouteCode labels at K = 8, 16, or 32 recover a large fraction of oracle gap on
validation, and predicted labels beat semantic clusters or dataset lookup on
test under leakage-safe splits.
```

### Gate 2: new-model calibration beats same-budget direct retraining

Required for the new-model claim:

```text
RouteCode label calibration reaches direct-router utility with 3x to 5x fewer
new-model evaluations, averaged across held-out model roles.
```

### Gate 3: transfer is not just lucky model overlap

Required for the model-pool transfer claim:

```text
source-target overlap is reported, and RouteCode transfer remains competitive
when source and target pools have low or zero overlap.
```

### Gate 4: provider cost is real

Required for any GPT/Claude/Gemini applied claim:

```text
provider outputs are cached, model versions are logged, token counts are saved,
pricing source URLs and checked dates are recorded, and cost-aware utility uses
those costs.
```

### Gate 5: no overclaiming

If the outcome is:

```text
oracle labels strong, predicted labels weak
```

then the conclusion is:

```text
model-choice structure is compressible but only partially observable from query
text.
```

If the outcome is:

```text
new-model calibration helps only some held-out models
```

then the conclusion is:

```text
RouteCode calibration is useful when new models have label-specific strengths,
not as a universal guarantee.
```

---

## 9. Recommended Execution Order

1. Finish Phase 3 compression ladder on the chosen benchmark matrix.
2. Run Phase 4 codebook and predictability-constrained sweeps.
3. Select K and alpha on validation.
4. Generate Phase 4 code cards.
5. Run held-out model calibration on LLMRouterBench pilot.
6. Run held-out model calibration on LLMRouterBench broad20.
7. Run model-pool transfer on broad10 to broad20 and broad20 internal pool splits.
8. Add RouterBench or RouteLLM secondary validation.
9. Run local vLLM controlled model pool.
10. Only after positive offline evidence, run provider calibration for GPT,
    Claude, and Gemini families with explicit budget and cached outputs.

Recommended commands for the current repo:

```bash
PYTHONPATH=src python experiments/01_compression_ladder.py \
  --config configs/llmrouterbench_broad20.yaml

PYTHONPATH=src python experiments/02_rate_distortion_curve.py \
  --config configs/llmrouterbench_broad20.yaml

PYTHONPATH=src python experiments/06_predictability_constrained.py \
  --config configs/llmrouterbench_broad20.yaml

PYTHONPATH=src python experiments/07_new_model_calibration.py \
  --config configs/llmrouterbench_broad20.yaml

PYTHONPATH=src python experiments/19_model_pool_transfer.py \
  --config configs/llmrouterbench_broad20.yaml
```

---

## 10. Paper Story If Phase 5 Works

The strong story:

```text
RouteCode learns compact, interpretable routing states. These states recover
most cost-aware routing utility and let new candidate models be calibrated with
far fewer examples than direct router retraining.
```

The evidence chain:

1. Phase 3: routing has a measurable information frontier.
2. Phase 4: RouteCode labels are a good point on that frontier.
3. Phase 5: the label space is reusable for adding and swapping models.

The paper value is not token savings. The value is:

```text
information structure
sample efficiency
model-pool transfer
benchmark diagnosis
explainability
cost-aware deployment planning
```

---

## 11. Paper Story If Phase 5 Is Mixed

A mixed result is still publishable if the analysis is clean.

Possible conclusion:

```text
LLM routing benchmarks contain low-rate oracle structure, but deployable
transfer depends on whether model strengths align with query-visible route
states. RouteCode exposes when calibration by label is reliable and when full
router retraining is necessary.
```

This becomes an information-frontier and diagnostic paper rather than a
near-oracle routing-method paper.
