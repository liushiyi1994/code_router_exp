# RouteCode Research Findings

This document consolidates the current RouteCode pilot results, test evidence,
benchmark setup, model pools, observations, and claim posture.

It is a research status document, not a final paper draft. The most important
conclusion is conservative:

> RouteCode found strong low-rate oracle routing structure, but the current
> deployable query-to-route-label methods do not recover the oracle.

In plain terms: the route labels can be very useful if the correct label is
known, but predicting that label from query features is the current bottleneck.

## 1. Research Question

RouteCode asks:

> How many bits of query information does an LLM router need to choose a good
> model?

A normal router maps:

```text
query -> selected model
```

RouteCode maps:

```text
query -> route label -> selected model
```

The route label is not a normal topic label. It is learned from query-model
utility patterns, then later explained with code cards.

## 2. What Was Tested

The project followed the intended research flow:

1. Synthetic sanity check.
2. Real-data pilot on LLMRouterBench.
3. Observation synthesis.
4. Flat RouteCode and regret-optimized RouteCode.
5. Predictability-constrained RouteCode, called D2 in the repo.
6. Code cards.
7. New-model calibration.
8. External baseline coverage.
9. Ablation and sensitivity.
10. Claim audit and completion audit.

The main test was not whether a router can overfit query IDs. The main test was
whether low-rate labels learned from train utility structure can preserve model
selection utility on held-out queries without leakage.

## 3. Benchmark Data

### Synthetic Demo

Synthetic data is used only to validate the pipeline.

Evidence file: `results/demo/outcomes.csv`

Scope:

```text
rows: 14,400
queries: 2,400
datasets: synthetic_code, synthetic_data, synthetic_debug, synthetic_easy,
          synthetic_instructions, synthetic_math
domains: data_transformation, general_knowledge, instruction_following,
         routine_code, symbolic_math, systems_debugging
models: code_7b, frontier_expensive, general_8b, math_7b, reasoner_13b,
        tiny_cheap
```

Synthetic results are not used for scientific claims.

### Main Real-Data Pilot: LLMRouterBench Pilot

Evidence file: `results/llmrouterbench_pilot/outcomes.csv`

Scope:

```text
rows: 17,382
queries: 2,897
datasets: aime, gpqa, humaneval, math500, mbpp, mmlupro
domains: broad_knowledge, code, math, science
models: 6
```

The six pilot models were:

```text
Qwen3-8B
Qwen2.5-Coder-7B-Instruct
DeepSeek-R1-Distill-Qwen-7B
Llama-3.1-8B-Instruct
MiniCPM4.1-8B
Intern-S1-mini
```

Config: `configs/llmrouterbench_pilot.yaml`

Important settings:

```text
source: LLMRouterBench released outcome records
external API calls: none
drop incomplete queries: true
train/val/test split: 60/20/20 by query_id
hashing features: 256
utility lambda_cost: 0.0
embedding clusters: 16
kNN k: 15
RouteCode K sweep: 1, 2, 4, 8, 16, 32, 64, 128
D2 K: 16
D2 alpha sweep: 0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0
bootstrap samples: 300
confidence interval: 95%
```

The pilot uses quality-only utility because `lambda_cost = 0.0`:

```text
U(q, m) = quality(q, m) - lambda * cost(q, m)
        = quality(q, m)
```

### Robustness and Broader Runs

Additional LLMRouterBench runs were generated for robustness and claim audit.

#### Broad20

Evidence file: `results/llmrouterbench_broad20/outcomes.csv`

```text
rows: 280,820
queries: 14,041
datasets: 18
models: 20
domains: broad_knowledge, code, commonsense_reasoning, dialogue, finance,
         logical_reasoning, math, medicine, multilingual, reasoning, science
```

The Broad20 models were:

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

#### Scale20

Evidence file: `results/llmrouterbench_scale20/outcomes.csv`

```text
rows: 57,940
queries: 2,897
datasets: aime, gpqa, humaneval, math500, mbpp, mmlupro
models: same 20-model pool as Broad20
```

#### 32-Model Run

Evidence file: `results/llmrouterbench_32model/outcomes.csv`

```text
rows: 77,920
queries: 2,435
datasets: aime, gpqa, livecodebench, livemathbench, mmlupro
models: 32
```

This run includes the local/open models above plus larger provider-style model
IDs such as:

```text
claude-sonnet-4
deepseek-r1-0528
deepseek-v3-0324
gemini-2.5-flash
gemini-2.5-pro
glm-4.6
gpt-5
gpt-5-chat
intern-s1
kimi-k2-0905
qwen3-235b-a22b-2507
qwen3-235b-a22b-thinking-2507
```

These are outcome records from benchmark artifacts. RouteCode did not call
external model APIs to generate them.

## 4. What The Oracles Mean

There are multiple oracles. They are not the same thing.

### Query Oracle

Code: `src/routecode/routers/oracle.py`

Definition:

```python
utility.idxmax(axis=1)
```

For each test query, the query oracle directly chooses the model with maximum
test utility:

```text
selected_model(q) = argmax_m U_test(q, m)
```

This is not deployable. It uses the answer we are trying to predict. It is the
upper bound.

### Flat RouteCode Oracle Labels

Code: `src/routecode/codes/routecode.py`

This learns a utility-vector codebook on train queries. For diagnostic oracle
evaluation, test labels can be assigned using test utility vectors and train
utility centroids.

This answers:

> If the right utility label were known, how much routing performance could
> low-rate labels preserve?

It is not deployable because the test assignment uses test utility information.

### Regret-Optimized RouteCode Oracle Labels

Code: `src/routecode/codes/regret.py`

This also learns labels from train utilities, but it refines labels for routing
regret, meaning labels are optimized for the selected model decision rather
than only utility-vector similarity.

This is also not deployable when evaluated with utility-based test assignment.
It is a diagnostic upper bound on label structure.

## 5. How The Tests Were Conducted

### Experimental Tests

The main experimental scripts are:

```text
experiments/00_data_audit.py
experiments/01_compression_ladder.py
experiments/02_rate_distortion_curve.py
experiments/03_residual_concentration.py
experiments/04_split_sensitivity.py
experiments/06_predictability_constrained.py
experiments/07_new_model_calibration.py
experiments/08_ablation_summary.py
experiments/09_sensitivity_suite.py
experiments/21_external_command_readiness.py
experiments/41_paper_evidence_summary.py
experiments/42_research_flow_completion_audit.py
experiments/43_external_blocker_resolution.py
experiments/44_phase_e_baseline_coverage.py
```

Main pilot command pattern:

```bash
python experiments/<script>.py --config configs/llmrouterbench_pilot.yaml
```

The later audit scripts use result directories and readiness tables, for
example:

```bash
python experiments/42_research_flow_completion_audit.py \
  --root /home/liush/projects/code_router_exp \
  --output-dir /home/liush/projects/code_router_exp/results
```

### Automated Test Suite

The automated test suite checks:

```text
data loading and schema
query_id split behavior
utility computation
metrics and bootstrap confidence intervals
oracle router
best-single, dataset-label, embedding-cluster, kNN routers
flat RouteCode assignment
regret RouteCode assignment
predictability-constrained RouteCode
code cards and interpretability outputs
new-model calibration
external baseline adapters and readiness checks
GraphRouter, RouteLLM, LLMRouter, FrugalGPT, EmbedLLM, Avengers-Pro adapters
claim audit scripts
Research Flow completion audit
Phase E baseline coverage audit
```

Fresh verification command:

```bash
pytest -q
```

Exact result:

```text
230 passed, 20 warnings in 30.92s
```

Warnings were deprecation or upstream-library warnings from FrugalGPT, Pydantic,
Torch JIT, and SWIG-related imports. No tests failed.

## 6. Main Pilot Results

Evidence files:

```text
results/llmrouterbench_pilot/table_routability.csv
results/llmrouterbench_pilot/table_recovered_gap.csv
results/llmrouterbench_pilot/table_rate_distortion.csv
results/llmrouterbench_pilot/table_predictability_constrained.csv
```

### Routability

| method | mean utility | oracle regret | recovered gap vs oracle |
|---|---:|---:|---:|
| cheapest | 0.6138 | 0.2828 | -0.2331 |
| best_single | 0.6672 | 0.2293 | 0.0000 |
| dataset_label_lookup | 0.7534 | 0.1431 | 0.3759 |
| query_oracle | 0.8966 | 0.0000 | 1.0000 |

Observation:

The pilot model pool is routable. There is a large gap between best single and
query oracle. But the oracle winner distribution is partly dominated: the
dominant oracle winner takes about 77.1% of test queries, and oracle model-win
entropy is 1.2265 bits.

### Compression Ladder

| method | K | mean utility | oracle regret | recovered gap vs oracle |
|---|---:|---:|---:|---:|
| random | 6 | 0.5724 | 0.3241 | -0.4135 |
| best_single |  | 0.6672 | 0.2293 | 0.0000 |
| dataset_label_lookup | 6 | 0.7534 | 0.1431 | 0.3759 |
| dataset_oracle | 6 | 0.7638 | 0.1328 | 0.4211 |
| predicted_topic_lookup | 6 | 0.7448 | 0.1517 | 0.3383 |
| embedding_cluster_lookup | 16 | 0.7362 | 0.1603 | 0.3008 |
| kNN |  | 0.7362 | 0.1603 | 0.3008 |
| logistic_embedding_router |  | 0.6741 | 0.2224 | 0.0301 |
| mlp_embedding_router |  | 0.6776 | 0.2190 | 0.0451 |
| svm_embedding_router |  | 0.6724 | 0.2241 | 0.0226 |
| routecode_oracle_labels | 16 | 0.8897 | 0.0069 | 0.9699 |
| routecode_predicted_labels | 16 | 0.6138 | 0.2828 | -0.2331 |
| routecode_mlp_predicted_labels | 16 | 0.6345 | 0.2621 | -0.1429 |
| query_oracle | 6 | 0.8966 | 0.0000 | 1.0000 |

Observation:

Flat utility RouteCode has a very strong oracle, but the applied predicted
label version fails badly. It performs below best single.

### Rate-Distortion Curve

Selected rows from `table_rate_distortion.csv`:

| method | K | rate log2(K) | empirical H(Z) | mean utility | recovered gap |
|---|---:|---:|---:|---:|---:|
| regret_routecode_oracle_labels | 2 | 1 | 0.8592 | 0.8155 | 0.6466 |
| regret_routecode_oracle_labels | 4 | 2 | 1.7460 | 0.8621 | 0.8496 |
| regret_routecode_oracle_labels | 8 | 3 | 2.3278 | 0.8966 | 1.0000 |
| routecode_oracle_labels | 16 | 4 | 3.6971 | 0.8897 | 0.9699 |
| routecode_oracle_labels | 32 | 5 | 4.4593 | 0.8966 | 1.0000 |
| routecode_predicted_labels | 16 | 4 | 3.8289 | 0.6138 | -0.2331 |
| regret_routecode_predicted_labels | 16 | 4 | 2.3420 | 0.6914 | 0.1053 |

Observation:

The hindsight rate-distortion curve is excellent. A regret-optimized oracle
codebook reaches the query oracle at K=8. But predicted labels remain far from
that frontier.

## 7. Predictability-Constrained RouteCode Results

Evidence file:

```text
results/llmrouterbench_pilot/table_predictability_constrained.csv
```

D2 was introduced because flat utility labels were strong but hard to infer
from query features.

Key rows:

| method | alpha | mean utility | recovered gap | label accuracy |
|---|---:|---:|---:|---:|
| best_single |  | 0.6672 | 0.0000 |  |
| kNN |  | 0.7362 | 0.3008 |  |
| dataset_label_lookup |  | 0.7534 | 0.3759 |  |
| flat_routecode_utility_oracle |  | 0.8897 | 0.9699 |  |
| flat_routecode_logistic_label_predictor |  | 0.6138 | -0.2331 |  |
| d2_joint_oracle_labels | 0.00 | 0.8759 | 0.9098 | 1.0000 |
| d2_embedding_centroid | 0.00 | 0.6948 | 0.1203 | 0.1500 |
| d2_logistic_label_predictor | 0.00 | 0.6914 | 0.1053 | 0.1466 |
| d2_joint_oracle_labels | 3.00 | 0.7466 | 0.3459 | 1.0000 |
| d2_embedding_centroid | 3.00 | 0.7466 | 0.3459 | 0.9810 |
| d2_logistic_label_predictor | 3.00 | 0.7448 | 0.3383 | 0.8724 |
| d2_embedding_centroid | 10.00 | 0.7414 | 0.3233 | 0.9897 |

Observation:

D2 fixes part of the prediction problem. The best deployable D2 row is:

```text
method: d2_embedding_centroid
alpha: 3.0
K: 16
mean utility: 0.7466
oracle regret: 0.1500
recovered gap vs oracle: 0.3459
label accuracy: 0.9810 against D2 joint labels
```

But it does not reach the utility oracle. It is below:

```text
query_oracle: 0.8966
flat_routecode_utility_oracle: 0.8897
dataset_label_lookup: 0.7534
```

The interpretation is:

> D2 learns labels that are much more predictable, but doing so sacrifices much
> of the utility-oracle advantage.

## 8. New-Model Calibration

Evidence file:

```text
results/llmrouterbench_pilot/table_new_model_integration.csv
```

Setup:

```text
holdout/new models: all six configured pilot models
r examples per label: 1, 2, 4, 8, 16, 32, 64
direct retraining baselines: logistic, SVM, kNN, MLP, gradient boosting
RouteCode K: 16
D2 alpha: 3.0
```

Observation from the Phase C memo:

```text
At r=32 averaged across held-out models:
RouteCode label calibration mean utility: 0.7374
new-model evaluations: about 411.8
strongest matched-budget direct retraining row: MLP at r=2, mean utility 0.6672
```

This is diagnostically positive, but not yet a paper-level claim because it
needs broader held-out models, stronger baselines, and sensitivity checks.

## 9. Residual Concentration and Adaptive Refinement

Evidence files:

```text
results/llmrouterbench_pilot/table_residual_concentration.csv
results/llmrouterbench_pilot/table_residual_risk.csv
```

Observation from the memo:

```text
Top 5% of queries account for 17.7% of predicted RouteCode regret.
Top 10% account for 35.4%.
Top 20% account for 70.7%.
```

Interpretation:

There is some concentration, especially at top 20%, but not enough to support
adaptive refinement as a main claim. The project keeps adaptive refinement
deferred unless stronger risk prediction emerges.

## 10. Split Sensitivity and Benchmark Diagnosis

Evidence files:

```text
results/llmrouterbench_pilot/table_split_sensitivity.csv
results/llmrouterbench_pilot/table_split_rank_correlation.csv
```

The most unstable pilot split was grouped code-domain holdout:

| scenario | rank correlation vs random |
|---|---:|
| leave_domain_out:code | 0.1928 |
| leave_dataset_out:aime | 0.5394 |
| leave_dataset_out:gpqa | 0.5645 |
| leave_domain_out:broad_knowledge | 0.6265 |
| domain_homogeneous:broad_knowledge | 0.6723 |
| cluster_held_out:0 | 0.6826 |
| domain_homogeneous:code | 0.7958 |
| model_pool_holdout:Intern-S1-mini | 0.9940 |

Observation:

Router rankings can reorder under split changes. This supports a benchmark
diagnosis thread, but the domain map is still coarse and manually configured.

## 11. External Baselines and Coverage

Evidence files:

```text
results/llmrouterbench_pilot/table_phase_e_baseline_coverage.csv
results/llmrouterbench_pilot/table_external_command_readiness.csv
results/table_external_blocker_resolution.csv
```

The required Phase E baseline coverage is complete.

Covered groups:

```text
random
cheapest
best single
dataset oracle
query oracle
dataset-label lookup
predicted-topic lookup
embedding-cluster lookup
kNN
logistic/MLP/SVM learned routers
RouteLLM-MF when locally runnable
LLMRouter KNN/SVM adapters
GraphRouter when locally available
Avengers-Pro when included in LLMRouterBench
cost-quality metrics
```

Optional checkpoint-heavy rows remain blocked but documented:

```text
routellm_bert_cli: missing BERT checkpoint
best_route_train_cli: missing local BEST-Route checkpoint and llm_blender
routerdc_train_cli: missing RouterDC local checkpoint and deepspeed
modelsat_train_cli: missing MODEL-SAT base/embedding checkpoints and nltk/deepspeed
```

These are not treated as blockers for the completed Research Flow because the
required baseline coverage table is complete and the missing rows are extra
checkpoint-heavy extensions.

## 12. Global Claim Audit

Evidence file:

```text
results/table_claim_status_global.csv
```

| claim | status | best value | worst value | interpretation |
|---|---:|---:|---:|---|
| low_rate_oracle_codes | diagnostic_supported | 1.0000 | 0.9535 | Low-rate oracle structure exists. Use diagnostic framing. |
| small_inferred_labels | not_supported | 0.3459 | 0.0233 | Do not claim small inferred labels recover most routing performance. |
| model_pool_transfer | mixed_evidence | 0.3083 | -0.0537 | Keep diagnostic. |
| new_model_calibration | diagnostic_alive | 0.8140 | 0.2339 | Promising, not paper-level yet. |
| benchmark_diagnosis | mixed_evidence | 0.7904 | 0.1198 | Keep diagnostic. |
| adaptive_refinement | not_supported | 0.2683 | 0.1521 | Do not claim as main result. |

The recommended framing is:

```text
information_frontier_diagnostic
```

The paper should not claim:

```text
few inferred bits are enough
```

The paper can claim, conservatively:

```text
low-rate utility structure exists, but query-to-label prediction is the
bottleneck.
```

## 13. Completion State

Evidence file:

```text
results/table_research_flow_completion.csv
```

Status:

```text
complete phases: 10
deferred phases: 1
blocked phases: 0
incomplete phases: 0
```

The only deferred phase is:

```text
phase_d5_adaptive_refinement
```

Reason:

```text
Adaptive refinement is deferred unless a stronger deployable residual-risk
signal appears.
```

## 14. Main Observations

### Observation 1: The workload is routable.

Evidence:

```text
best_single mean utility: 0.6672
query_oracle mean utility: 0.8966
oracle regret: 0.2293
```

This means model selection matters on this benchmark and model pool.

### Observation 2: Low-rate utility labels exist.

Evidence:

```text
routecode_oracle_labels K=16:
  mean utility: 0.8897
  recovered gap: 0.9699

regret_routecode_oracle_labels K=8:
  mean utility: 0.8966
  recovered gap: 1.0000
```

This is the strongest positive finding.

### Observation 3: The applied flat RouteCode method does not work.

Evidence:

```text
flat routecode utility oracle K=16: 0.8897
flat routecode logistic label predictor K=16: 0.6138
best_single: 0.6672
```

The applied flat predicted-label method falls below best single.

### Observation 4: D2 improves deployability but not enough.

Evidence:

```text
d2_embedding_centroid alpha=3:
  mean utility: 0.7466
  recovered gap: 0.3459
  label accuracy: 0.9810

query_oracle: 0.8966
flat utility oracle: 0.8897
dataset_label_lookup: 0.7534
```

D2 is the best deployable RouteCode-style method in the pilot, but it does not
reach oracle status and does not beat dataset-label lookup.

### Observation 5: Dataset and topic labels are strong.

Evidence:

```text
dataset_label_lookup: 0.7534, recovered gap 0.3759
predicted_topic_lookup: 0.7448, recovered gap 0.3383
```

This suggests benchmark partition structure carries routing signal. It is a
diagnostic finding, not necessarily a deployable product method.

### Observation 6: The current offensive claim is not supported.

Evidence:

```text
small_inferred_labels global status: not_supported
best observed recovered gap: 0.3459
worst observed recovered gap: 0.0233
```

The pre-committed offensive threshold was roughly 85% recovery. We are nowhere
near that with deployable inferred labels.

## 15. Exact Bottom Line

The applied method does not achieve the oracle.

The current result is:

```text
Oracle route labels show that routing behavior is compressible.
Deployable query-to-route-label prediction is still weak.
Predictability-constrained labels improve the situation, but not enough to
support the claim that few inferred bits are enough.
```

The strongest honest paper direction is:

```text
RouteCode as a routing information-frontier and benchmark-diagnosis framework.
```

The paper should emphasize:

```text
1. Low-rate utility structure exists.
2. There is a large oracle-to-deployable gap.
3. That gap identifies query-to-label prediction as the key bottleneck.
4. Dataset/topic artifacts explain part of observed routing performance.
5. New-model calibration is promising but still diagnostic.
```

The paper should not currently claim:

```text
small inferred route labels recover most router performance
adaptive refinement improves cost-quality
unconditional model-pool transfer
```

## 16. Where To Read The Raw Evidence

Primary result files:

```text
results/llmrouterbench_pilot/table_routability.csv
results/llmrouterbench_pilot/table_recovered_gap.csv
results/llmrouterbench_pilot/table_rate_distortion.csv
results/llmrouterbench_pilot/table_predictability_constrained.csv
results/llmrouterbench_pilot/table_new_model_integration.csv
results/llmrouterbench_pilot/table_split_sensitivity.csv
results/llmrouterbench_pilot/table_phase_e_baseline_coverage.csv
results/table_claim_status_global.csv
results/table_paper_evidence_summary.csv
results/table_research_flow_completion.csv
```

Main narrative memos:

```text
results/llmrouterbench_pilot/phase_c_observation_memo.md
results/llmrouterbench_pilot/phase_d_method_memo.md
results/llmrouterbench_pilot/phase_e5_new_model_calibration_memo.md
results/llmrouterbench_pilot/phase_f_g_ablation_memo.md
results/llmrouterbench_pilot/phase_g_sensitivity_memo.md
results/phase_h_paper_evidence_summary.md
results/phase_h_research_flow_completion_audit.md
paper_notes.md
```

Figures:

```text
results/llmrouterbench_pilot/fig_compression_ladder.pdf
results/llmrouterbench_pilot/fig_rate_distortion.pdf
results/llmrouterbench_pilot/fig_predictability_constrained_tradeoff.pdf
results/llmrouterbench_pilot/fig_transfer_calibration_curve.pdf
results/llmrouterbench_pilot/fig_split_sensitivity.pdf
results/llmrouterbench_pilot/fig_sensitivity_summary.pdf
```

