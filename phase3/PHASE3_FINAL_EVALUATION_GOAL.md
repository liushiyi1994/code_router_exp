# Phase 3 Final Evaluation Goal

This document defines the focused Phase 3 evaluation package for RouteCode /
ProbeCode. The goal is to freeze the current method, run a small but defensible
set of comparisons, and test the most important research claim:

```text
learned route/probe states are useful cost-aware calibration strata for model
routing and new-model onboarding.
```

The main result should not try to compare against every router paper. The main
result should show that our method is:

1. close to the cost-aware oracle;
2. cheaper than using GPT/Gemini everywhere;
3. competitive in quality with large-model-only policies;
4. a good calibration strategy when adding a new model.

## Current Starting Point

The current supported Phase 3 result is:

```text
ProbeCode / ProbeRoute++ reaches the configured oracle-level target on cached
Broad100 and controlled exact-math when the action pool includes learned
verifiability states plus verifiable local/tool actions.
```

Important caveat:

```text
Do not claim that clean no-tool query-only routing is solved.
The no-tool action-pool oracle still leaves a large gap to the full-action-pool
oracle.
```

Read before running or editing Phase 3 experiments:

1. `AGENTS.md`
2. `PHASE3_AGENT_HANDOFF.md`
3. `phase3/BENCHMARK_AGNOSTIC_PROBE_STATE_PLAN.md`
4. `results/controlled/phase3_final_claim_package/PHASE3_FINAL_CLAIM_PACKAGE.md`
5. `results/controlled/phase3_oracle_level_modification/ORACLE_LEVEL_METHOD_MODIFICATION_MEMO.md`
6. `results/controlled/PHASE3_GOAL_COMPLETION_AUDIT.md`

## Final Method To Package

Freeze one final method implementation and config before the final evaluation.

Working name:

```text
ProbeCode-StateCal
```

Conceptual flow:

```text
query
  -> cheap local/verifiable behavior
  -> learned probe/route state
  -> state-to-action utility table
  -> cost-aware selected action
```

The method must keep these two parts separate:

```text
query -> state
state -> model/action
```

This separation is central to the calibration claim. When a new model arrives,
we should keep `query -> state` frozen and update only the state-to-model
utility table.

Required final-method artifacts:

- `configs/probecode_final_eval.yaml`
- `experiments/230_phase3_package_final_method.py`
- `results/phase3_final/final_method/METHOD_CARD.md`
- `results/phase3_final/final_method/table_final_state_cards.csv`
- `results/phase3_final/final_method/code_cards.md`

The method card must document:

- model/action pool;
- state features;
- whether deterministic/verifiable actions are allowed;
- train/validation/test split rule;
- thresholds selected on validation;
- utility objective and cost lambda;
- known caveats.

## Benchmark Scope

Use existing cached outcome matrices first. New model calls are allowed only
when a later run explicitly provides a cost cap, cache path, and command.

Primary benchmark package:

```text
Broad controlled benchmark set:
- GSM8K
- MATH500
- AIME
- HumanEval
- MBPP
- GPQA
- MMLU-Pro
- BBH
- LiveMathBench if available
```

Minimum acceptable final run:

```text
At least 8 benchmark families.
At least one held-out test split.
All methods evaluated on the same query/model/action matrix.
No train/test leakage in states, clusters, thresholds, or action tables.
```

## Model Scope

Small/local model pool:

- Include all cached or runnable small/local language models in the final
  matrix.
- At minimum include one cheap local model, one medium local model, and one
  stronger local model.
- Use vLLM for local model serving when live local calls are needed.

Large/frontier model pool:

- GPT-family cached/API model, if records exist.
- Gemini-family cached/API model, if records exist.

Claude/Anthropic:

- Document as a future closed-source provider family.
- Do not use Claude in the main Phase 3 run unless API access, pricing,
  budget, caching, and token logging are explicitly enabled.

## Reduced Baseline Set

The main comparison table should stay small. Do not add every available router
baseline to the headline table.

### Required Simple Baselines

1. Random assignment
   - Randomly choose an action/model from the allowed action pool.
   - Report mean and confidence interval over multiple random seeds.

2. All small/local models
   - Route every query to each small/local model.
   - Report every local model and the best local model.

3. All GPT
   - Route every query to the GPT-family model.
   - This is the main strong closed-source quality/cost reference.

4. All Gemini
   - Route every query to the Gemini-family model.
   - This is the second strong closed-source quality/cost reference.

5. Cost-aware oracle
   - Post-hoc upper bound.
   - Not deployable; used only to measure distance to oracle.

### Required Open-Source Literature Baselines

Pick exactly these three open-source comparison methods first. If one cannot
run cleanly on the final matrix, record the blocker and use the fallback.

1. RouteLLM
   - Repo: `https://github.com/lm-sys/routellm`
   - Preferred methods: Matrix Factorization router and/or BERT router.
   - Why: canonical open-source learned LLM router.

2. LLMRouter / GraphRouter
   - Repo: `https://github.com/ulab-uiuc/LLMRouter`
   - Preferred method: GraphRouter if available through the library.
   - Fallback inside same repo: kNN or SVM router.
   - Why: open-source router library with structured/router baselines.

3. Avengers-Pro
   - Repo: `https://github.com/ZhangYiqun018/AvengersPro`
   - Local fallback source: `data/raw/external/LLMRouterBench/baselines/AvengersPro`
   - Why: cluster-based performance-efficiency routing baseline, close to our
     state/grouping story.

For every external baseline, save:

- repo URL;
- commit hash or release;
- command run;
- adapter code path;
- split;
- hyperparameters;
- runtime;
- any mismatch from the original paper setup.

Do not put additional baselines in the main table unless one of the three above
is blocked.

## Main Metrics

Report these for every main routing method:

- mean quality;
- mean cost-aware utility;
- oracle regret;
- quality gap to cost-aware oracle;
- oracle utility ratio;
- remote cost per 1K queries;
- normalized remote cost versus all-GPT and all-Gemini;
- frontier-call rate;
- local-call rate;
- p50 latency if available;
- p95 latency if available;
- quality at fixed cost;
- cost at fixed quality.

Primary routing success gate:

```text
quality gap to cost-aware oracle <= 0.03
oracle utility ratio >= 0.95
frontier-call rate <= 0.40
```

Cost success gate:

```text
remote cost materially lower than all-GPT and all-Gemini
```

Calibration success gate:

```text
state-based calibration reaches matched utility with 3x--5x fewer new-model
evaluations than direct router retraining or random calibration.
```

## Priority Experiments

These are the experiments that matter for the Phase 3 paper story.

### Experiment 0: Main Routing Evaluation

Question:

```text
Is ProbeCode-StateCal close to oracle, cheaper than GPT/Gemini everywhere, and
competitive with the selected open-source router baselines?
```

Setup:

- use the final held-out test split over the broad benchmark outcome matrix;
- evaluate all methods on the same query/model/action rows;
- select thresholds and hyperparameters on validation only;
- report final held-out test once after method selection.

Compare:

- random assignment;
- all small/local models;
- all GPT;
- all Gemini;
- RouteLLM;
- LLMRouter / GraphRouter;
- Avengers-Pro;
- ProbeCode-StateCal;
- cost-aware oracle.

Outputs:

- `results/phase3_final/main_eval/table_main_routing_eval.csv`
- `results/phase3_final/main_eval/table_per_benchmark_eval.csv`
- `results/phase3_final/main_eval/table_action_mix.csv`
- `results/phase3_final/main_eval/fig_quality_cost_frontier.pdf`
- `results/phase3_final/main_eval/fig_oracle_gap.pdf`
- `results/phase3_final/main_eval/MAIN_ROUTING_EVAL_MEMO.md`

Expected result:

```text
ProbeCode-StateCal is the closest deployable method to the cost-aware oracle,
is much cheaper than all-GPT/all-Gemini, and sits on or near the quality-cost
frontier.
```

### Experiment 1: Are States Good Calibration Strata?

Question:

```text
Are learned route/probe states better calibration groups than random groups,
dataset labels, or embedding clusters?
```

Setup:

Use existing benchmark outcome matrices. Compare grouping methods:

- random groups;
- dataset/domain labels;
- embedding clusters;
- utility clusters;
- RouteCode states;
- calibration-aware RouteCode states.

For each grouping method, estimate how stable model utility is inside each
group/state.

Metrics:

- within-state utility variance;
- within-state quality variance;
- new-model utility estimation error;
- best-model identification accuracy per state;
- state-level confidence interval width;
- samples needed for stable utility estimate;
- state traffic share.

Outputs:

- `experiments/232_phase3_calibration_strata.py`
- `results/phase3_final/calibration_strata/table_state_variance.csv`
- `results/phase3_final/calibration_strata/table_state_estimation_error.csv`
- `results/phase3_final/calibration_strata/table_state_best_model_accuracy.csv`
- `results/phase3_final/calibration_strata/fig_state_variance.pdf`
- `results/phase3_final/calibration_strata/CALIBRATION_STRATA_MEMO.md`

Expected result:

```text
RouteCode states have lower within-state utility variance than dataset labels
and embedding clusters.
```

Interpretation:

```text
A few samples from a state should reliably estimate model performance on that
state.
```

This is the most important experiment for proving that the states are useful
calibration data, not just a routing trick.

### Experiment 2: Simulated New-Model Onboarding

Question:

```text
Can we add a held-out model cheaply by calibrating state-level utility instead
of retraining a full query-to-model router?
```

Protocol:

For each representative held-out model in the existing benchmark matrix:

1. remove the model from the model pool;
2. learn/freeze route states using the remaining models;
3. treat the removed model as new;
4. calibrate the new model with limited examples;
5. update the state-to-model utility table;
6. evaluate routing on held-out test queries.

Held-out model types:

- cheap local model;
- medium/strong local model;
- code specialist if available;
- math/reasoning local model if available;
- GPT or Gemini frontier-style model if outcome records exist.

Calibration budgets:

```text
B = 20, 40, 80, 160, 320, 640 total new-model evaluations
```

Methods to compare:

- random query calibration;
- dataset-stratified calibration;
- embedding-cluster calibration;
- uniform route-state calibration;
- active route-state calibration;
- direct router retraining under the same new-model evaluation budget;
- full calibration / oracle calibration.

Active acquisition rules to include:

- uniform per state;
- traffic-weighted;
- uncertainty-only;
- smallest gap to current best;
- value-of-calibration or Thompson-style sampling if simple to implement.

Metrics:

- mean utility after onboarding;
- quality after onboarding;
- regret to full calibration;
- number of new-model evaluations;
- calibration cost;
- quality at fixed cost;
- cost at fixed quality;
- best-model identification accuracy per state;
- training time.

Outputs:

- `experiments/233_phase3_new_model_onboarding.py`
- `results/phase3_final/new_model_onboarding/table_new_model_onboarding.csv`
- `results/phase3_final/new_model_onboarding/table_onboarding_by_model_type.csv`
- `results/phase3_final/new_model_onboarding/table_acquisition_ablation.csv`
- `results/phase3_final/new_model_onboarding/fig_utility_vs_calibration_budget.pdf`
- `results/phase3_final/new_model_onboarding/fig_quality_vs_calibration_budget.pdf`
- `results/phase3_final/new_model_onboarding/NEW_MODEL_ONBOARDING_MEMO.md`

Expected result:

```text
active route-state calibration reaches the same utility with 3x--5x fewer
evaluations than direct retraining or random calibration.
```

Very strong result:

```text
active state calibration reaches within 3 quality points of full calibration
using <=160--320 evaluations.
```

### Experiment 3: Frozen State Router vs Full Router Retraining

Question:

```text
Can we avoid retraining the query-to-model router when a new model arrives?
```

Protocol:

Compare two systems under the same new-model evaluation budget.

Direct router baseline:

```text
query features -> model
```

When the new model arrives:

```text
retrain direct router with limited new-model labels
```

ProbeCode-StateCal:

```text
query -> state is frozen
only U(state, new_model) is updated
```

Metrics:

- new-model evaluation budget;
- training time;
- routing utility;
- quality;
- cost;
- latency if available;
- direct-router degradation under low budget;
- state-router degradation under low budget.

Outputs:

- `experiments/234_phase3_frozen_state_vs_retrain.py`
- `results/phase3_final/frozen_state_vs_retrain/table_frozen_state_vs_retrain.csv`
- `results/phase3_final/frozen_state_vs_retrain/fig_budget_vs_utility.pdf`
- `results/phase3_final/frozen_state_vs_retrain/FROZEN_STATE_VS_RETRAIN_MEMO.md`

Expected result:

```text
frozen state router + active calibration ~= retrained direct router quality
with fewer evaluations and less training.
```

This directly supports the claim:

```text
No extensive router retraining is needed for new-model onboarding.
```

### Experiment 4: Cost And Price Sensitivity

Question:

```text
If model prices change, can we update the state-to-model action table without
retraining the query-to-state model?
```

Protocol:

Use the same frozen states and sweep:

```text
frontier price multiplier = 0.5, 1, 2, 5
lambda_cost = 0, 0.1, 0.35, 0.7, 1.0
```

Metrics:

- mean utility;
- mean quality;
- remote cost per 1K queries;
- frontier-call rate;
- action-table changes by state;
- gap to cost-aware oracle under each price setting.

Outputs:

- `experiments/236_phase3_cost_sensitivity.py`
- `results/phase3_final/sensitivity/table_price_sensitivity.csv`
- `results/phase3_final/sensitivity/fig_price_sensitivity.pdf`
- `results/phase3_final/sensitivity/SENSITIVITY_MEMO.md`

Expected result:

```text
the same query-to-state model can adapt the state-to-action table as model
prices change.
```

## Optional Experiments

These are useful but not required for the first Phase 3 final package.

### Optional A: Real Local/Frontier New-Model Calibration

Run only if live calls are explicitly approved.

Setup:

- use vLLM for local model serving;
- use capped GPT/Gemini calls only if budget and cache paths are explicit;
- test practical budgets: `40, 80, 160, 320` evaluations.

Outputs:

- `results/phase3_final/real_new_model_calibration/table_real_new_model_calibration.csv`
- `results/phase3_final/real_new_model_calibration/cost_latency_summary.csv`
- `results/phase3_final/real_new_model_calibration/REAL_NEW_MODEL_CALIBRATION_MEMO.md`

### Optional B: Minimal Ablation Table

Run only the ablations needed to explain the final method:

- full method;
- no probe/local-behavior features;
- no RouteCode state, direct action predictor only;
- no active calibration, uniform state calibration only;
- local-only action pool;
- frontier-only action pool.

Outputs:

- `results/phase3_final/ablation/table_final_ablation.csv`
- `results/phase3_final/ablation/fig_ablation_utility.pdf`
- `results/phase3_final/ablation/ABLATION_MEMO.md`

## Final Report Package

After the priority experiments, generate:

- `results/phase3_final/FINAL_EVALUATION_REPORT.md`
- `results/phase3_final/table_final_claims.csv`
- `results/phase3_final/table_final_main_eval.csv`
- `results/phase3_final/table_final_baselines.csv`
- `results/phase3_final/table_final_calibration.csv`
- `results/phase3_final/table_final_onboarding.csv`
- `results/phase3_final/table_final_sensitivity.csv`
- `results/phase3_final/fig_final_quality_cost_frontier.pdf`
- `results/phase3_final/fig_final_calibration_efficiency.pdf`
- `results/phase3_final/fig_final_oracle_gap.pdf`

The final report must answer:

1. Is ProbeCode-StateCal closest to the oracle among deployable methods?
2. Is it cheaper than all-GPT and all-Gemini?
3. Is it similarly accurate to large-model-only policies?
4. Are learned states better calibration strata than dataset labels or
   embedding clusters?
5. Can a new model be added with fewer evaluations than direct retraining?
6. Can the state-to-action table adapt when GPT/Gemini prices change?
7. Which ablations explain the gain?
8. Which settings break the method?

## Claim Rules

Allowed if supported:

```text
State-based ProbeCode can reach near-oracle cost-aware routing in the controlled
verifiability/action-pool setting.
```

Allowed if calibration experiments support it:

```text
Learned route states are effective calibration strata for new-model onboarding.
```

Allowed if the onboarding experiment supports it:

```text
Frozen query-to-state routing can reduce the amount of new-model calibration
needed when the model pool changes.
```

Do not claim unless the final tables support it:

```text
RouteCode is SOTA.
Clean no-tool routing is solved.
Small inferred query labels always recover oracle routing.
The method works for all benchmarks or all model pools.
```

## Completion Criteria

This goal is complete only when:

- the final method is frozen and documented;
- all model/action outputs needed for the final matrix are cached;
- main routing evaluation table exists;
- reduced baseline table exists;
- three open-source literature baselines are either run or explicitly blocked
  with documented reasons;
- calibration-strata experiment is complete;
- simulated new-model onboarding experiment is complete;
- frozen-state vs retrain comparison is complete;
- cost/price sensitivity table exists;
- final report states supported and unsupported claims conservatively.

