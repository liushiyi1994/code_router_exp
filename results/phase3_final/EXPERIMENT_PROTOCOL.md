# Phase 3 Experiment Protocol

This document records how the current final result package was produced and how
to interpret the benchmark split.

## Research Question

RouteCode asks how much query information is needed to choose a good model under
a cost-quality objective:

```text
U(q, a) = quality(q, a) - lambda_cost * normalized_cost(q, a)
```

For Phase 3, the deployable method became ProbeCode-StateCal:

```text
query + cheap local/probe behavior
  -> predicted utility/probe state
  -> cost-aware action table
  -> selected local, verifiable, GPT, or Gemini action
```

## Data And Benchmark Scope

The final Broad100 package uses 860 queries across 9 benchmark families:

```text
aime
bbh
gpqa
gsm8k
humaneval
livemathbench
math500
mbpp
mmlupro
```

The split is by `query_id`, not by query-model row. The final evaluation uses
train, validation, and test queries from the same benchmark families.

Important interpretation:

- Current evidence supports held-out-query generalization within these 9
  benchmark families.
- Current evidence does not yet support an out-of-benchmark-family claim.
- To claim new-benchmark generalization, run leave-one-benchmark-family-out or
  add new benchmark families that were not used to learn/select states.

## Models And Actions

The final action pool includes local vLLM/cached local actions, verifiable
actions, and frontier provider actions.

Main live/cached Stage0 model coverage:

| Model | Provider | Successful Queries | Mean Quality | Total Cost |
| --- | --- | ---: | ---: | ---: |
| gpt-5.5 | openai | 860 | 0.6512 | $6.2018 |
| gemini-3.5-flash | google | 860 | 0.3663 | $0.4951 |
| qwen3-14b-awq-local | local | 859 | 0.4820 | $0.0000 |
| qwen3-8b-local | local | 859 | 0.3260 | $0.0000 |
| qwen3-32b-awq-local | local | 819 | 0.4982 | $0.0000 |
| qwen3-4b-local | local | 814 | 0.3575 | $0.0000 |

The broader final routing table also includes verifiable/tool-style actions and
cached strong-solve/self-consistency variants. Claude is documented as a target
provider family, but it was not tested in this final package.

## State Learning And Selection

The current states are learned from the Broad100 train split.

Main compact state policy:

- state method: `gb_depth2_thr0.9844_state_k8`;
- source: cached Broad100 train outcomes plus observable local/probe behavior;
- selected on validation;
- evaluated on held-out Broad100 test queries.

Predicted utility-state calibration:

- utility states are learned from train-split utility vectors;
- random-forest or extra-trees predictors map observable probe features to
  states;
- K and predictor variants are chosen on validation;
- held-out test reports use only the selected state method.

The strongest calibration-strata state is:

```text
predicted_utility_state_rf_probe_plus_benchmark_k16
```

The selected onboarding state is:

```text
predicted_utility_state_rf_probe_plus_benchmark_k6
```

## Experiment Order

All commands below are cache-backed unless explicitly stated.

```bash
PYTHONPATH=src python experiments/230_phase3_package_final_method.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/237_phase3_broad100_literature_baselines.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/231_phase3_final_main_routing_eval.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/232_phase3_calibration_strata.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/233_phase3_new_model_onboarding.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/234_phase3_frozen_state_vs_retrain.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/236_phase3_cost_sensitivity.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/238_phase3_final_ablation.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/239_phase3_real_new_model_calibration.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/240_phase3_predicted_utility_state_calibration.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/241_phase3_live_predicted_utility_state_calibration.py --config configs/probecode_final_eval.yaml
PYTHONPATH=src python experiments/235_phase3_final_report.py --config configs/probecode_final_eval.yaml
```

Optional live smoke calls are controlled separately and should only be run with
explicit budget approval:

```bash
PYTHONPATH=src python experiments/239_phase3_real_new_model_calibration.py \
  --config configs/probecode_final_eval.yaml \
  --run-live-smoke \
  --smoke-limit 8
```

## Main Evaluations

### 1. Accuracy, Cost, And Latency

Artifact:

- `main_eval/table_main_routing_eval.csv`
- `table_final_main_eval.csv`

Primary comparison:

| Method | Quality | Utility | Remote $ / 1K | P95 Latency | Frontier Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cost-aware oracle | 0.8721 | 0.8463 | 0.5324 | 4.3124 | 0.1395 |
| ProbeCode-StateCal | 0.8547 | 0.8238 | 0.6354 | 4.9226 | 0.1919 |
| RouteLLM MF adapter | 0.7326 | 0.6076 | 2.5756 | 5.5709 | 0.5814 |
| LLMRouter/GraphRouter kNN fallback | 0.7616 | 0.6061 | 3.2054 | 6.7348 | 0.7674 |
| AvengersPro adapter | 0.7209 | 0.5899 | 2.6988 | 6.4580 | 0.6221 |
| Best local single | 0.4767 | 0.4767 | 0.0000 | 6.5401 | 0.0000 |
| All Gemini strong | 0.6977 | 0.4727 | 4.6347 | 10.2808 | 1.0000 |
| All GPT | 0.5930 | 0.2271 | 7.5399 | 10.5162 | 1.0000 |
| Random assignment | 0.4151 | 0.3382 | 1.5852 | 7.1470 | 0.3640 |

### 2. Sensitivity

Artifacts:

- `sensitivity/table_price_sensitivity.csv`
- `sensitivity/table_price_sensitivity_action_table.csv`
- `sensitivity/fig_price_sensitivity.pdf`

What changed:

- lambda cost: `0.00`, `0.10`, `0.35`, `0.70`, `1.00`;
- frontier price multiplier: `0.5`, `1.0`, `2.0`, `5.0`.

Finding:

- the frozen query-to-state model can adapt to price changes by updating the
  state-to-action table;
- higher cost weight and higher frontier prices reduce frontier actions.

### 3. Ablations

Artifacts:

- `ablation/table_final_ablation.csv`
- `ablation/fig_ablation_utility.pdf`

Key rows:

| Ablation | Utility | Delta vs Full |
| --- | ---: | ---: |
| full_method | 0.8238 | 0.0000 |
| compact_routecode_state_policy | 0.8136 | -0.0103 |
| no_verifiable_tool_actions | 0.7360 | -0.0878 |
| local_only_action_pool_diagnostic | 0.6919 | -0.1320 |
| direct_probe_action_predictor_no_state | 0.6733 | -0.1505 |
| no_probe_local_behavior_features | 0.6652 | -0.1586 |
| direct_action_probability_no_state | 0.6208 | -0.2030 |

Finding:

- compact state routing is close to the full method;
- probe/local behavior features matter;
- direct no-state predictors are much weaker;
- removing verifiable/tool-style actions causes a large drop, so the current
  success should not be sold as clean no-tool routing.

### 4. Calibration-Strata Claim

Artifacts:

- `live_predicted_utility_states/table_live_predicted_state_variance.csv`
- `live_predicted_utility_states/table_live_predicted_state_claims.csv`
- `live_predicted_utility_states/fig_predicted_state_variance.pdf`

Live held-out test variance:

| Grouping | Utility Variance |
| --- | ---: |
| utility_cluster_k8_diagnostic | 0.0667 |
| predicted_utility_state_rf_probe_plus_benchmark_k16 | 0.1366 |
| benchmark_label | 0.1666 |
| text_cluster_k8 | 0.1923 |

Claim:

- predicted utility states are better live calibration strata than benchmark
  labels or text clusters on held-out Broad100 test queries.

### 5. New-Model / Frontier Onboarding Claim

Artifacts:

- `live_predicted_utility_states/table_live_frontier_onboarding_validation.csv`
- `live_predicted_utility_states/table_live_frontier_onboarding_test.csv`
- `live_predicted_utility_states/table_live_frontier_budget_efficiency.csv`
- `live_predicted_utility_states/fig_predicted_state_onboarding.pdf`

Validation selected budget:

```text
40 new-model evaluations
```

Held-out test at budget 40:

| Method | Utility |
| --- | ---: |
| active predicted utility state | 0.5627 |
| best competitor at same budget | 0.5510 |

Budget-to-match:

| Competitor | Active Budget | Match Budget | Reduction |
| --- | ---: | ---: | ---: |
| uniform state calibration | 40 | 80 | 2.0x |
| random calibration | 40 | 320 | 8.0x |
| direct retrain proxy | 40 | >320 | at least 8.0x |

Claim:

- active predicted-state calibration is supported for the GPT/Gemini
  frontier-onboarding slice;
- all-model average active acquisition is only weakly supported.

## What Still Needs New Benchmarks

For a stronger paper claim, add at least one of:

1. Leave-one-benchmark-family-out evaluation:
   - learn states on 8 families;
   - select K/thresholds on validation from those families;
   - test on the held-out 9th family;
   - rotate across all 9 families.

2. New benchmark family evaluation:
   - learn/select states on the current Broad100 families;
   - freeze the query-to-state model;
   - add new benchmarks not used in state learning;
   - report whether state calibration and routing still work.

3. Provider-family onboarding:
   - treat a new GPT/Gemini/Claude model as unseen;
   - calibrate it with 40, 80, 160, and 320 examples;
   - compare state calibration against random calibration and direct router
     retraining under the same evaluation budget.

Until one of these is done, use this wording:

```text
The current states generalize to held-out queries within the Broad100 benchmark
families. Cross-benchmark-family generalization remains a required next test.
```
