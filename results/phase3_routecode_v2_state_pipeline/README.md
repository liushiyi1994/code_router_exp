# RouteCode V2 State Pipeline

This run implements the split proposed in the Phase 3 direction change:

```text
Model 1: utility matrix -> latent states + state-to-model table
Model 2: query text embedding -> p(state | query)
```

State learner variants:

- `raw_kmeans`
- `relative_kmeans`
- `two_stage_relative_kmeans`
- `calibration_refined`

Query-to-state predictors:

- KNN over local query embeddings
- MLP over local query embeddings

Low-confidence predictions trigger either:

- deployable fallback to the train-best model; or
- diagnostic active-state reveal, an upper bound for what active probing could recover.

## Top Test Policies

| method | quality | utility | oracle utility ratio | active/probe rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| relative_kmeans_k16_diagnostic_true_state | 0.7529 | 0.7295 | 0.9970 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 80, "qwen3-32b-awq-local": 17, "qwen3-4b-local": 20, "qwen3-8b-local": 23} |
| calibration_refined_k16_diagnostic_true_state | 0.7529 | 0.7295 | 0.9970 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 80, "qwen3-32b-awq-local": 17, "qwen3-4b-local": 20, "qwen3-8b-local": 23} |
| relative_kmeans_k24_diagnostic_true_state | 0.7412 | 0.7177 | 0.9809 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 90, "qwen3-32b-awq-local": 24, "qwen3-4b-local": 15, "qwen3-8b-local": 11} |
| calibration_refined_k24_diagnostic_true_state | 0.7412 | 0.7177 | 0.9809 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 90, "qwen3-32b-awq-local": 24, "qwen3-4b-local": 15, "qwen3-8b-local": 11} |
| raw_kmeans_k24_diagnostic_true_state | 0.7353 | 0.7138 | 0.9757 | 0.0000 | {"gemini-3.5-flash": 12, "gpt-5.5": 10, "qwen3-14b-awq-local": 70, "qwen3-32b-awq-local": 12, "qwen3-4b-local": 55, "qwen3-8b-local": 11} |
| two_stage_relative_kmeans_k24_diagnostic_true_state | 0.7294 | 0.7081 | 0.9678 | 0.0000 | {"gemini-3.5-flash": 11, "gpt-5.5": 10, "qwen3-14b-awq-local": 110, "qwen3-32b-awq-local": 20, "qwen3-4b-local": 16, "qwen3-8b-local": 3} |
| raw_kmeans_k16_diagnostic_true_state | 0.7176 | 0.6962 | 0.9516 | 0.0000 | {"gemini-3.5-flash": 12, "gpt-5.5": 10, "qwen3-14b-awq-local": 58, "qwen3-32b-awq-local": 22, "qwen3-4b-local": 14, "qwen3-8b-local": 54} |
| two_stage_relative_kmeans_k16_diagnostic_true_state | 0.6765 | 0.6506 | 0.8893 | 0.0000 | {"gemini-3.5-flash": 10, "gpt-5.5": 13, "qwen3-14b-awq-local": 95, "qwen3-32b-awq-local": 39, "qwen3-4b-local": 13} |
| two_stage_relative_kmeans_k24_mlp_active_state_reveal | 0.5941 | 0.5729 | 0.7830 | 0.2059 | {"gemini-3.5-flash": 14, "gpt-5.5": 11, "qwen3-14b-awq-local": 129, "qwen3-32b-awq-local": 11, "qwen3-4b-local": 4, "qwen3-8b-local": 1} |
| relative_kmeans_k16_mlp_active_state_reveal | 0.6000 | 0.5698 | 0.7788 | 0.3706 | {"gemini-3.5-flash": 15, "gpt-5.5": 12, "qwen3-14b-awq-local": 87, "qwen3-32b-awq-local": 24, "qwen3-4b-local": 8, "qwen3-8b-local": 24} |
| calibration_refined_k16_mlp_active_state_reveal | 0.6000 | 0.5698 | 0.7788 | 0.3706 | {"gemini-3.5-flash": 15, "gpt-5.5": 12, "qwen3-14b-awq-local": 87, "qwen3-32b-awq-local": 24, "qwen3-4b-local": 8, "qwen3-8b-local": 24} |
| two_stage_relative_kmeans_k24_knn_active_state_reveal | 0.5706 | 0.5535 | 0.7565 | 0.1941 | {"gemini-3.5-flash": 10, "gpt-5.5": 7, "qwen3-14b-awq-local": 118, "qwen3-32b-awq-local": 29, "qwen3-4b-local": 5, "qwen3-8b-local": 1} |

## Query-to-State Diagnostics

| state method | K | predictor | state accuracy | mean confidence | ECE | probe rate | covered accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| raw_kmeans | 16 | knn | 0.3176 | 0.4196 | 0.1962 | 0.2118 | 0.3731 |
| raw_kmeans | 16 | mlp | 0.3118 | 0.3420 | 0.0726 | 0.2118 | 0.3507 |
| raw_kmeans | 24 | knn | 0.3118 | 0.3953 | 0.1708 | 0.2294 | 0.3588 |
| raw_kmeans | 24 | mlp | 0.2882 | 0.4091 | 0.1306 | 0.2882 | 0.3471 |
| calibration_refined | 16 | mlp | 0.2824 | 0.3540 | 0.1266 | 0.3706 | 0.3832 |
| relative_kmeans | 16 | mlp | 0.2824 | 0.3540 | 0.1266 | 0.3706 | 0.3832 |
| calibration_refined | 24 | mlp | 0.2706 | 0.5963 | 0.3257 | 0.1647 | 0.2958 |
| calibration_refined | 16 | knn | 0.2706 | 0.4033 | 0.1702 | 0.1941 | 0.3212 |
| relative_kmeans | 24 | mlp | 0.2706 | 0.5963 | 0.3257 | 0.1647 | 0.2958 |
| relative_kmeans | 16 | knn | 0.2706 | 0.4033 | 0.1702 | 0.1941 | 0.3212 |
| calibration_refined | 24 | knn | 0.2647 | 0.3957 | 0.1796 | 0.2235 | 0.3182 |
| relative_kmeans | 24 | knn | 0.2647 | 0.3957 | 0.1796 | 0.2235 | 0.3182 |

## New Benchmark Smoke

| method | quality | utility | oracle utility ratio | gate rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| raw_kmeans_k16_mlp_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 1.0000 | {"gpt-5.5": 15} |
| relative_kmeans_k16_knn_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 0.8667 | {"gpt-5.5": 15} |
| relative_kmeans_k16_mlp_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 1.0000 | {"gpt-5.5": 15} |
| two_stage_relative_kmeans_k16_mlp_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 1.0000 | {"gpt-5.5": 15} |
| calibration_refined_k16_knn_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 0.8667 | {"gpt-5.5": 15} |
| calibration_refined_k16_mlp_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 1.0000 | {"gpt-5.5": 15} |
| two_stage_relative_kmeans_k16_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| raw_kmeans_k24_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| relative_kmeans_k24_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| two_stage_relative_kmeans_k24_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| calibration_refined_k24_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| two_stage_relative_kmeans_k24_mlp_semantic_or_lowconf_remote_gate | 0.6667 | 0.3259 | 0.5932 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |

## Active Query-State Labeling Simulation

This is train-only active learning for the query-to-state classifier. It starts
with one labeled query per state, then either samples random additional state
labels or samples low-confidence training queries. Evaluation is on held-out
Broad100 test queries.

| predictor | strategy | labeled queries | state accuracy | utility | oracle utility ratio | selected models |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| mlp | random | 64 | 0.2118 | 0.4195 | 0.5734 | {"gemini-3.5-flash": 1, "gpt-5.5": 19, "qwen3-14b-awq-local": 82, "qwen3-32b-awq-local": 27, "qwen3-4b-local": 31, "qwen3-8b-local": 10} |
| knn | random | 64 | 0.2824 | 0.4111 | 0.5619 | {"gemini-3.5-flash": 13, "qwen3-14b-awq-local": 105, "qwen3-32b-awq-local": 9, "qwen3-4b-local": 34, "qwen3-8b-local": 9} |
| mlp | active_low_confidence | 64 | 0.0647 | 0.4063 | 0.5554 | {"gemini-3.5-flash": 2, "gpt-5.5": 25, "qwen3-14b-awq-local": 77, "qwen3-32b-awq-local": 24, "qwen3-4b-local": 41, "qwen3-8b-local": 1} |
| knn | active_low_confidence | 64 | 0.2882 | 0.3118 | 0.4261 | {"qwen3-14b-awq-local": 69, "qwen3-32b-awq-local": 2, "qwen3-4b-local": 39, "qwen3-8b-local": 60} |
| mlp | active_low_confidence | 128 | 0.2235 | 0.4387 | 0.5996 | {"gemini-3.5-flash": 2, "gpt-5.5": 6, "qwen3-14b-awq-local": 104, "qwen3-32b-awq-local": 32, "qwen3-4b-local": 10, "qwen3-8b-local": 16} |
| knn | active_low_confidence | 128 | 0.2353 | 0.4285 | 0.5856 | {"gemini-3.5-flash": 1, "gpt-5.5": 3, "qwen3-14b-awq-local": 87, "qwen3-32b-awq-local": 20, "qwen3-4b-local": 29, "qwen3-8b-local": 30} |
| mlp | random | 128 | 0.2471 | 0.4209 | 0.5753 | {"gemini-3.5-flash": 26, "gpt-5.5": 4, "qwen3-14b-awq-local": 106, "qwen3-32b-awq-local": 20, "qwen3-4b-local": 2, "qwen3-8b-local": 12} |
| knn | random | 128 | 0.2588 | 0.4172 | 0.5702 | {"gemini-3.5-flash": 5, "gpt-5.5": 6, "qwen3-14b-awq-local": 92, "qwen3-32b-awq-local": 9, "qwen3-4b-local": 7, "qwen3-8b-local": 51} |
| mlp | active_low_confidence | 256 | 0.3000 | 0.5150 | 0.7039 | {"gemini-3.5-flash": 2, "gpt-5.5": 19, "qwen3-14b-awq-local": 92, "qwen3-32b-awq-local": 43, "qwen3-4b-local": 3, "qwen3-8b-local": 11} |
| mlp | random | 256 | 0.2588 | 0.4886 | 0.6678 | {"gemini-3.5-flash": 8, "gpt-5.5": 1, "qwen3-14b-awq-local": 90, "qwen3-32b-awq-local": 47, "qwen3-4b-local": 6, "qwen3-8b-local": 18} |
| knn | active_low_confidence | 256 | 0.2353 | 0.4593 | 0.6277 | {"gemini-3.5-flash": 4, "gpt-5.5": 3, "qwen3-14b-awq-local": 88, "qwen3-32b-awq-local": 28, "qwen3-4b-local": 8, "qwen3-8b-local": 39} |
| knn | random | 256 | 0.2941 | 0.4310 | 0.5890 | {"gemini-3.5-flash": 4, "gpt-5.5": 4, "qwen3-14b-awq-local": 89, "qwen3-32b-awq-local": 20, "qwen3-4b-local": 37, "qwen3-8b-local": 16} |
| mlp | active_low_confidence | 492 | 0.2941 | 0.4476 | 0.6118 | {"gpt-5.5": 2, "qwen3-14b-awq-local": 114, "qwen3-32b-awq-local": 25, "qwen3-4b-local": 11, "qwen3-8b-local": 18} |
| mlp | random | 492 | 0.2588 | 0.4398 | 0.6012 | {"gemini-3.5-flash": 9, "gpt-5.5": 8, "qwen3-14b-awq-local": 81, "qwen3-32b-awq-local": 36, "qwen3-4b-local": 8, "qwen3-8b-local": 28} |
| knn | active_low_confidence | 492 | 0.2706 | 0.4239 | 0.5794 | {"gemini-3.5-flash": 2, "gpt-5.5": 4, "qwen3-14b-awq-local": 84, "qwen3-32b-awq-local": 35, "qwen3-4b-local": 27, "qwen3-8b-local": 18} |
| knn | random | 492 | 0.2706 | 0.4239 | 0.5794 | {"gemini-3.5-flash": 2, "gpt-5.5": 4, "qwen3-14b-awq-local": 84, "qwen3-32b-awq-local": 35, "qwen3-4b-local": 27, "qwen3-8b-local": 18} |

## Artifacts

- `table_v2_state_policy.csv`
- `table_v2_query_state_predictor_diagnostics.csv`
- `table_v2_state_assignments.csv`
- `table_v2_state_cards.csv`
- `table_v2_new_benchmark_policy.csv`
- `table_v2_new_benchmark_assignments.csv`
- `table_v2_active_query_state_learning.csv`

Command:

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py
```
