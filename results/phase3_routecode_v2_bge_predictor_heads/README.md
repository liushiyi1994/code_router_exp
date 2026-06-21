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

- `knn`
- `mlp`
- `torch_mlp`

Available predictor names:

- `knn`: distance-weighted KNN over query embeddings
- `mlp`: sklearn MLP over query embeddings
- `torch_mlp`: deeper PyTorch MLP over query embeddings with optional routing-aware loss
- `text_cnn`: trainable token CNN over query text with optional routing-aware loss
- `transformer`: frozen local Hugging Face encoder plus MLP head; uses `--transformer-model` and defaults to local files only

Low-confidence predictions trigger either:

- deployable fallback to the train-best model; or
- diagnostic active-state reveal, an upper bound for what active probing could recover.

## Top Test Policies

| method | quality | utility | oracle utility ratio | active/probe rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| relative_kmeans_k16_diagnostic_true_state | 0.7529 | 0.7295 | 0.9970 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 80, "qwen3-32b-awq-local": 17, "qwen3-4b-local": 20, "qwen3-8b-local": 23} |
| relative_kmeans_k4_diagnostic_true_state | 0.7118 | 0.6716 | 0.9180 | 0.0000 | {"gpt-5.5": 26, "qwen3-14b-awq-local": 87, "qwen3-4b-local": 57} |
| relative_kmeans_k8_diagnostic_true_state | 0.6765 | 0.6537 | 0.8935 | 0.0000 | {"gemini-3.5-flash": 27, "gpt-5.5": 10, "qwen3-14b-awq-local": 88, "qwen3-32b-awq-local": 25, "qwen3-4b-local": 20} |
| relative_kmeans_k2_knn_active_state_reveal | 0.6176 | 0.6065 | 0.8289 | 0.2294 | {"gemini-3.5-flash": 97, "qwen3-32b-awq-local": 73} |
| relative_kmeans_k2_diagnostic_true_state | 0.6118 | 0.5997 | 0.8196 | 0.0000 | {"gemini-3.5-flash": 96, "qwen3-32b-awq-local": 74} |
| relative_kmeans_k2_mlp_active_state_reveal | 0.6059 | 0.5916 | 0.8086 | 0.1765 | {"gemini-3.5-flash": 109, "qwen3-32b-awq-local": 61} |
| relative_kmeans_k8_mlp_active_state_reveal | 0.6000 | 0.5906 | 0.8073 | 0.4294 | {"gemini-3.5-flash": 25, "gpt-5.5": 4, "qwen3-14b-awq-local": 102, "qwen3-32b-awq-local": 27, "qwen3-4b-local": 12} |
| relative_kmeans_k2_knn_lowconf_fallback | 0.5882 | 0.5802 | 0.7931 | 0.2294 | {"gemini-3.5-flash": 81, "qwen3-32b-awq-local": 89} |
| relative_kmeans_k2_knn_plain | 0.5824 | 0.5681 | 0.7765 | 0.0000 | {"gemini-3.5-flash": 109, "qwen3-32b-awq-local": 61} |
| relative_kmeans_k2_mlp_lowconf_fallback | 0.5765 | 0.5634 | 0.7700 | 0.1765 | {"gemini-3.5-flash": 97, "qwen3-32b-awq-local": 73} |
| relative_kmeans_k2_torch_mlp_lowconf_fallback | 0.5706 | 0.5569 | 0.7612 | 0.1000 | {"gemini-3.5-flash": 97, "qwen3-32b-awq-local": 73} |
| relative_kmeans_k2_torch_mlp_active_state_reveal | 0.5706 | 0.5561 | 0.7601 | 0.1000 | {"gemini-3.5-flash": 104, "qwen3-32b-awq-local": 66} |

## Query-to-State Diagnostics

| state method | K | predictor | state accuracy | mean confidence | ECE | probe rate | covered accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| relative_kmeans | 2 | knn | 0.7235 | 0.7821 | 0.1091 | 0.2294 | 0.8244 |
| relative_kmeans | 2 | mlp | 0.6941 | 0.9305 | 0.2364 | 0.1765 | 0.7357 |
| relative_kmeans | 2 | torch_mlp | 0.6882 | 0.9738 | 0.3054 | 0.1000 | 0.6993 |
| relative_kmeans | 4 | knn | 0.5000 | 0.6040 | 0.1443 | 0.0706 | 0.5000 |
| relative_kmeans | 4 | mlp | 0.4706 | 0.6423 | 0.1782 | 0.2706 | 0.5000 |
| relative_kmeans | 4 | torch_mlp | 0.4471 | 0.9615 | 0.5145 | 0.2706 | 0.4919 |
| relative_kmeans | 8 | knn | 0.4059 | 0.4589 | 0.0953 | 0.0412 | 0.4172 |
| relative_kmeans | 8 | mlp | 0.3882 | 0.3484 | 0.0653 | 0.4294 | 0.4227 |
| relative_kmeans | 8 | torch_mlp | 0.3706 | 0.8928 | 0.5222 | 0.2412 | 0.3798 |
| relative_kmeans | 16 | knn | 0.3471 | 0.3481 | 0.0800 | 0.1941 | 0.3869 |
| relative_kmeans | 16 | mlp | 0.3176 | 0.3347 | 0.0918 | 0.2059 | 0.3259 |
| relative_kmeans | 16 | torch_mlp | 0.2529 | 0.8711 | 0.6181 | 0.2471 | 0.2578 |

## New Benchmark Smoke

| method | quality | utility | oracle utility ratio | gate rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| relative_kmeans_k8_mlp_semantic_or_lowconf_remote_gate | 0.6667 | 0.3718 | 0.6768 | 0.8667 | {"gpt-5.5": 13, "qwen3-4b-local": 2} |
| relative_kmeans_k4_knn_semantic_or_lowconf_remote_gate | 0.5333 | 0.3115 | 0.5670 | 0.6000 | {"gpt-5.5": 11, "qwen3-4b-local": 4} |
| relative_kmeans_k8_torch_mlp_semantic_or_lowconf_remote_gate | 0.5333 | 0.3094 | 0.5631 | 0.5333 | {"gpt-5.5": 10, "qwen3-4b-local": 5} |
| relative_kmeans_k4_mlp_semantic_or_lowconf_remote_gate | 0.4667 | 0.2574 | 0.4685 | 0.6667 | {"gpt-5.5": 10, "qwen3-4b-local": 5} |
| relative_kmeans_k4_torch_mlp_semantic_or_lowconf_remote_gate | 0.4667 | 0.2470 | 0.4495 | 0.6000 | {"gpt-5.5": 9, "qwen3-4b-local": 6} |
| relative_kmeans_k2_knn_semantic_or_lowconf_remote_gate | 0.4000 | 0.2373 | 0.4319 | 0.4667 | {"gpt-5.5": 7, "qwen3-4b-local": 8} |
| relative_kmeans_k16_torch_mlp_semantic_or_lowconf_remote_gate | 0.4667 | 0.2040 | 0.3714 | 0.4000 | {"gpt-5.5": 10, "qwen3-4b-local": 5} |
| relative_kmeans_k16_knn_semantic_or_lowconf_remote_gate | 0.3333 | 0.1913 | 0.3482 | 0.3333 | {"gpt-5.5": 7, "qwen3-4b-local": 8} |
| relative_kmeans_k2_mlp_semantic_or_lowconf_remote_gate | 0.4000 | 0.1807 | 0.3290 | 0.5333 | {"gpt-5.5": 9, "qwen3-4b-local": 6} |
| relative_kmeans_k8_knn_semantic_or_lowconf_remote_gate | 0.2667 | 0.1339 | 0.2436 | 0.3333 | {"gpt-5.5": 6, "qwen3-4b-local": 9} |
| relative_kmeans_k16_mlp_semantic_or_lowconf_remote_gate | 0.4000 | 0.1022 | 0.1861 | 0.6667 | {"gpt-5.5": 10, "qwen3-4b-local": 5} |
| relative_kmeans_k4_mlp_plain_predicted_state | 0.2000 | 0.0921 | 0.1677 | 0.3333 | {"gpt-5.5": 5, "qwen3-4b-local": 10} |

## Active Query-State Labeling Simulation

This is train-only active learning for the query-to-state classifier. It starts
with one labeled query per state, then either samples random additional state
labels or samples low-confidence training queries. Evaluation is on held-out
Broad100 test queries.

| predictor | strategy | labeled queries | state accuracy | utility | oracle utility ratio | selected models |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| mlp | active_low_confidence | 64 | 0.7235 | 0.5512 | 0.7534 | {"gemini-3.5-flash": 75, "qwen3-32b-awq-local": 95} |
| mlp | random | 64 | 0.6412 | 0.5411 | 0.7395 | {"gemini-3.5-flash": 77, "qwen3-32b-awq-local": 93} |
| torch_mlp | active_low_confidence | 64 | 0.7059 | 0.5307 | 0.7253 | {"gemini-3.5-flash": 92, "qwen3-32b-awq-local": 78} |
| knn | random | 64 | 0.6824 | 0.5227 | 0.7144 | {"gemini-3.5-flash": 124, "qwen3-32b-awq-local": 46} |
| torch_mlp | random | 64 | 0.6294 | 0.5067 | 0.6925 | {"gemini-3.5-flash": 113, "qwen3-32b-awq-local": 57} |
| knn | active_low_confidence | 64 | 0.4824 | 0.4833 | 0.6606 | {"gemini-3.5-flash": 36, "qwen3-32b-awq-local": 134} |

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
