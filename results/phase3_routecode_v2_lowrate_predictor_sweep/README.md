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

- `text_cnn`

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
| relative_kmeans_k2_text_cnn_active_state_reveal | 0.6176 | 0.6028 | 0.8239 | 0.2176 | {"gemini-3.5-flash": 110, "qwen3-32b-awq-local": 60} |
| relative_kmeans_k2_text_cnn_lowconf_fallback | 0.6118 | 0.6003 | 0.8205 | 0.2176 | {"gemini-3.5-flash": 90, "qwen3-32b-awq-local": 80} |
| relative_kmeans_k2_diagnostic_true_state | 0.6118 | 0.5997 | 0.8196 | 0.0000 | {"gemini-3.5-flash": 96, "qwen3-32b-awq-local": 74} |
| relative_kmeans_k2_text_cnn_plain | 0.6000 | 0.5856 | 0.8004 | 0.0000 | {"gemini-3.5-flash": 108, "qwen3-32b-awq-local": 62} |
| relative_kmeans_k8_text_cnn_active_state_reveal | 0.5824 | 0.5645 | 0.7715 | 0.2588 | {"gemini-3.5-flash": 32, "gpt-5.5": 7, "qwen3-14b-awq-local": 76, "qwen3-32b-awq-local": 34, "qwen3-4b-local": 21} |
| relative_kmeans_k4_text_cnn_active_state_reveal | 0.5882 | 0.5594 | 0.7646 | 0.2471 | {"gpt-5.5": 17, "qwen3-14b-awq-local": 98, "qwen3-4b-local": 55} |
| relative_kmeans_k16_text_cnn_active_state_reveal | 0.5647 | 0.5541 | 0.7573 | 0.2471 | {"gemini-3.5-flash": 3, "gpt-5.5": 5, "qwen3-14b-awq-local": 87, "qwen3-32b-awq-local": 36, "qwen3-4b-local": 14, "qwen3-8b-local": 25} |
| relative_kmeans_k4_text_cnn_lowconf_fallback | 0.5353 | 0.5237 | 0.7158 | 0.2471 | {"gpt-5.5": 7, "qwen3-14b-awq-local": 74, "qwen3-32b-awq-local": 42, "qwen3-4b-local": 47} |
| relative_kmeans_k8_text_cnn_lowconf_fallback | 0.5294 | 0.5231 | 0.7149 | 0.2588 | {"gemini-3.5-flash": 26, "gpt-5.5": 2, "qwen3-14b-awq-local": 57, "qwen3-32b-awq-local": 66, "qwen3-4b-local": 19} |

## Query-to-State Diagnostics

| state method | K | predictor | state accuracy | mean confidence | ECE | probe rate | covered accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| relative_kmeans | 2 | text_cnn | 0.7176 | 0.9400 | 0.2224 | 0.2176 | 0.7744 |
| relative_kmeans | 4 | text_cnn | 0.5471 | 0.8346 | 0.2956 | 0.2471 | 0.5781 |
| relative_kmeans | 8 | text_cnn | 0.4353 | 0.7365 | 0.3012 | 0.2588 | 0.4841 |
| relative_kmeans | 16 | text_cnn | 0.3353 | 0.6721 | 0.3394 | 0.2471 | 0.3672 |

## New Benchmark Smoke

| method | quality | utility | oracle utility ratio | gate rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| relative_kmeans_k4_text_cnn_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 1.0000 | {"gpt-5.5": 15} |
| relative_kmeans_k8_text_cnn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3259 | 0.5932 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| relative_kmeans_k16_text_cnn_semantic_or_lowconf_remote_gate | 0.6000 | 0.2744 | 0.4995 | 0.8667 | {"gpt-5.5": 13, "qwen3-4b-local": 2} |
| relative_kmeans_k4_text_cnn_plain_predicted_state | 0.4000 | 0.1730 | 0.3149 | 1.0000 | {"gpt-5.5": 9, "qwen3-4b-local": 6} |
| relative_kmeans_k8_text_cnn_plain_predicted_state | 0.2000 | 0.1598 | 0.2909 | 0.7333 | {"gpt-5.5": 3, "qwen3-4b-local": 12} |
| relative_kmeans_k2_text_cnn_semantic_or_lowconf_remote_gate | 0.4667 | 0.1597 | 0.2908 | 0.7333 | {"gpt-5.5": 11, "qwen3-4b-local": 4} |
| relative_kmeans_k2_text_cnn_plain_predicted_state | 0.0000 | 0.0000 | 0.0000 | 0.6000 | {"qwen3-4b-local": 15} |
| relative_kmeans_k16_text_cnn_plain_predicted_state | 0.0000 | 0.0000 | 0.0000 | 0.4667 | {"qwen3-4b-local": 15} |

## Active Query-State Labeling Simulation

This is train-only active learning for the query-to-state classifier. It starts
with one labeled query per state, then either samples random additional state
labels or samples low-confidence training queries. Evaluation is on held-out
Broad100 test queries.

| predictor | strategy | labeled queries | state accuracy | utility | oracle utility ratio | selected models |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| not run | | | | | | |

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
