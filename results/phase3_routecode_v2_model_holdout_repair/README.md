# RouteCode V2 State Pipeline

This run implements the split proposed in the Phase 3 direction change:

```text
Model 1: utility matrix -> latent states + state-to-model table
Model 2: query text embedding -> p(state | query)
```

State learner variants:

- `raw_kmeans`
- `relative_kmeans`
- `calibration_refined`
- `model_holdout_repaired`

`model_holdout_repaired` uses relative routing features plus raw and centered
utility columns, then splits states with high model-holdout variance/error.
Model-holdout repair was merged back to the requested K budget.

Query-to-state predictors:

- `knn`

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
| model_holdout_repaired_k24_diagnostic_true_state | 0.7529 | 0.7316 | 1.0000 | 0.0000 | {"gemini-3.5-flash": 10, "gpt-5.5": 10, "qwen3-14b-awq-local": 110, "qwen3-32b-awq-local": 20, "qwen3-4b-local": 16, "qwen3-8b-local": 4} |
| relative_kmeans_k16_diagnostic_true_state | 0.7529 | 0.7295 | 0.9970 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 80, "qwen3-32b-awq-local": 17, "qwen3-4b-local": 20, "qwen3-8b-local": 23} |
| calibration_refined_k16_diagnostic_true_state | 0.7529 | 0.7295 | 0.9970 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 80, "qwen3-32b-awq-local": 17, "qwen3-4b-local": 20, "qwen3-8b-local": 23} |
| relative_kmeans_k24_diagnostic_true_state | 0.7412 | 0.7177 | 0.9809 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 90, "qwen3-32b-awq-local": 24, "qwen3-4b-local": 15, "qwen3-8b-local": 11} |
| calibration_refined_k24_diagnostic_true_state | 0.7412 | 0.7177 | 0.9809 | 0.0000 | {"gemini-3.5-flash": 20, "gpt-5.5": 10, "qwen3-14b-awq-local": 90, "qwen3-32b-awq-local": 24, "qwen3-4b-local": 15, "qwen3-8b-local": 11} |
| raw_kmeans_k24_diagnostic_true_state | 0.7353 | 0.7138 | 0.9757 | 0.0000 | {"gemini-3.5-flash": 12, "gpt-5.5": 10, "qwen3-14b-awq-local": 70, "qwen3-32b-awq-local": 12, "qwen3-4b-local": 55, "qwen3-8b-local": 11} |
| model_holdout_repaired_k16_diagnostic_true_state | 0.7176 | 0.6963 | 0.9518 | 0.0000 | {"gemini-3.5-flash": 10, "gpt-5.5": 10, "qwen3-14b-awq-local": 114, "qwen3-32b-awq-local": 17, "qwen3-4b-local": 15, "qwen3-8b-local": 4} |
| raw_kmeans_k16_diagnostic_true_state | 0.7176 | 0.6962 | 0.9516 | 0.0000 | {"gemini-3.5-flash": 12, "gpt-5.5": 10, "qwen3-14b-awq-local": 58, "qwen3-32b-awq-local": 22, "qwen3-4b-local": 14, "qwen3-8b-local": 54} |
| model_holdout_repaired_k16_knn_active_state_reveal | 0.5882 | 0.5687 | 0.7772 | 0.2824 | {"gemini-3.5-flash": 3, "gpt-5.5": 9, "qwen3-14b-awq-local": 119, "qwen3-32b-awq-local": 31, "qwen3-4b-local": 8} |
| relative_kmeans_k24_knn_active_state_reveal | 0.5353 | 0.5178 | 0.7077 | 0.2235 | {"gemini-3.5-flash": 8, "gpt-5.5": 7, "qwen3-14b-awq-local": 111, "qwen3-32b-awq-local": 13, "qwen3-4b-local": 26, "qwen3-8b-local": 5} |
| calibration_refined_k24_knn_active_state_reveal | 0.5353 | 0.5178 | 0.7077 | 0.2235 | {"gemini-3.5-flash": 8, "gpt-5.5": 7, "qwen3-14b-awq-local": 111, "qwen3-32b-awq-local": 13, "qwen3-4b-local": 26, "qwen3-8b-local": 5} |
| relative_kmeans_k16_knn_active_state_reveal | 0.5294 | 0.5121 | 0.7000 | 0.1941 | {"gemini-3.5-flash": 5, "gpt-5.5": 7, "qwen3-14b-awq-local": 85, "qwen3-32b-awq-local": 36, "qwen3-4b-local": 20, "qwen3-8b-local": 17} |

## Query-to-State Diagnostics

| state method | K | predictor | state accuracy | mean confidence | ECE | probe rate | covered accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| model_holdout_repaired | 16 | knn | 0.3176 | 0.4517 | 0.1702 | 0.2824 | 0.3689 |
| raw_kmeans | 16 | knn | 0.3176 | 0.4196 | 0.1962 | 0.2118 | 0.3731 |
| raw_kmeans | 24 | knn | 0.3118 | 0.3953 | 0.1708 | 0.2294 | 0.3588 |
| model_holdout_repaired | 24 | knn | 0.2824 | 0.4329 | 0.1985 | 0.1000 | 0.2941 |
| relative_kmeans | 16 | knn | 0.2706 | 0.4033 | 0.1702 | 0.1941 | 0.3212 |
| calibration_refined | 16 | knn | 0.2706 | 0.4033 | 0.1702 | 0.1941 | 0.3212 |
| calibration_refined | 24 | knn | 0.2647 | 0.3957 | 0.1796 | 0.2235 | 0.3182 |
| relative_kmeans | 24 | knn | 0.2647 | 0.3957 | 0.1796 | 0.2235 | 0.3182 |

## New Benchmark Smoke

| method | quality | utility | oracle utility ratio | gate rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| relative_kmeans_k16_knn_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 0.8667 | {"gpt-5.5": 15} |
| calibration_refined_k16_knn_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 0.8667 | {"gpt-5.5": 15} |
| model_holdout_repaired_k16_knn_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 1.0000 | {"gpt-5.5": 15} |
| raw_kmeans_k24_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| relative_kmeans_k24_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| calibration_refined_k24_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| model_holdout_repaired_k24_knn_semantic_or_lowconf_remote_gate | 0.6667 | 0.3319 | 0.6041 | 0.9333 | {"gpt-5.5": 14, "qwen3-4b-local": 1} |
| raw_kmeans_k16_knn_semantic_or_lowconf_remote_gate | 0.6000 | 0.2712 | 0.4937 | 0.8667 | {"gpt-5.5": 13, "qwen3-4b-local": 2} |
| relative_kmeans_k16_knn_plain_predicted_state | 0.3333 | 0.2702 | 0.4919 | 0.2000 | {"gpt-5.5": 5, "qwen3-4b-local": 10} |
| calibration_refined_k16_knn_plain_predicted_state | 0.3333 | 0.2702 | 0.4919 | 0.2000 | {"gpt-5.5": 5, "qwen3-4b-local": 10} |
| raw_kmeans_k24_knn_plain_predicted_state | 0.1333 | 0.0840 | 0.1529 | 0.3333 | {"gpt-5.5": 2, "qwen3-4b-local": 13} |
| relative_kmeans_k24_knn_plain_predicted_state | 0.0667 | 0.0607 | 0.1104 | 0.4000 | {"gpt-5.5": 1, "qwen3-4b-local": 14} |

## Active Query-State Labeling Simulation

This is train-only active learning for the query-to-state classifier. It starts
with one labeled query per state, then either samples random additional state
labels or samples low-confidence training queries. Evaluation is on held-out
Broad100 test queries.

| predictor | strategy | labeled queries | state accuracy | utility | oracle utility ratio | selected models |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| knn | active_low_confidence | 64 | 0.2294 | 0.5515 | 0.7538 | {"gemini-3.5-flash": 18, "gpt-5.5": 19, "qwen3-14b-awq-local": 54, "qwen3-32b-awq-local": 66, "qwen3-4b-local": 1, "qwen3-8b-local": 12} |
| knn | random | 64 | 0.2824 | 0.4111 | 0.5619 | {"gemini-3.5-flash": 13, "qwen3-14b-awq-local": 105, "qwen3-32b-awq-local": 9, "qwen3-4b-local": 34, "qwen3-8b-local": 9} |

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
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py --output-dir results/phase3_routecode_v2_model_holdout_repair --state-methods raw_kmeans relative_kmeans calibration_refined model_holdout_repaired --k-values 16 24 --predictors knn --active-label-budgets 64 --model-holdout-variance-threshold 0.025 --model-holdout-error-threshold 0.10 --model-holdout-min-state-size 8 --model-holdout-max-split-fraction 1.0
```
