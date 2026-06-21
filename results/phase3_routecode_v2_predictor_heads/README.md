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

- `torch_mlp`
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
| relative_kmeans_k16_text_cnn_active_state_reveal | 0.5588 | 0.5437 | 0.7431 | 0.2706 | {"gemini-3.5-flash": 21, "gpt-5.5": 6, "qwen3-14b-awq-local": 72, "qwen3-32b-awq-local": 34, "qwen3-4b-local": 21, "qwen3-8b-local": 16} |
| relative_kmeans_k16_torch_mlp_active_state_reveal | 0.5706 | 0.5319 | 0.7270 | 0.2529 | {"gemini-3.5-flash": 6, "gpt-5.5": 14, "qwen3-14b-awq-local": 76, "qwen3-32b-awq-local": 32, "qwen3-4b-local": 16, "qwen3-8b-local": 26} |
| relative_kmeans_k16_text_cnn_plain | 0.4765 | 0.4708 | 0.6434 | 0.0000 | {"gemini-3.5-flash": 26, "gpt-5.5": 2, "qwen3-14b-awq-local": 74, "qwen3-32b-awq-local": 31, "qwen3-4b-local": 21, "qwen3-8b-local": 16} |
| relative_kmeans_k16_text_cnn_lowconf_fallback | 0.4647 | 0.4628 | 0.6326 | 0.2706 | {"gemini-3.5-flash": 15, "qwen3-14b-awq-local": 55, "qwen3-32b-awq-local": 75, "qwen3-4b-local": 12, "qwen3-8b-local": 13} |
| relative_kmeans_k16_torch_mlp_plain | 0.4941 | 0.4613 | 0.6305 | 0.0000 | {"gemini-3.5-flash": 7, "gpt-5.5": 11, "qwen3-14b-awq-local": 81, "qwen3-32b-awq-local": 30, "qwen3-4b-local": 20, "qwen3-8b-local": 21} |
| relative_kmeans_k16_torch_mlp_lowconf_fallback | 0.4824 | 0.4518 | 0.6175 | 0.2529 | {"gpt-5.5": 10, "qwen3-14b-awq-local": 64, "qwen3-32b-awq-local": 69, "qwen3-4b-local": 12, "qwen3-8b-local": 15} |

## Query-to-State Diagnostics

| state method | K | predictor | state accuracy | mean confidence | ECE | probe rate | covered accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| relative_kmeans | 16 | text_cnn | 0.3118 | 0.6418 | 0.3300 | 0.2706 | 0.3306 |
| relative_kmeans | 16 | torch_mlp | 0.2000 | 0.8851 | 0.6851 | 0.2529 | 0.2126 |

## New Benchmark Smoke

| method | quality | utility | oracle utility ratio | gate rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| relative_kmeans_k16_torch_mlp_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 0.9333 | {"gpt-5.5": 15} |
| relative_kmeans_k16_text_cnn_semantic_or_lowconf_remote_gate | 0.6000 | 0.2744 | 0.4995 | 0.8667 | {"gpt-5.5": 13, "qwen3-4b-local": 2} |
| relative_kmeans_k16_torch_mlp_plain_predicted_state | 0.4000 | 0.1438 | 0.2617 | 0.4667 | {"gpt-5.5": 9, "qwen3-4b-local": 6} |
| relative_kmeans_k16_text_cnn_plain_predicted_state | 0.0000 | 0.0000 | 0.0000 | 0.5333 | {"qwen3-4b-local": 15} |

## Active Query-State Labeling Simulation

This is train-only active learning for the query-to-state classifier. It starts
with one labeled query per state, then either samples random additional state
labels or samples low-confidence training queries. Evaluation is on held-out
Broad100 test queries.

| predictor | strategy | labeled queries | state accuracy | utility | oracle utility ratio | selected models |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| torch_mlp | random | 64 | 0.1882 | 0.4627 | 0.6324 | {"gemini-3.5-flash": 13, "gpt-5.5": 13, "qwen3-14b-awq-local": 90, "qwen3-32b-awq-local": 18, "qwen3-4b-local": 25, "qwen3-8b-local": 11} |
| torch_mlp | active_low_confidence | 64 | 0.1941 | 0.4357 | 0.5955 | {"gemini-3.5-flash": 32, "gpt-5.5": 4, "qwen3-14b-awq-local": 61, "qwen3-32b-awq-local": 41, "qwen3-4b-local": 14, "qwen3-8b-local": 18} |
| torch_mlp | active_low_confidence | 128 | 0.2235 | 0.4679 | 0.6395 | {"gemini-3.5-flash": 6, "gpt-5.5": 21, "qwen3-14b-awq-local": 66, "qwen3-32b-awq-local": 41, "qwen3-4b-local": 14, "qwen3-8b-local": 22} |
| torch_mlp | random | 128 | 0.2118 | 0.4296 | 0.5872 | {"gemini-3.5-flash": 22, "gpt-5.5": 16, "qwen3-14b-awq-local": 103, "qwen3-32b-awq-local": 12, "qwen3-4b-local": 4, "qwen3-8b-local": 13} |
| torch_mlp | random | 256 | 0.2176 | 0.4674 | 0.6388 | {"gemini-3.5-flash": 15, "gpt-5.5": 12, "qwen3-14b-awq-local": 48, "qwen3-32b-awq-local": 47, "qwen3-4b-local": 15, "qwen3-8b-local": 33} |
| torch_mlp | active_low_confidence | 256 | 0.2765 | 0.4556 | 0.6227 | {"gemini-3.5-flash": 11, "gpt-5.5": 18, "qwen3-14b-awq-local": 67, "qwen3-32b-awq-local": 31, "qwen3-4b-local": 9, "qwen3-8b-local": 34} |

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
