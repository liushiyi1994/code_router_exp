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

- `transformer`

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
| relative_kmeans_k16_transformer_active_state_reveal | 0.5471 | 0.5296 | 0.7239 | 0.2412 | {"gemini-3.5-flash": 12, "gpt-5.5": 8, "qwen3-14b-awq-local": 67, "qwen3-32b-awq-local": 44, "qwen3-4b-local": 18, "qwen3-8b-local": 21} |
| relative_kmeans_k16_transformer_lowconf_fallback | 0.4824 | 0.4670 | 0.6383 | 0.2412 | {"gemini-3.5-flash": 7, "gpt-5.5": 7, "qwen3-14b-awq-local": 50, "qwen3-32b-awq-local": 81, "qwen3-4b-local": 10, "qwen3-8b-local": 15} |
| relative_kmeans_k16_transformer_plain | 0.4765 | 0.4485 | 0.6130 | 0.0000 | {"gemini-3.5-flash": 10, "gpt-5.5": 11, "qwen3-14b-awq-local": 64, "qwen3-32b-awq-local": 48, "qwen3-4b-local": 16, "qwen3-8b-local": 21} |

## Query-to-State Diagnostics

| state method | K | predictor | state accuracy | mean confidence | ECE | probe rate | covered accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| relative_kmeans | 16 | transformer | 0.2588 | 0.8861 | 0.6272 | 0.2412 | 0.2636 |

## New Benchmark Smoke

| method | quality | utility | oracle utility ratio | gate rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| relative_kmeans_k16_transformer_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 1.0000 | {"gpt-5.5": 15} |
| relative_kmeans_k16_transformer_plain_predicted_state | 0.1333 | 0.1089 | 0.1982 | 0.6000 | {"gpt-5.5": 2, "qwen3-4b-local": 13} |

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
