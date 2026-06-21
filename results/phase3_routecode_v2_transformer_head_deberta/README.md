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
| relative_kmeans_k16_transformer_active_state_reveal | 0.5412 | 0.5051 | 0.6904 | 0.2706 | {"gemini-3.5-flash": 8, "gpt-5.5": 14, "qwen3-14b-awq-local": 73, "qwen3-32b-awq-local": 43, "qwen3-4b-local": 13, "qwen3-8b-local": 19} |
| relative_kmeans_k16_transformer_lowconf_fallback | 0.4824 | 0.4509 | 0.6163 | 0.2706 | {"gemini-3.5-flash": 6, "gpt-5.5": 12, "qwen3-14b-awq-local": 50, "qwen3-32b-awq-local": 83, "qwen3-4b-local": 5, "qwen3-8b-local": 14} |
| relative_kmeans_k16_transformer_plain | 0.4824 | 0.4353 | 0.5949 | 0.0000 | {"gemini-3.5-flash": 12, "gpt-5.5": 18, "qwen3-14b-awq-local": 68, "qwen3-32b-awq-local": 41, "qwen3-4b-local": 10, "qwen3-8b-local": 21} |

## Query-to-State Diagnostics

| state method | K | predictor | state accuracy | mean confidence | ECE | probe rate | covered accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| relative_kmeans | 16 | transformer | 0.2059 | 0.8804 | 0.6745 | 0.2706 | 0.2177 |

## New Benchmark Smoke

| method | quality | utility | oracle utility ratio | gate rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
| relative_kmeans_k16_transformer_semantic_or_lowconf_remote_gate | 0.7333 | 0.3833 | 0.6977 | 1.0000 | {"gpt-5.5": 15} |
| relative_kmeans_k16_transformer_plain_predicted_state | 0.2667 | 0.1610 | 0.2930 | 0.4667 | {"gpt-5.5": 5, "qwen3-4b-local": 10} |

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
