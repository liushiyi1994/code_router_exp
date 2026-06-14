# Phase E RouteLLM MF Split-Aligned Memo

Command: `python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_pilot.yaml`

Artifact directory: `results/llmrouterbench_pilot/routellm_mf_split_aligned`.

This run trains the local LLMRouterBench RouteLLM MF model class on the RouteCode split-aligned pairwise assets and evaluates the checkpoint on the RouteCode test split.

It is not the upstream published RouteLLM checkpoint and it uses deterministic local RouteCode embeddings rather than an API-backed embedding generator.

## Training Summary

- Validation records: `206`
- Validation accuracy: `0.7476`
- Validation loss: `0.5761`

## Evaluation Summary

| mean_utility | oracle_regret | mean_quality | normalized_cost | method | K | utility_ci_low | utility_ci_high | recovered_gap_vs_learned | recovered_gap_vs_oracle | selected_model_entropy | rate_log2K | empirical_H_Z | threshold | strong_model | weak_model | selection_accuracy | routing_accuracy_decisive | decisive_count | tie_count | strong_selection_rate | weak_selection_rate | mean_strong_win_rate | train_loss | validation_accuracy | official_training_code_used | official_upstream_checkpoint | split_aligned_with_routecode | routecode_metric_compatible | baseline_family | implementation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6672 | 0.2293 | 0.6672 | 0.1614 | routellm_mf_split_aligned_t0.25 | 2 | 0.6259 | 0.7044 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2500 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | 0.6672 | 0.5825 | 206 | 374 | 1.0000 | 0.0000 | 0.6062 | None | 0.7476 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |
| 0.7259 | 0.1707 | 0.7259 | 0.1164 | routellm_mf_split_aligned_t0.5 | 2 | 0.6897 | 0.7586 | 0.8500 | 0.2556 | 0.7857 | 1.0000 | 0.7857 | 0.5000 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | 0.7259 | 0.7476 | 206 | 374 | 0.7655 | 0.2345 | 0.6062 | None | 0.7476 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |
| 0.6121 | 0.2845 | 0.6121 | 0.0070 | routellm_mf_split_aligned_t0.75 | 2 | 0.5715 | 0.6448 | -0.8000 | -0.2406 | 0.1732 | 1.0000 | 0.1732 | 0.7500 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | 0.6121 | 0.4272 | 206 | 374 | 0.0259 | 0.9741 | 0.6062 | None | 0.7476 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |

## Remaining External-Baseline Gap

- Add BERT, GraphRouter, and Avengers/Avengers-Pro adapter outputs if local dependencies can be pinned.
- Decide whether to install the full LLMRouterBench baseline environment for exact upstream command execution.
