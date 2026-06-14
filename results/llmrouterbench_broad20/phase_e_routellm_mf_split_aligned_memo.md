# Phase E RouteLLM MF Split-Aligned Memo

Command: `python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_broad20.yaml`

Artifact directory: `results/llmrouterbench_broad20/routellm_mf_split_aligned`.

This run trains the local LLMRouterBench RouteLLM MF model class on the RouteCode split-aligned pairwise assets and evaluates the checkpoint on the RouteCode test split.

It is not the upstream published RouteLLM checkpoint and it uses deterministic local RouteCode embeddings rather than an API-backed embedding generator.

## Training Summary

- Validation records: `1230`
- Validation accuracy: `0.9106`
- Validation loss: `0.2512`

## Evaluation Summary

| mean_utility | oracle_regret | mean_quality | normalized_cost | method | K | utility_ci_low | utility_ci_high | recovered_gap_vs_learned | recovered_gap_vs_oracle | selected_model_entropy | rate_log2K | empirical_H_Z | threshold | strong_model | weak_model | selection_accuracy | routing_accuracy_decisive | decisive_count | tie_count | strong_selection_rate | weak_selection_rate | mean_strong_win_rate | train_loss | validation_accuracy | official_training_code_used | official_upstream_checkpoint | split_aligned_with_routecode | routecode_metric_compatible | baseline_family | implementation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7041 | 0.2119 | 0.7041 | 0.0973 | routellm_mf_split_aligned_t0.25 | 2 | 0.6880 | 0.7206 | 0.0000 | 0.0017 | 0.0046 | 1.0000 | 0.0046 | 0.2500 | Qwen3-8B | MiMo-7B-RL-0530 | 0.7041 | 0.9089 | 1230 | 1578 | 0.9996 | 0.0004 | 0.9067 | None | 0.9106 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |
| 0.7073 | 0.2087 | 0.7073 | 0.0959 | routellm_mf_split_aligned_t0.5 | 2 | 0.6910 | 0.7222 | 0.0000 | 0.0168 | 0.0945 | 1.0000 | 0.0945 | 0.5000 | Qwen3-8B | MiMo-7B-RL-0530 | 0.7073 | 0.9163 | 1230 | 1578 | 0.9879 | 0.0121 | 0.9067 | None | 0.9106 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |
| 0.7001 | 0.2158 | 0.7001 | 0.0888 | routellm_mf_split_aligned_t0.75 | 2 | 0.6845 | 0.7140 | 0.0000 | -0.0168 | 0.4138 | 1.0000 | 0.4138 | 0.7500 | Qwen3-8B | MiMo-7B-RL-0530 | 0.7001 | 0.9000 | 1230 | 1578 | 0.9167 | 0.0833 | 0.9067 | None | 0.9106 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |

## Remaining External-Baseline Gap

- Add BERT, GraphRouter, and Avengers/Avengers-Pro adapter outputs if local dependencies can be pinned.
- Decide whether to install the full LLMRouterBench baseline environment for exact upstream command execution.
