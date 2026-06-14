# Phase E RouteLLM MF Asset Memo

Command: `python experiments/15_routellm_mf_assets.py --config configs/llmrouterbench_pilot.yaml`

Artifact directory: `results/llmrouterbench_pilot/routellm_mf_assets`.

Binary pair: strong/model_a `Qwen3-8B`, weak/model_b `Qwen2.5-Coder-7B-Instruct`.

These assets are ready for the local LLMRouterBench RouteLLM MF trainer: they include `idx`, score/cost fields, `prompt_embeddings.npy`, and a local CPU training config.

This is not a trained RouteLLM MF result. The next step is to run the local MF trainer and evaluate the checkpoint on the RouteCode test split.

## Compatibility

- `split_aligned_with_routecode`: `True`
- Train/test query overlap: `0`
- `official_trainer_compatible`: `True`
- `official_routellm_result`: `False`
- Pair present in official `MODEL_IDS`: `True`
- Local training config: `results/llmrouterbench_pilot/routellm_mf_assets/mf_train_config.local.json`

## Asset Summary

| split | record_count | decisive_count | tie_count | model_a_quality_win_count | model_b_quality_win_count | model_a_quality_win_rate | model_b_quality_win_rate | quality_tie_rate | model_a_utility_win_count | model_b_utility_win_count | utility_tie_count | mean_utility_margin_model_a_minus_b | strong_model | weak_model | split_aligned_with_routecode | official_trainer_compatible | official_routellm_result | routecode_metric_compatible | implementation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 627 | 627 | 0 | 393 | 234 | 0.6268 | 0.3732 | 0.0000 | 393 | 234 | 0 | 0.2536 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | True | False | False | Official-trainer-compatible RouteLLM MF assets with local RouteCode embeddings; not a trained RouteLLM MF result. |
| test | 580 | 206 | 374 | 120 | 86 | 0.2069 | 0.1483 | 0.6448 | 120 | 86 | 374 | 0.0586 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | True | False | False | Official-trainer-compatible RouteLLM MF assets with local RouteCode embeddings; not a trained RouteLLM MF result. |
| overall | 1207 | 833 | 374 | 513 | 320 | 0.4250 | 0.2651 | 0.3099 | 513 | 320 | 374 | 0.1599 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | True | False | False | Official-trainer-compatible RouteLLM MF assets with local RouteCode embeddings; not a trained RouteLLM MF result. |

## Remaining External-Baseline Gap

- Run the LLMRouterBench RouteLLM MF trainer on `mf_train_config.local.json`.
- Evaluate the resulting checkpoint on `pairwise_test.json` and convert selections back to RouteCode utility metrics.
- Keep BERT, GraphRouter, and Avengers/Avengers-Pro as separate adapter tasks.
