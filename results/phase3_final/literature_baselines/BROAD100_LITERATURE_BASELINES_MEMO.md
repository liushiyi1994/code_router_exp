# Broad100 Literature Baseline Adapters

This run evaluates the three selected literature-baseline slots on the final cached Broad100 matrix.
No provider, vLLM, or external embedding calls are made.

## Input

- Outcome matrix: `results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet`

## Results

- `routellm_pairwise_mf_adapter`: utility `0.6076`, quality `0.7326`, frontier rate `0.5814`
- `llmrouter_knn_fallback_k63`: utility `0.6061`, quality `0.7616`, frontier rate `0.7674`
- `avengerspro_cluster_adapter_k32`: utility `0.5899`, quality `0.7209`, frontier rate `0.6221`

## Adapter Status

- `routellm_mf`: `broad100_adapter_executed`; commit `0b64fdafe049e596a3f5657c219329f24af24198`; Official RouteLLM is two-model routing; this cached adapter routes between train-best strong and train-best local weak action.
- `graphrouter`: `fallback_broad100_adapter_executed`; commit `c65a32b1435bacdb1488280effef28a6ff89edf6`; GraphRouter native cached-Broad100 adapter not available; LLMRouter kNN fallback was run per fallback policy.
- `avengerspro`: `broad100_adapter_executed`; commit `c77cb0506949d8f959e97967d2fefca0e8ff1b05`; Cached no-API implementation of the released cluster-routing contract.

## Caveat

These are cached Broad100 adapters. RouteLLM is represented as a two-model pairwise adapter; GraphRouter uses the documented LLMRouter kNN fallback because a native cached-Broad100 GraphRouter adapter is not available.
