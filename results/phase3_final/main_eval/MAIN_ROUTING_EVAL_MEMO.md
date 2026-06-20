# Final Main Routing Evaluation

This is a cache-backed Broad100 final-evaluation pass. It makes no provider, vLLM, or local generation calls.

## Current Method

- Method: `et_flip_leaf4_thr0.8502_capNone`
- Mean quality: `0.8547`
- Mean utility: `0.8238`
- Quality gap to oracle: `0.0174`
- Oracle utility ratio: `0.9735`
- Frontier-call rate: `0.1919`

## Oracle

- Oracle mean quality: `0.8721`
- Oracle mean utility: `0.8463`

## Literature Baseline Status

- `routellm_mf`: `broad100_adapter_executed`. Official RouteLLM is two-model routing; this cached adapter routes between train-best strong and train-best local weak action.
- `graphrouter`: `fallback_broad100_adapter_executed`. GraphRouter native cached-Broad100 adapter not available; LLMRouter kNN fallback was run per fallback policy.
- `avengerspro`: `broad100_adapter_executed`. Cached no-API implementation of the released cluster-routing contract.

## Caveat

The three literature baselines are not yet included in this Broad100 final table. Existing LLMRouterBench broad20 baseline artifacts are documented separately, but a final split-aligned adapter pass is still required.
