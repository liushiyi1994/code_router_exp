# Phase E External Baseline Surrogate Memo

Command: `python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_pilot.yaml`

Binary pair: strong `Qwen3-8B`, weak `Qwen2.5-Coder-7B-Instruct`.

These rows are local surrogates inspired by RouteLLM/LLMRouter-style baselines. They are not official external-repo reproductions.

Official RouteLLM/LLMRouterBench RouteLLM inspection found that the upstream adapter expects its own embedding/checkpoint pipeline; this run keeps the no-API local RouteCode split and deterministic embeddings.

| method | baseline_family | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- |
| query_oracle | reference | 0.8966 | 0.0000 | 1.0000 |
| kNN | reference | 0.7362 | 0.1603 | 0.3008 |
| routellm_style_mf_utility_router | external_style_surrogate | 0.7052 | 0.1914 | 0.1654 |
| routellm_binary_logistic_surrogate_t0.25 | external_style_surrogate | 0.6931 | 0.2034 | 0.1128 |
| best_single | reference | 0.6672 | 0.2293 | 0.0000 |
| routellm_pair_strong_only | binary_pair_reference | 0.6672 | 0.2293 | 0.0000 |
| routellm_binary_logistic_surrogate_t0.5 | external_style_surrogate | 0.6552 | 0.2414 | -0.0526 |
| routellm_binary_logistic_surrogate_t0.75 | external_style_surrogate | 0.6241 | 0.2724 | -0.1880 |
| routellm_pair_weak_only | binary_pair_reference | 0.6086 | 0.2879 | -0.2556 |

## References Used

- RouteLLM paper/repo: https://arxiv.org/abs/2406.18665 ; https://github.com/lm-sys/routellm
- LLMRouter repo: https://github.com/ulab-uiuc/LLMRouter
- LLMRouterBench paper/repo: https://arxiv.org/abs/2601.07206 ; https://github.com/ynulihao/LLMRouterBench

## Remaining External-Baseline Gap

- Run an official RouteLLM-MF/BERT baseline or an LLMRouterBench adapter output when its dependency and embedding pipeline can be pinned locally.
- Run GraphRouter/Avengers-Pro only after their commands, data contracts, and leakage controls are pinned.
