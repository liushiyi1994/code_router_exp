# Phase E External Baseline Surrogate Memo

Command: `python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_broad20.yaml`

Binary pair: strong `Qwen3-8B`, weak `MiMo-7B-RL-0530`.

These rows are local surrogates inspired by RouteLLM/LLMRouter-style baselines. They are not official external-repo reproductions.

Official RouteLLM/LLMRouterBench RouteLLM inspection found that the upstream adapter expects its own embedding/checkpoint pipeline; this run keeps the no-API local RouteCode split and deterministic embeddings.

| method | baseline_family | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- |
| query_oracle | reference | 0.9160 | 0.0000 | 1.0000 |
| best_single | reference | 0.7037 | 0.2123 | 0.0000 |
| routellm_pair_strong_only | binary_pair_reference | 0.7037 | 0.2123 | 0.0000 |
| kNN | reference | 0.7023 | 0.2137 | -0.0067 |
| routellm_style_mf_utility_router | external_style_surrogate | 0.6934 | 0.2226 | -0.0487 |
| routellm_binary_logistic_surrogate_t0.25 | external_style_surrogate | 0.6706 | 0.2454 | -0.1560 |
| routellm_binary_logistic_surrogate_t0.5 | external_style_surrogate | 0.5545 | 0.3615 | -0.7030 |
| routellm_binary_logistic_surrogate_t0.75 | external_style_surrogate | 0.3846 | 0.5313 | -1.5034 |
| routellm_pair_weak_only | binary_pair_reference | 0.3462 | 0.5698 | -1.6846 |

## References Used

- RouteLLM paper/repo: https://arxiv.org/abs/2406.18665 ; https://github.com/lm-sys/routellm
- LLMRouter repo: https://github.com/ulab-uiuc/LLMRouter
- LLMRouterBench paper/repo: https://arxiv.org/abs/2601.07206 ; https://github.com/ynulihao/LLMRouterBench

## Remaining External-Baseline Gap

- Run an official RouteLLM-MF/BERT baseline or an LLMRouterBench adapter output when its dependency and embedding pipeline can be pinned locally.
- Run GraphRouter/Avengers-Pro only after their commands, data contracts, and leakage controls are pinned.
