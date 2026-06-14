# Phase E External Baseline Surrogate Memo

Command: `python experiments/10_external_baseline_surrogates.py --config configs/synthetic.yaml`

Binary pair: strong `reasoner_13b`, weak `general_8b`.

These rows are local surrogates inspired by RouteLLM/LLMRouter-style baselines. They are not official external-repo reproductions.

Official RouteLLM/LLMRouterBench RouteLLM inspection found that the upstream adapter expects its own embedding/checkpoint pipeline; this run keeps the no-API local RouteCode split and deterministic embeddings.

| method | baseline_family | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- |
| query_oracle | reference | 0.6761 | 0.0000 | 1.0000 |
| kNN | reference | 0.6663 | 0.0097 | 0.9717 |
| routellm_style_mf_utility_router | external_style_surrogate | 0.6243 | 0.0518 | 0.8497 |
| routellm_binary_logistic_surrogate_t0.5 | external_style_surrogate | 0.3679 | 0.3082 | 0.1057 |
| routellm_binary_logistic_surrogate_t0.75 | external_style_surrogate | 0.3612 | 0.3149 | 0.0863 |
| routellm_binary_logistic_surrogate_t0.25 | external_style_surrogate | 0.3492 | 0.3269 | 0.0513 |
| best_single | reference | 0.3315 | 0.3446 | 0.0000 |
| routellm_pair_strong_only | binary_pair_reference | 0.3315 | 0.3446 | 0.0000 |
| routellm_pair_weak_only | binary_pair_reference | 0.1979 | 0.4782 | -0.3877 |

## References Used

- RouteLLM paper/repo: https://arxiv.org/abs/2406.18665 ; https://github.com/lm-sys/routellm
- LLMRouter repo: https://github.com/ulab-uiuc/LLMRouter
- LLMRouterBench paper/repo: https://arxiv.org/abs/2601.07206 ; https://github.com/ynulihao/LLMRouterBench

## Remaining External-Baseline Gap

- Run an official RouteLLM-MF/BERT baseline or an LLMRouterBench adapter output when its dependency and embedding pipeline can be pinned locally.
- Run GraphRouter/Avengers-Pro only after their commands, data contracts, and leakage controls are pinned.
