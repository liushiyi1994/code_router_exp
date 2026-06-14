# Phase E RouteLLM Pairwise Alignment Memo

Command: `python experiments/14_routellm_pairwise_alignment.py --config configs/llmrouterbench_pilot.yaml`

Artifact directory: `results/llmrouterbench_pilot/routellm_pairwise`.

Binary pair: strong/model_a `Qwen3-8B`, weak/model_b `Qwen2.5-Coder-7B-Instruct`.

This memo records a RouteCode split-aligned RouteLLM-style pairwise data substrate. It preserves the RouteCode query-level train/test split and writes the strong/weak utility winner needed by RouteLLM-style binary routers.

The official RouteLLM evaluation remains incomplete: no RouteLLM MF/BERT model is trained or evaluated by this script, and no external embedding API or checkpoint download is used.

## Split Alignment

- `split_aligned_with_routecode`: `True`
- Train/test query overlap: `0`
- `official_routellm_result`: `False`

## Pairwise Summary

| split | record_count | decisive_count | tie_count | model_a_win_count | model_b_win_count | model_a_win_rate | model_b_win_rate | tie_rate | mean_utility_margin_model_a_minus_b | strong_model | weak_model | model_a | model_b | split_aligned_with_routecode | official_routellm_result | routecode_metric_compatible | implementation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1738 | 627 | 1111 | 393 | 234 | 0.2261 | 0.1346 | 0.6392 | 0.0915 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | False | False | Pairwise data substrate for later official RouteLLM evaluation; not an official RouteLLM MF/BERT result. |
| test | 580 | 206 | 374 | 120 | 86 | 0.2069 | 0.1483 | 0.6448 | 0.0586 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | False | False | Pairwise data substrate for later official RouteLLM evaluation; not an official RouteLLM MF/BERT result. |
| overall | 2318 | 833 | 1485 | 513 | 320 | 0.2213 | 0.1381 | 0.6406 | 0.0833 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | False | False | Pairwise data substrate for later official RouteLLM evaluation; not an official RouteLLM MF/BERT result. |

## References Used

- RouteLLM paper/repo: https://arxiv.org/abs/2406.18665 ; https://github.com/lm-sys/routellm
- LLMRouterBench paper/repo: https://arxiv.org/abs/2601.07206 ; https://github.com/ynulihao/LLMRouterBench

## Remaining External-Baseline Gap

- Run official RouteLLM-MF/BERT on this split-aligned pairwise substrate after local embedding/checkpoint dependencies are pinned.
- Report the exact command, pair, split, thresholds, and metric compatibility before ranking against RouteCode.
- GraphRouter and Avengers/Avengers-Pro remain separate external-baseline adapter tasks.
