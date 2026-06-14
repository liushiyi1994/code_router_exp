# Phase E Official Baseline Artifact Memo

Command: `python experiments/12_official_baseline_artifacts.py --config configs/llmrouterbench_pilot.yaml`

Source directory: `data/raw/external/LLMRouterBench/baselines/RouteLLM/results`.

This memo records official RouteLLM MF artifacts found in the local LLMRouterBench checkout. These artifacts are useful for baseline inspection, dependency pinning, and novelty-boundary tracking, but they are not evaluated on the RouteCode train/test split or utility objective.

## Compatibility

- `split_aligned_with_routecode` is false for every row.
- `routecode_metric_compatible` is false for every row.
- Do not use these rows for direct method-ranking claims against RouteCode utility tables.

## Overall Upstream MF Results

| method | seed | total | selection_accuracy | routing_accuracy | total_cost | csv_selection_accuracy | csv_total_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RouteLLM-MF | 42 | 3858 | 0.6029 | 0.6881 | 124.9606 | 0.6029 | 124.9606 |
| RouteLLM-MF | 999 | 3858 | 0.6262 | 0.6952 | 126.6179 | 0.6262 | 126.6179 |
| RouteLLM-MF | 2024 | 3858 | 0.6260 | 0.6929 | 126.1494 | 0.6260 | 126.1494 |
| RouteLLM-MF | 2025 | 3858 | 0.6179 | 0.6980 | 124.6618 | 0.6179 | 124.6618 |
| RouteLLM-MF | 3407 | 3858 | 0.6094 | 0.6827 | 126.0154 | 0.6094 | 126.0154 |

## Dataset Coverage

aime, arc-agi, arenahard_coding, arenahard_creative_writing, arenahard_math, gpqa, hle, livecodebench, livemathbench, mmlupro, simpleqa, swe-bench, tau2

## References Used

- LLMRouterBench RouteLLM baseline artifacts: `data/raw/external/LLMRouterBench/baselines/RouteLLM/results`.
- RouteLLM paper/repo: https://arxiv.org/abs/2406.18665 ; https://github.com/lm-sys/routellm
- LLMRouterBench paper/repo: https://arxiv.org/abs/2601.07206 ; https://github.com/ynulihao/LLMRouterBench

## Remaining External-Baseline Gap

- A split-aligned official RouteLLM reproduction still requires running the upstream router on the RouteCode train/test split with pinned embeddings/checkpoints.
- GraphRouter, BEST-Route, and other external baselines still need pinned local commands before any direct ranking claim.
