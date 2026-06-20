# Phase E Avengers-Pro Split-Aligned Memo

Command: `python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_pilot.yaml`

Run assets: `results/llmrouterbench_pilot/avengerspro_split_aligned`.

This run uses a local implementation of the Avengers-Pro cluster-routing contract: K-means over query embeddings, train-only per-cluster model rankings, and nearest-cluster routing. It uses RouteCode deterministic embeddings and the RouteCode train/test split, so it makes no embedding API calls.

This is not an official upstream command-path run, not an upstream checkpoint, and not evidence for paper-level Avengers-Pro performance. It is a split-aligned compatibility baseline for the RouteCode pilot.

| method | mean_utility | oracle_regret | recovered_gap_vs_oracle | selected_model_entropy |
| --- | --- | --- | --- | --- |
| avengerspro_simple_cluster_k16 | 0.7397 | 0.1569 | 0.3158 | 1.3777 |
| avengerspro_balance_cluster_k16_w0.7_c0.3 | 0.7241 | 0.1724 | 0.2481 | 1.1424 |

## Adapter Notes

- Avengers-Pro source inspected: `data/raw/external/LLMRouterBench/baselines/AvengersPro`.
- Official Avengers-Pro scripts require an embedding service configuration by default; RouteCode also writes a bounded cache-backed smoke config for local no-API upstream command checks.
- GraphRouter remains blocked for local metric rows because the current environment lacks PyG packages and its data construction path expects generated graph inputs plus embedding configuration.

## References Used

- Avengers-Pro source in LLMRouterBench: https://github.com/ynulihao/LLMRouterBench/tree/main/baselines/AvengersPro
- GraphRouter source in LLMRouterBench: https://github.com/ynulihao/LLMRouterBench/tree/main/baselines/GraphRouter
