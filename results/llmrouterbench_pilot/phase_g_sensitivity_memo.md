# Phase G Sensitivity Memo

Command: `python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_pilot.yaml`

This is a bounded sensitivity layer. It is not a full robustness proof.

| sensitivity | method | mean_gap | min_gap | max_gap |
| --- | --- | --- | --- | --- |
| bootstrap_sampling | d2_embedding_centroid | 0.3459 | 0.3459 | 0.3459 |
| clustering_algorithm | d2_embedding_centroid | 0.3459 | 0.3459 | 0.3459 |
| clustering_algorithm | semantic_embedding_cluster | 0.3308 | 0.3008 | 0.3609 |
| clustering_algorithm | kNN | 0.3008 | 0.3008 | 0.3008 |
| clustering_algorithm | best_single | 0.0000 | 0.0000 | 0.0000 |
| cost_misestimation | d2_embedding_centroid | 0.4305 | 0.4180 | 0.4461 |
| cost_misestimation | kNN | 0.4236 | 0.3843 | 0.4753 |
| cost_misestimation | best_single | 0.0161 | 0.0000 | 0.0483 |
| domain_granularity | d2_embedding_centroid | 0.1668 | -0.2000 | 0.7600 |
| domain_granularity | kNN | 0.0277 | -1.0000 | 0.8000 |
| domain_granularity | best_single | 0.0000 | 0.0000 | 0.0000 |
| embedding_backbone | d2_embedding_centroid | 0.3233 | 0.2782 | 0.3459 |
| embedding_backbone | kNN | 0.2757 | 0.2256 | 0.3008 |
| embedding_backbone | best_single | 0.0000 | 0.0000 | 0.0000 |
| label_noise | logistic_embedding_router | -0.0627 | -0.1579 | 0.0301 |
| model_pool | d2_embedding_centroid | 0.3682 | 0.3008 | 0.4539 |
| model_pool | kNN | 0.3678 | 0.3008 | 0.4276 |
| model_pool | best_single | 0.0000 | 0.0000 | 0.0000 |
| model_pool_auto | d2_embedding_centroid | 0.2839 | -0.0200 | 0.6796 |
| model_pool_auto | kNN | 0.2167 | -0.2041 | 0.6893 |
| model_pool_auto | best_single | 0.0000 | 0.0000 | 0.0000 |
| model_pool_composition | d2_embedding_centroid | 0.2246 | -0.0267 | 0.5116 |
| model_pool_composition | kNN | 0.1730 | -0.0778 | 0.4767 |
| model_pool_composition | best_single | 0.0000 | 0.0000 | 0.0000 |
| price_ratio | kNN | 0.3959 | 0.3519 | 0.4440 |
| price_ratio | d2_embedding_centroid | 0.3699 | 0.3167 | 0.4203 |
| price_ratio | best_single | 0.0000 | 0.0000 | 0.0000 |
| query_length_bucket | d2_embedding_centroid | 0.3163 | -0.0476 | 0.6140 |

## Current Readout

- Covered here: embedding feature variant, clustering algorithm, label noise, cost mis-estimation, price-ratio objective stress, model-pool subset/composition, automatic dominated/complementary model-pool construction, domain-granularity bucket sensitivity, query-length bucket sensitivity, and bootstrap sampling sensitivity.
- Configured model-pool composition now includes the pilot `qwen_pair`, `qwen_deepseek_llama`, and `compact_pair` slices when those models are available.
- Automatic model-pool construction selects dominated and complementary pools from the available model columns for configured sizes.
- Price-ratio rows flatten or expand model-average cost ratios before recomputing the cost-quality utility objective.
- Domain-granularity rows evaluate global router selections within coarse-domain, curated task-family/task-subtype taxonomy, dataset, and train-fitted text-cluster buckets using bucket-local references.
- Split sensitivity now uses the configured coarse LLMRouterBench domain map; still missing or shallow are external embedding backbones and larger benchmark-scale taxonomy coverage.
