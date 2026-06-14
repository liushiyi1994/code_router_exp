# Phase G Sensitivity Memo

Command: `python experiments/09_sensitivity_suite.py --config configs/synthetic.yaml`

This is a bounded sensitivity layer. It is not a full robustness proof.

| sensitivity | method | mean_gap | min_gap | max_gap |
| --- | --- | --- | --- | --- |
| bootstrap_sampling | d2_embedding_centroid | 0.9706 | 0.9706 | 0.9706 |
| clustering_algorithm | kNN | 0.9717 | 0.9717 | 0.9717 |
| clustering_algorithm | semantic_embedding_cluster | 0.9712 | 0.9706 | 0.9717 |
| clustering_algorithm | d2_embedding_centroid | 0.9706 | 0.9706 | 0.9706 |
| clustering_algorithm | best_single | 0.0000 | 0.0000 | 0.0000 |
| cost_misestimation | kNN | 0.9681 | 0.9610 | 0.9717 |
| cost_misestimation | d2_embedding_centroid | 0.9610 | 0.9417 | 0.9706 |
| cost_misestimation | best_single | -0.1649 | -0.2865 | 0.0000 |
| domain_granularity | kNN | 0.6927 | -0.7946 | 0.9973 |
| domain_granularity | d2_embedding_centroid | 0.6844 | -0.7946 | 0.9973 |
| domain_granularity | best_single | 0.0000 | 0.0000 | 0.0000 |
| embedding_backbone | kNN | 0.9435 | 0.9243 | 0.9717 |
| embedding_backbone | d2_embedding_centroid | 0.8768 | 0.7918 | 0.9706 |
| embedding_backbone | best_single | 0.0000 | 0.0000 | 0.0000 |
| label_noise | logistic_embedding_router | 0.9503 | 0.9470 | 0.9527 |
| model_pool | kNN | 0.9602 | 0.9502 | 0.9717 |
| model_pool | d2_embedding_centroid | 0.6702 | 0.5942 | 0.7405 |
| model_pool | best_single | 0.0000 | 0.0000 | 0.0000 |
| model_pool_auto | kNN | 0.7933 | 0.4328 | 0.9655 |
| model_pool_auto | d2_embedding_centroid | 0.4646 | 0.0496 | 0.8186 |
| model_pool_auto | best_single | 0.0000 | 0.0000 | 0.0000 |
| model_pool_composition | kNN | 0.8741 | 0.8696 | 0.8786 |
| model_pool_composition | d2_embedding_centroid | 0.5024 | 0.4024 | 0.6025 |
| model_pool_composition | best_single | 0.0000 | 0.0000 | 0.0000 |
| price_ratio | d2_embedding_centroid | 0.9426 | 0.8855 | 0.9706 |
| price_ratio | kNN | 0.9421 | 0.8841 | 0.9717 |
| price_ratio | best_single | 0.0000 | 0.0000 | 0.0000 |
| query_length_bucket | d2_embedding_centroid | 0.9708 | 0.9659 | 0.9765 |

## Current Readout

- Covered here: embedding feature variant, clustering algorithm, label noise, cost mis-estimation, price-ratio objective stress, model-pool subset/composition, automatic dominated/complementary model-pool construction, domain-granularity bucket sensitivity, query-length bucket sensitivity, and bootstrap sampling sensitivity.
- Configured model-pool composition now includes the pilot `qwen_pair`, `qwen_deepseek_llama`, and `compact_pair` slices when those models are available.
- Automatic model-pool construction selects dominated and complementary pools from the available model columns for configured sizes.
- Price-ratio rows flatten or expand model-average cost ratios before recomputing the cost-quality utility objective.
- Domain-granularity rows evaluate global router selections within coarse-domain, dataset, and train-fitted text-cluster buckets using bucket-local references.
- Split sensitivity now uses the configured coarse LLMRouterBench domain map; still missing or shallow are external embedding backbones and curated fine-grained benchmark taxonomies.
