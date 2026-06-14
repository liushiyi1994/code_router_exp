# Phase F/G Held-Out Model-Pool Transfer Memo

Command: `python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_broad20.yaml`

This run evaluates disjoint source and target model pools. Route labels are learned on the source pool, then remapped to target-pool models using target train utility only.

| transfer_scenario | source_model_count | target_model_count | method | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| complementary_to_remaining_top | 8 | 8 | source_d2_label_transfer | 0.7140 | 0.1739 |
| complementary_to_remaining_top | 8 | 8 | target_best_single | 0.6813 | 0.0000 |
| complementary_to_remaining_top | 8 | 8 | target_d2_native | 0.7130 | 0.1682 |
| complementary_to_remaining_top | 8 | 8 | target_direct_knn | 0.6830 | 0.0095 |
| complementary_to_remaining_top | 8 | 8 | target_direct_logistic | 0.6841 | 0.0151 |
| complementary_to_remaining_top | 8 | 8 | target_direct_svm | 0.6820 | 0.0038 |
| complementary_to_remaining_top | 8 | 8 | target_kNN | 0.6927 | 0.0605 |
| dominated_to_remaining_top | 8 | 8 | source_d2_label_transfer | 0.7123 | 0.2162 |
| dominated_to_remaining_top | 8 | 8 | target_best_single | 0.6656 | 0.0000 |
| dominated_to_remaining_top | 8 | 8 | target_d2_native | 0.7098 | 0.2046 |
| dominated_to_remaining_top | 8 | 8 | target_direct_knn | 0.6660 | 0.0017 |
| dominated_to_remaining_top | 8 | 8 | target_direct_logistic | 0.6699 | 0.0198 |
| dominated_to_remaining_top | 8 | 8 | target_direct_svm | 0.6674 | 0.0083 |
| dominated_to_remaining_top | 8 | 8 | target_kNN | 0.7037 | 0.1766 |
| top_to_next | 8 | 8 | source_d2_label_transfer | 0.6047 | 0.2274 |
| top_to_next | 8 | 8 | target_best_single | 0.5410 | 0.0000 |
| top_to_next | 8 | 8 | target_d2_native | 0.6136 | 0.2592 |
| top_to_next | 8 | 8 | target_direct_knn | 0.5520 | 0.0394 |
| top_to_next | 8 | 8 | target_direct_logistic | 0.5495 | 0.0305 |
| top_to_next | 8 | 8 | target_direct_svm | 0.5413 | 0.0013 |
| top_to_next | 8 | 8 | target_kNN | 0.5990 | 0.2071 |

## Transfer Readout

- Transfer scenarios: `3`.
- Source/target overlap max: `0`.
- Transferred D2 recovered-gap range: `0.1739` to `0.2274`.
- Direct retraining recovered-gap range: `0.0013` to `0.0394`.

Interpretation: this is a held-out model-pool diagnostic. It does not prove transfer unless transferred labels are competitive with same-budget direct retraining across broader datasets and pools.
