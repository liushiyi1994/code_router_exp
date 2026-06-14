# Phase F/G Held-Out Model-Pool Transfer Memo

Command: `python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_scale20.yaml`

This run evaluates disjoint source and target model pools. Route labels are learned on the source pool, then remapped to target-pool models using target train utility only.

| transfer_scenario | source_model_count | target_model_count | method | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| complementary_to_remaining_top | 8 | 8 | source_d2_label_transfer | 0.7207 | 0.3588 |
| complementary_to_remaining_top | 8 | 8 | target_best_single | 0.6155 | 0.0000 |
| complementary_to_remaining_top | 8 | 8 | target_d2_native | 0.7207 | 0.3588 |
| complementary_to_remaining_top | 8 | 8 | target_direct_gradient_boosting | 0.6138 | -0.0059 |
| complementary_to_remaining_top | 8 | 8 | target_direct_knn | 0.6190 | 0.0118 |
| complementary_to_remaining_top | 8 | 8 | target_direct_logistic | 0.6310 | 0.0529 |
| complementary_to_remaining_top | 8 | 8 | target_direct_mlp | 0.6466 | 0.1059 |
| complementary_to_remaining_top | 8 | 8 | target_direct_svm | 0.6207 | 0.0176 |
| complementary_to_remaining_top | 8 | 8 | target_kNN | 0.7121 | 0.3294 |
| dominated_to_remaining_top | 8 | 8 | source_d2_label_transfer | 0.7328 | 0.2714 |
| dominated_to_remaining_top | 8 | 8 | target_best_single | 0.6672 | 0.0000 |
| dominated_to_remaining_top | 8 | 8 | target_d2_native | 0.7276 | 0.2500 |
| dominated_to_remaining_top | 8 | 8 | target_direct_gradient_boosting | 0.6672 | 0.0000 |
| dominated_to_remaining_top | 8 | 8 | target_direct_knn | 0.6690 | 0.0071 |
| dominated_to_remaining_top | 8 | 8 | target_direct_logistic | 0.6741 | 0.0286 |
| dominated_to_remaining_top | 8 | 8 | target_direct_mlp | 0.6603 | -0.0286 |
| dominated_to_remaining_top | 8 | 8 | target_direct_svm | 0.6741 | 0.0286 |
| dominated_to_remaining_top | 8 | 8 | target_kNN | 0.7172 | 0.2071 |
| top_to_next | 8 | 8 | source_d2_label_transfer | 0.5966 | 0.0442 |
| top_to_next | 8 | 8 | target_best_single | 0.5828 | 0.0000 |
| top_to_next | 8 | 8 | target_d2_native | 0.6103 | 0.0884 |
| top_to_next | 8 | 8 | target_direct_gradient_boosting | 0.5776 | -0.0166 |
| top_to_next | 8 | 8 | target_direct_knn | 0.5862 | 0.0110 |
| top_to_next | 8 | 8 | target_direct_logistic | 0.6000 | 0.0552 |
| top_to_next | 8 | 8 | target_direct_mlp | 0.5810 | -0.0055 |
| top_to_next | 8 | 8 | target_direct_svm | 0.5897 | 0.0221 |
| top_to_next | 8 | 8 | target_kNN | 0.6052 | 0.0718 |

## Transfer Readout

- Transfer scenarios: `3`.
- Source/target overlap max: `0`.
- Transferred D2 recovered-gap range: `0.0442` to `0.3588`.
- Direct retraining recovered-gap range: `-0.0286` to `0.1059`.

Interpretation: this is a held-out model-pool diagnostic. It does not prove transfer unless transferred labels are competitive with same-budget direct retraining across broader datasets and pools.
