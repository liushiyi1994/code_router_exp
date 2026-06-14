# Model-Pool Scale Run

## Model-Pool Scale Robustness

Command:

```bash
python experiments/18_model_pool_scale.py --config configs/llmrouterbench_scale20.yaml
```

Outputs:

- `table_model_pool_scale.csv`: top, complementary, dominated, and full model-pool rows for best-single, kNN, and D2.
- `phase_f_g_model_pool_scale_memo.md`: model-pool scale/composition checkpoint memo.

| pool_family | pool_name | model_count | method | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| complementary | complementary_2 | 2 | best_single | 0.7069 | 0.0000 |
| complementary | complementary_2 | 2 | d2_embedding_centroid | 0.7534 | 0.3913 |
| complementary | complementary_2 | 2 | kNN | 0.7552 | 0.4058 |
| dominated | dominated_2 | 2 | best_single | 0.7069 | 0.0000 |
| dominated | dominated_2 | 2 | d2_embedding_centroid | 0.7069 | 0.0000 |
| dominated | dominated_2 | 2 | kNN | 0.7017 | -0.1875 |
| top | top_2 | 2 | best_single | 0.7069 | 0.0000 |
| top | top_2 | 2 | d2_embedding_centroid | 0.6966 | -0.1154 |
| top | top_2 | 2 | kNN | 0.6966 | -0.1154 |
| complementary | complementary_4 | 4 | best_single | 0.7069 | 0.0000 |
| complementary | complementary_4 | 4 | d2_embedding_centroid | 0.7448 | 0.2157 |
| complementary | complementary_4 | 4 | kNN | 0.7517 | 0.2549 |
| dominated | dominated_4 | 4 | best_single | 0.7069 | 0.0000 |
| dominated | dominated_4 | 4 | d2_embedding_centroid | 0.7069 | 0.0000 |
| dominated | dominated_4 | 4 | kNN | 0.6966 | -0.1224 |
| top | top_4 | 4 | best_single | 0.7069 | 0.0000 |
| top | top_4 | 4 | d2_embedding_centroid | 0.7362 | 0.1828 |
| top | top_4 | 4 | kNN | 0.7345 | 0.1720 |
| complementary | complementary_8 | 8 | best_single | 0.7069 | 0.0000 |
| complementary | complementary_8 | 8 | d2_embedding_centroid | 0.7414 | 0.1538 |
| complementary | complementary_8 | 8 | kNN | 0.7431 | 0.1615 |
| dominated | dominated_8 | 8 | best_single | 0.7069 | 0.0000 |
| dominated | dominated_8 | 8 | d2_embedding_centroid | 0.6897 | -0.0935 |
| dominated | dominated_8 | 8 | kNN | 0.7138 | 0.0374 |
| top | top_8 | 8 | best_single | 0.7069 | 0.0000 |
| top | top_8 | 8 | d2_embedding_centroid | 0.7500 | 0.1953 |
| top | top_8 | 8 | kNN | 0.7517 | 0.2031 |
| complementary | complementary_12 | 12 | best_single | 0.7069 | 0.0000 |
| complementary | complementary_12 | 12 | d2_embedding_centroid | 0.7328 | 0.1042 |
| complementary | complementary_12 | 12 | kNN | 0.7448 | 0.1528 |
| dominated | dominated_12 | 12 | best_single | 0.7069 | 0.0000 |
| dominated | dominated_12 | 12 | d2_embedding_centroid | 0.6948 | -0.0534 |
| dominated | dominated_12 | 12 | kNN | 0.7155 | 0.0382 |
| top | top_12 | 12 | best_single | 0.7069 | 0.0000 |
| top | top_12 | 12 | d2_embedding_centroid | 0.7466 | 0.1597 |
| top | top_12 | 12 | kNN | 0.7431 | 0.1458 |
| complementary | complementary_16 | 16 | best_single | 0.7069 | 0.0000 |
| complementary | complementary_16 | 16 | d2_embedding_centroid | 0.7448 | 0.1477 |
| complementary | complementary_16 | 16 | kNN | 0.7397 | 0.1275 |
| dominated | dominated_16 | 16 | best_single | 0.7069 | 0.0000 |
| dominated | dominated_16 | 16 | d2_embedding_centroid | 0.7155 | 0.0357 |
| dominated | dominated_16 | 16 | kNN | 0.7017 | -0.0214 |
| top | top_16 | 16 | best_single | 0.7069 | 0.0000 |
| top | top_16 | 16 | d2_embedding_centroid | 0.7534 | 0.1837 |
| top | top_16 | 16 | kNN | 0.7397 | 0.1293 |
| full | full_20 | 20 | best_single | 0.7069 | 0.0000 |
| full | full_20 | 20 | d2_embedding_centroid | 0.7397 | 0.1275 |
| full | full_20 | 20 | kNN | 0.7241 | 0.0671 |

## Held-Out Model-Pool Transfer

Command:

```bash
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_scale20.yaml
```

Outputs:

- `table_model_pool_transfer.csv`: disjoint source/target pool transfer rows for target baselines, native D2, and transferred source-D2 labels.
- `phase_f_g_model_pool_transfer_memo.md`: held-out model-pool transfer checkpoint memo.

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
