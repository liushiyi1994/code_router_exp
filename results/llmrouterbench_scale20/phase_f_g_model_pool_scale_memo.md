# Phase F/G Model-Pool Scale Memo

Command: `python experiments/18_model_pool_scale.py --config configs/llmrouterbench_scale20.yaml`

This run evaluates top, complementary, and dominated model-pool scenarios using train-only pool construction. It extends the bounded model-pool sensitivity layer without changing the RouteCode method.

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

## D2 Range

- D2 rows: `16`.
- D2 recovered-gap range: `-0.1154` to `0.3913`.
- Model-count range: `2` to `20`.

Interpretation: this is a robustness and diagnosis layer. It should not be used as a final model-pool transfer claim without additional held-out pool protocols and direct-router retraining comparisons.
