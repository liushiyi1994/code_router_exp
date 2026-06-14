# Phase F/G Ablation And Robustness Memo

Command: `python experiments/08_ablation_summary.py --config configs/synthetic.yaml`

This is a bounded robustness layer, not the full ablation matrix.

## Seed Stability

| method | mean_gap | std_gap |
| --- | --- | --- |
| d2_embedding_centroid | 0.9632 | 0.0223 |
| kNN | 0.9612 | 0.0266 |
| svm_embedding_router | 0.9515 | 0.0106 |
| logistic_embedding_router | 0.9423 | 0.0105 |
| best_single | 0.0000 | 0.0000 |

## Best K/Lambda Rows

| method | K | lambda_cost | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- |
| flat_routecode_utility_oracle | 128 | 0.3500 | 0.6709 | 0.9848 |
| flat_routecode_utility_oracle | 64 | 0.3500 | 0.6693 | 0.9802 |
| flat_routecode_utility_oracle | 32 | 0.3500 | 0.6688 | 0.9789 |
| d2_embedding_centroid | 64 | 0.3500 | 0.6679 | 0.9762 |
| flat_routecode_utility_oracle | 128 | 0.7000 | 0.5625 | 0.9759 |
| flat_routecode_utility_oracle | 16 | 0.3500 | 0.6673 | 0.9746 |
| flat_routecode_utility_oracle | 64 | 0.7000 | 0.5620 | 0.9743 |
| semantic_embedding_kmeans | 64 | 0.3500 | 0.6663 | 0.9717 |
| semantic_embedding_kmeans | 32 | 0.3500 | 0.6663 | 0.9717 |
| semantic_embedding_kmeans | 16 | 0.3500 | 0.6663 | 0.9717 |
| d2_embedding_centroid | 128 | 0.3500 | 0.6660 | 0.9708 |
| d2_embedding_centroid | 16 | 0.3500 | 0.6660 | 0.9706 |

## Current Readout

- This covers seed stability, K/lambda sensitivity through the configured K sweep, code-objective comparison, and training-fraction sensitivity for lightweight local methods.
- The separate Phase G sensitivity suite covers local embedding-feature variants, clustering algorithm, label noise, cost mis-estimation, bounded model-pool scenarios, query length, and bootstrap counts.
- Remaining robustness work still includes external embedding backbones, true domain granularity, broader model pools, and stronger external baselines.
