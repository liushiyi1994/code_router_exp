# Phase F/G Ablation And Robustness Memo

Command: `python experiments/08_ablation_summary.py --config configs/llmrouterbench_broad20.yaml`

This is a bounded robustness layer, not the full ablation matrix.

## Seed Stability

| method | mean_gap | std_gap |
| --- | --- | --- |
| d2_embedding_centroid | 0.0901 | 0.0337 |
| kNN | 0.0145 | 0.0429 |
| best_single | 0.0000 | 0.0000 |
| logistic_embedding_router | -0.9806 | 0.0626 |
| svm_embedding_router | -1.0200 | 0.0664 |

## Best K/Lambda Rows

| method | K | lambda_cost | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- |
| regret_routecode_utility_oracle | 32 | 0.0000 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 128 | 0.0000 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 64 | 0.0000 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 32 | 0.0500 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 64 | 0.1000 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 128 | 0.1000 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 64 | 0.2000 | 0.9159 | 1.0000 |
| regret_routecode_utility_oracle | 32 | 0.2000 | 0.9159 | 1.0000 |
| regret_routecode_utility_oracle | 32 | 0.1000 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 128 | 0.0500 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 64 | 0.0500 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 128 | 0.2000 | 0.9159 | 1.0000 |

## D2 Rate Penalty

| method | K | lambda_cost | d2_beta | mean_utility | recovered_gap_vs_oracle | empirical_H_Z |
| --- | --- | --- | --- | --- | --- | --- |
| d2_embedding_centroid | 16 | 0.0000 | 0.0000 | 0.7147 | 0.0520 | 3.5714 |
| d2_embedding_centroid | 16 | 0.0000 | 0.1000 | 0.7144 | 0.0503 | 3.5714 |
| d2_embedding_centroid | 16 | 0.0000 | 1.0000 | 0.7147 | 0.0520 | 3.5729 |
| d2_embedding_centroid | 16 | 0.0000 | 3.0000 | 0.7147 | 0.0520 | 3.5736 |

## Current Readout

- This covers seed stability, K/lambda sensitivity through the configured K sweep, semantic vs utility-vector vs regret-objective vs predictability-constrained code-objective comparison, D2 rate-penalty sensitivity, and training-fraction sensitivity for best-single, kNN, lightweight direct routers, and D2 RouteCode.
- Regret-objective RouteCode is strong as an oracle-code diagnostic, but its embedding-centroid deployable rows in `table_rate_distortion.csv` remain far below the oracle-code ceiling.
- The separate Phase G sensitivity suite covers local embedding-feature variants, clustering algorithm, label noise, cost mis-estimation, bounded model-pool scenarios, query length, and bootstrap counts.
- Remaining robustness work still includes external embedding backbones, broader domain granularity beyond the coarse configured domain map, broader model pools, and stronger external baselines.
