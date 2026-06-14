# Phase F/G Ablation And Robustness Memo

Command: `python experiments/08_ablation_summary.py --config configs/llmrouterbench_pilot.yaml`

This is a bounded robustness layer, not the full ablation matrix.

## Seed Stability

| method | mean_gap | std_gap |
| --- | --- | --- |
| d2_embedding_centroid | 0.3320 | 0.0177 |
| kNN | 0.2723 | 0.0356 |
| svm_embedding_router | 0.0052 | 0.0232 |
| best_single | 0.0000 | 0.0000 |
| logistic_embedding_router | -0.0064 | 0.0401 |

## Best K/Lambda Rows

| method | K | lambda_cost | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- |
| regret_routecode_utility_oracle | 32 | 0.0000 | 0.8966 | 1.0000 |
| flat_routecode_utility_oracle | 32 | 0.0000 | 0.8966 | 1.0000 |
| regret_routecode_utility_oracle | 16 | 0.0000 | 0.8966 | 1.0000 |
| regret_routecode_utility_oracle | 8 | 0.0000 | 0.8966 | 1.0000 |
| regret_routecode_utility_oracle | 64 | 0.0000 | 0.8966 | 1.0000 |
| regret_routecode_utility_oracle | 128 | 0.0000 | 0.8966 | 1.0000 |
| regret_routecode_utility_oracle | 8 | 0.0500 | 0.8965 | 1.0000 |
| regret_routecode_utility_oracle | 128 | 0.1000 | 0.8965 | 1.0000 |
| regret_routecode_utility_oracle | 64 | 0.2000 | 0.8965 | 1.0000 |
| regret_routecode_utility_oracle | 128 | 0.2000 | 0.8965 | 1.0000 |
| regret_routecode_utility_oracle | 32 | 0.2000 | 0.8965 | 1.0000 |
| regret_routecode_utility_oracle | 16 | 0.2000 | 0.8965 | 1.0000 |

## D2 Rate Penalty

| method | K | lambda_cost | d2_beta | mean_utility | recovered_gap_vs_oracle | empirical_H_Z |
| --- | --- | --- | --- | --- | --- | --- |
| d2_embedding_centroid | 16 | 0.0000 | 0.0000 | 0.7466 | 0.3459 | 3.1120 |
| d2_embedding_centroid | 16 | 0.0000 | 0.1000 | 0.7466 | 0.3459 | 3.1120 |
| d2_embedding_centroid | 16 | 0.0000 | 1.0000 | 0.7466 | 0.3459 | 3.1120 |
| d2_embedding_centroid | 16 | 0.0000 | 3.0000 | 0.7466 | 0.3459 | 3.1120 |

## Current Readout

- This covers seed stability, K/lambda sensitivity through the configured K sweep, semantic vs utility-vector vs regret-objective vs predictability-constrained code-objective comparison, D2 rate-penalty sensitivity, and training-fraction sensitivity for lightweight local methods.
- Regret-objective RouteCode is strong as an oracle-code diagnostic, but its embedding-centroid deployable rows in `table_rate_distortion.csv` remain far below the oracle-code ceiling.
- The separate Phase G sensitivity suite covers local embedding-feature variants, clustering algorithm, label noise, cost mis-estimation, bounded model-pool scenarios, query length, and bootstrap counts.
- Remaining robustness work still includes external embedding backbones, broader domain granularity beyond the coarse configured domain map, broader model pools, and stronger external baselines.
