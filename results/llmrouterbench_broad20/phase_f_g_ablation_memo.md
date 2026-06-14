# Phase F/G Ablation And Robustness Memo

Command: `python experiments/08_ablation_summary.py --config configs/llmrouterbench_broad20.yaml`

This is a bounded robustness layer, not the full ablation matrix.

## Seed Stability

| method | mean_gap | std_gap |
| --- | --- | --- |
| d2_embedding_centroid | 0.0990 | 0.0118 |
| best_single | 0.0000 | 0.0000 |
| kNN | -0.0102 | 0.0049 |
| logistic_embedding_router | -0.9608 | 0.0916 |
| svm_embedding_router | -1.0157 | 0.0910 |

## Best K/Lambda Rows

| method | K | lambda_cost | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- |
| regret_routecode_utility_oracle | 32 | 0.1000 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 32 | 0.0000 | 0.9160 | 1.0000 |
| regret_routecode_utility_oracle | 16 | 0.0000 | 0.9092 | 0.9681 |
| regret_routecode_utility_oracle | 16 | 0.1000 | 0.9088 | 0.9665 |
| regret_routecode_utility_oracle | 8 | 0.0000 | 0.8850 | 0.8540 |
| regret_routecode_utility_oracle | 8 | 0.1000 | 0.8793 | 0.8272 |
| flat_routecode_utility_oracle | 32 | 0.0000 | 0.8130 | 0.5151 |
| flat_routecode_utility_oracle | 32 | 0.1000 | 0.8105 | 0.5033 |
| flat_routecode_utility_oracle | 16 | 0.0000 | 0.7813 | 0.3658 |
| flat_routecode_utility_oracle | 16 | 0.1000 | 0.7599 | 0.2652 |
| flat_routecode_utility_oracle | 8 | 0.1000 | 0.7300 | 0.1242 |
| flat_routecode_utility_oracle | 8 | 0.0000 | 0.7290 | 0.1191 |

## D2 Rate Penalty

| method | K | lambda_cost | d2_beta | mean_utility | recovered_gap_vs_oracle | empirical_H_Z |
| --- | --- | --- | --- | --- | --- | --- |
| d2_embedding_centroid | 16 | 0.0000 | 0.0000 | 0.7229 | 0.0906 | 3.6436 |
| d2_embedding_centroid | 16 | 0.0000 | 1.0000 | 0.7229 | 0.0906 | 3.6444 |

## Current Readout

- This covers seed stability, K/lambda sensitivity through the configured K sweep, semantic vs utility-vector vs regret-objective vs predictability-constrained code-objective comparison, D2 rate-penalty sensitivity, and training-fraction sensitivity for lightweight local methods.
- Regret-objective RouteCode is strong as an oracle-code diagnostic, but its embedding-centroid deployable rows in `table_rate_distortion.csv` remain far below the oracle-code ceiling.
- The separate Phase G sensitivity suite covers local embedding-feature variants, clustering algorithm, label noise, cost mis-estimation, bounded model-pool scenarios, query length, and bootstrap counts.
- Remaining robustness work still includes external embedding backbones, broader domain granularity beyond the coarse configured domain map, broader model pools, and stronger external baselines.
