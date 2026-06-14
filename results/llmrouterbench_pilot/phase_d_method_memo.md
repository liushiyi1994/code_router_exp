# Phase D2 Predictability-Constrained RouteCode Memo

Command: `python experiments/06_predictability_constrained.py --config configs/llmrouterbench_pilot.yaml`

Data source: `llmrouterbench`. K = 16. Selected code-card alpha = 3. Beta = 0.

This is a pilot diagnostic. It should not be used as a novelty or full-benchmark claim.

## Main Comparison

| method | alpha | mean_utility | oracle_regret | recovered_gap_vs_oracle | label_accuracy | empirical_H_Z |
| --- | --- | --- | --- | --- | --- | --- |
| d2_embedding_centroid | 0.0000 | 0.6948 | 0.2017 | 0.1203 | 0.1500 | 3.7680 |
| d2_embedding_centroid | 0.0500 | 0.6086 | 0.2879 | -0.2556 | 0.1707 | 3.7602 |
| d2_embedding_centroid | 0.1000 | 0.6500 | 0.2466 | -0.0752 | 0.2862 | 3.6635 |
| d2_embedding_centroid | 0.3000 | 0.7379 | 0.1586 | 0.3083 | 0.5948 | 3.3379 |
| d2_embedding_centroid | 1.0000 | 0.7431 | 0.1534 | 0.3308 | 0.9086 | 3.2471 |
| d2_embedding_centroid | 3.0000 | 0.7466 | 0.1500 | 0.3459 | 0.9810 | 3.1120 |
| d2_embedding_centroid | 10.0000 | 0.7414 | 0.1552 | 0.3233 | 0.9897 | 3.0117 |
| d2_joint_oracle_labels | 0.0000 | 0.8759 | 0.0207 | 0.9098 | 1.0000 | 3.7324 |
| d2_joint_oracle_labels | 0.0500 | 0.8707 | 0.0259 | 0.8872 | 1.0000 | 3.7484 |
| d2_joint_oracle_labels | 0.1000 | 0.8466 | 0.0500 | 0.7820 | 1.0000 | 3.6662 |
| d2_joint_oracle_labels | 0.3000 | 0.7534 | 0.1431 | 0.3759 | 1.0000 | 3.2397 |
| d2_joint_oracle_labels | 1.0000 | 0.7466 | 0.1500 | 0.3459 | 1.0000 | 3.2357 |
| d2_joint_oracle_labels | 3.0000 | 0.7466 | 0.1500 | 0.3459 | 1.0000 | 3.1223 |
| d2_joint_oracle_labels | 10.0000 | 0.7431 | 0.1534 | 0.3308 | 1.0000 | 3.0067 |
| d2_logistic_label_predictor | 0.0000 | 0.6914 | 0.2052 | 0.1053 | 0.1466 | 3.8172 |
| d2_logistic_label_predictor | 0.0500 | 0.6328 | 0.2638 | -0.1504 | 0.1638 | 3.7613 |
| d2_logistic_label_predictor | 0.1000 | 0.6345 | 0.2621 | -0.1429 | 0.2207 | 3.6626 |
| d2_logistic_label_predictor | 0.3000 | 0.7241 | 0.1724 | 0.2481 | 0.4966 | 3.2099 |
| d2_logistic_label_predictor | 1.0000 | 0.7328 | 0.1638 | 0.2857 | 0.8259 | 3.2424 |
| d2_logistic_label_predictor | 3.0000 | 0.7448 | 0.1517 | 0.3383 | 0.8724 | 3.0611 |
| d2_logistic_label_predictor | 10.0000 | 0.7448 | 0.1517 | 0.3383 | 0.8672 | 2.9677 |
| dataset_label_lookup |  | 0.7534 | 0.1431 | 0.3759 |  |  |
| flat_routecode_logistic_label_predictor |  | 0.6138 | 0.2828 | -0.2331 |  | 3.8289 |
| flat_routecode_utility_oracle |  | 0.8897 | 0.0069 | 0.9699 |  | 3.6971 |
| kNN |  | 0.7362 | 0.1603 | 0.3008 |  |  |
| semantic_embedding_kmeans |  | 0.7362 | 0.1603 | 0.3008 |  | 3.6718 |

## Current Readout

- Best deployable D2 row in this sweep: `d2_embedding_centroid` at alpha `3`, mean utility `0.7466`, label accuracy `0.9810`.
- Interpret gains or losses against flat RouteCode and simple baselines before changing the main claim.
- If D2 improves predictability but loses substantial utility, the next step is a wider alpha/K sweep before new-model calibration.
