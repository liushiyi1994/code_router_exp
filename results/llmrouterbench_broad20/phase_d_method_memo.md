# Phase D2 Predictability-Constrained RouteCode Memo

Command: `python experiments/06_predictability_constrained.py --config configs/llmrouterbench_broad20.yaml`

Data source: `llmrouterbench`. K = 16. Selected code-card alpha = 1. Beta = 0.

This is a pilot diagnostic. It should not be used as a novelty or full-benchmark claim.

## Main Comparison

| method | alpha | mean_utility | oracle_regret | recovered_gap_vs_oracle | label_accuracy | empirical_H_Z |
| --- | --- | --- | --- | --- | --- | --- |
| d2_embedding_centroid | 0.0000 | 0.6585 | 0.2575 | -0.2131 | 0.2204 | 3.7428 |
| d2_embedding_centroid | 0.0500 | 0.6592 | 0.2568 | -0.2097 | 0.3048 | 3.7500 |
| d2_embedding_centroid | 0.1000 | 0.6802 | 0.2358 | -0.1107 | 0.3896 | 3.8454 |
| d2_embedding_centroid | 0.3000 | 0.7197 | 0.1962 | 0.0755 | 0.5648 | 3.6430 |
| d2_embedding_centroid | 1.0000 | 0.7229 | 0.1930 | 0.0906 | 0.7646 | 3.6436 |
| d2_embedding_centroid | 3.0000 | 0.7172 | 0.1987 | 0.0638 | 0.9776 | 3.2946 |
| d2_embedding_centroid | 10.0000 | 0.7190 | 0.1969 | 0.0721 | 0.9950 | 3.4146 |
| d2_joint_oracle_labels | 0.0000 | 0.7899 | 0.1261 | 0.4060 | 1.0000 | 3.8243 |
| d2_joint_oracle_labels | 0.0500 | 0.7696 | 0.1464 | 0.3104 | 1.0000 | 3.8572 |
| d2_joint_oracle_labels | 0.1000 | 0.7496 | 0.1663 | 0.2164 | 1.0000 | 3.8294 |
| d2_joint_oracle_labels | 0.3000 | 0.7240 | 0.1920 | 0.0956 | 1.0000 | 3.6138 |
| d2_joint_oracle_labels | 1.0000 | 0.7276 | 0.1884 | 0.1124 | 1.0000 | 3.6050 |
| d2_joint_oracle_labels | 3.0000 | 0.7183 | 0.1976 | 0.0688 | 1.0000 | 3.3036 |
| d2_joint_oracle_labels | 10.0000 | 0.7187 | 0.1973 | 0.0705 | 1.0000 | 3.4136 |
| d2_logistic_label_predictor | 0.0000 | 0.6115 | 0.3045 | -0.4346 | 0.2365 | 3.5318 |
| d2_logistic_label_predictor | 0.0500 | 0.6150 | 0.3009 | -0.4178 | 0.3063 | 3.6892 |
| d2_logistic_label_predictor | 0.1000 | 0.6278 | 0.2881 | -0.3574 | 0.3914 | 3.7432 |
| d2_logistic_label_predictor | 0.3000 | 0.7176 | 0.1984 | 0.0654 | 0.5552 | 3.5963 |
| d2_logistic_label_predictor | 1.0000 | 0.7187 | 0.1973 | 0.0705 | 0.7472 | 3.6059 |
| d2_logistic_label_predictor | 3.0000 | 0.7176 | 0.1984 | 0.0654 | 0.9523 | 3.3044 |
| d2_logistic_label_predictor | 10.0000 | 0.7194 | 0.1966 | 0.0738 | 0.9537 | 3.4115 |
| dataset_label_lookup |  | 0.7172 | 0.1987 | 0.0638 |  |  |
| flat_routecode_logistic_label_predictor |  | 0.6022 | 0.3137 | -0.4782 |  | 3.4755 |
| flat_routecode_utility_oracle |  | 0.7813 | 0.1346 | 0.3658 |  | 3.8177 |
| kNN |  | 0.7023 | 0.2137 | -0.0067 |  |  |
| semantic_embedding_kmeans |  | 0.7222 | 0.1937 | 0.0872 |  | 3.8902 |

## Current Readout

- Best deployable D2 row in this sweep: `d2_embedding_centroid` at alpha `1`, mean utility `0.7229`, label accuracy `0.7646`.
- Interpret gains or losses against flat RouteCode and simple baselines before changing the main claim.
- If D2 improves predictability but loses substantial utility, the next step is a wider alpha/K sweep before new-model calibration.
