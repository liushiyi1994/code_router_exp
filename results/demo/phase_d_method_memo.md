# Phase D2 Predictability-Constrained RouteCode Memo

Command: `python experiments/06_predictability_constrained.py --config configs/synthetic.yaml`

Data source: `synthetic`. K = 16. Selected code-card alpha = 3. Beta = 0.

This is a pilot diagnostic. It should not be used as a novelty or full-benchmark claim.

## Main Comparison

| method | alpha | mean_utility | oracle_regret | recovered_gap_vs_oracle | label_accuracy | empirical_H_Z |
| --- | --- | --- | --- | --- | --- | --- |
| d2_embedding_centroid | 0.0000 | 0.6607 | 0.0154 | 0.9552 | 0.3729 | 3.3932 |
| d2_embedding_centroid | 0.0500 | 0.6626 | 0.0135 | 0.9608 | 0.4292 | 3.6223 |
| d2_embedding_centroid | 0.1000 | 0.6639 | 0.0121 | 0.9648 | 0.6208 | 3.7213 |
| d2_embedding_centroid | 0.3000 | 0.6655 | 0.0106 | 0.9693 | 0.6792 | 3.9155 |
| d2_embedding_centroid | 1.0000 | 0.6660 | 0.0101 | 0.9706 | 0.7812 | 3.7877 |
| d2_embedding_centroid | 3.0000 | 0.6660 | 0.0101 | 0.9706 | 0.9688 | 3.8054 |
| d2_embedding_centroid | 10.0000 | 0.6660 | 0.0101 | 0.9706 | 0.9979 | 3.6662 |
| d2_joint_oracle_labels | 0.0000 | 0.6683 | 0.0078 | 0.9775 | 1.0000 | 3.8532 |
| d2_joint_oracle_labels | 0.0500 | 0.6710 | 0.0051 | 0.9851 | 1.0000 | 3.8832 |
| d2_joint_oracle_labels | 0.1000 | 0.6689 | 0.0072 | 0.9792 | 1.0000 | 3.8861 |
| d2_joint_oracle_labels | 0.3000 | 0.6659 | 0.0102 | 0.9705 | 1.0000 | 3.9017 |
| d2_joint_oracle_labels | 1.0000 | 0.6663 | 0.0097 | 0.9717 | 1.0000 | 3.7816 |
| d2_joint_oracle_labels | 3.0000 | 0.6660 | 0.0101 | 0.9706 | 1.0000 | 3.8102 |
| d2_joint_oracle_labels | 10.0000 | 0.6660 | 0.0101 | 0.9706 | 1.0000 | 3.6624 |
| d2_logistic_label_predictor | 0.0000 | 0.6537 | 0.0224 | 0.9349 | 0.3583 | 3.7185 |
| d2_logistic_label_predictor | 0.0500 | 0.6588 | 0.0173 | 0.9498 | 0.4083 | 3.8218 |
| d2_logistic_label_predictor | 0.1000 | 0.6637 | 0.0124 | 0.9639 | 0.6000 | 3.7883 |
| d2_logistic_label_predictor | 0.3000 | 0.6633 | 0.0128 | 0.9629 | 0.6896 | 3.8124 |
| d2_logistic_label_predictor | 1.0000 | 0.6663 | 0.0097 | 0.9717 | 0.8021 | 3.7707 |
| d2_logistic_label_predictor | 3.0000 | 0.6657 | 0.0104 | 0.9698 | 0.9375 | 3.7637 |
| d2_logistic_label_predictor | 10.0000 | 0.6663 | 0.0097 | 0.9717 | 0.9854 | 3.6495 |
| dataset_label_lookup |  | 0.6535 | 0.0226 | 0.9344 |  |  |
| flat_routecode_logistic_label_predictor |  | 0.6504 | 0.0257 | 0.9255 |  | 3.8035 |
| flat_routecode_utility_oracle |  | 0.6673 | 0.0088 | 0.9746 |  | 3.9156 |
| kNN |  | 0.6663 | 0.0097 | 0.9717 |  |  |
| semantic_embedding_kmeans |  | 0.6663 | 0.0097 | 0.9717 |  | 3.7225 |

## Current Readout

- Best deployable D2 row in this sweep: `d2_logistic_label_predictor` at alpha `10`, mean utility `0.6663`, label accuracy `0.9854`.
- Interpret gains or losses against flat RouteCode and simple baselines before changing the main claim.
- If D2 improves predictability but loses substantial utility, the next step is a wider alpha/K sweep before new-model calibration.
