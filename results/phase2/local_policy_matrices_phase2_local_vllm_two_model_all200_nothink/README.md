# RouteCode Phase 2 Results

## Local Outcomes Policy Matrices

Converts exact-scored local model outcomes to `query_model_utility` and `state_model_utility` matrices for ProbeRoute++ policy evaluation.

- Local outcomes: `results/phase2/local_vllm_two_model_all200_nothink/local_model_outcomes.parquet`
- State targets: `results/phase2/aligned_offline/aligned_state_targets.csv`
- Utility formula: `quality - 0 * cost_proxy`.

| outcome_rows | overlap_rows | train_rows | policy_rows | policy_queries | train_queries | model_count | state_count | lambda_cost | policy_split | train_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 328 | 246 | 82 | 41 | 123 | 2 | 16 | 0.0000 | test | train |

| artifact | path |
| --- | --- |
| query_model_utility | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_query_model_utility.csv |
| query_model_quality | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_query_model_quality.csv |
| query_model_cost | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_query_model_cost.csv |
| state_model_utility | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_state_model_utility.csv |
| state_model_quality | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_state_model_quality.csv |
| state_model_cost | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_state_model_cost.csv |
| metadata | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_policy_matrix_metadata.json |
