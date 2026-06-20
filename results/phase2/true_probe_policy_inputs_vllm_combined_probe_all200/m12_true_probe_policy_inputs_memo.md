# True Local Probe Policy Inputs

This step turns true local probe features into latent route-state beliefs for M5 policy evaluation. It is not direct probe-to-model routing.

Inputs:

- Probe features: `results/phase2/local_vllm_qwen3_4b_all200_combined_probe_features/probe_features.parquet`
- State targets: `results/phase2/aligned_offline/aligned_state_targets.csv`
- Query features: `results/phase2/aligned_offline/aligned_query_features.csv`

Summary:

- Train rows for belief models: `123`.
- Policy query rows: `41`.
- Route states: `16`.

Outputs:

| artifact | path |
| --- | --- |
| before_beliefs | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_before_beliefs.csv |
| after_beliefs | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_after_beliefs.csv |
| state_model_utility | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_state_model_utility.csv |
| query_model_utility | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_query_model_utility.csv |
| probe_cost | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_cost.csv |
| predicted_gain | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_predicted_gain.csv |
| metadata | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_policy_input_metadata.json |
