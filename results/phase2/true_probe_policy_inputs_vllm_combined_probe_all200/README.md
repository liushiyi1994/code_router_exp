# RouteCode Phase 2 Results

## True Local Probe Policy Inputs

Creates before/after latent route-state beliefs from true local probe features for M5. This preserves the Phase 2 invariant: query/probe -> belief over latent route states -> selected model.

- Train rows: `123`.
- Policy rows: `41`.
- Route states: `16`.

| artifact | path |
| --- | --- |
| before_beliefs | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_before_beliefs.csv |
| after_beliefs | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_after_beliefs.csv |
| state_model_utility | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_state_model_utility.csv |
| query_model_utility | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_query_model_utility.csv |
| probe_cost | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_cost.csv |
| predicted_gain | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_predicted_gain.csv |
| metadata | results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_policy_input_metadata.json |
