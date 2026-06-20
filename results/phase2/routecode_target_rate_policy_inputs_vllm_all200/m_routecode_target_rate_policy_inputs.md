# Target-Rate RouteCode Policy Inputs

Command:

```bash
PYTHONPATH=src python experiments/73_routecode_target_rate_policy_inputs.py --config configs/llmrouterbench_pilot.yaml --query-model-utility results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_query_model_utility.csv --output-dir results/phase2/routecode_target_rate_policy_inputs_vllm_all200 --k 32 --alpha 0.0
```

These inputs fit the RouteCode codebook on train only and predict one-hot route-state beliefs from query embeddings. They do not use held-out utility to assign labels.

Summary:

- K: `32`.
- Effective labels: `32`.
- Train rows: `1738`.
- Policy rows: `41`.

Outputs:

| artifact | path |
| --- | --- |
| before_beliefs | results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_before_beliefs.csv |
| after_beliefs | results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_after_beliefs.csv |
| state_model_utility | results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_state_model_utility.csv |
| query_model_utility | results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_query_model_utility.csv |
| probe_cost | results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_cost.csv |
| predicted_gain | results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_predicted_gain.csv |
| metadata | results/phase2/routecode_target_rate_policy_inputs_vllm_all200/routecode_target_rate_policy_input_metadata.json |
