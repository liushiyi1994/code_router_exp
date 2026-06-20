# Phase 2 ProbeRoute++ Policy

Command:

```bash
python experiments/54_proberoute_policy.py --output-dir results/phase2/routecode_target_rate_policy_vllm_all200 --before-beliefs results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_before_beliefs.csv --after-beliefs results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_after_beliefs.csv --state-model-utility results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_state_model_utility.csv --query-model-utility results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_query_model_utility.csv --probe-cost results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_cost.csv --predicted-gain results/phase2/routecode_target_rate_policy_inputs_vllm_all200/true_probe_predicted_gain.csv
```

M5 executed ProbeRoute++ policies through latent state beliefs with probe-cost accounting.

Outputs:

- `table_proberoute_policy.csv`
- `fig_gap_closed_vs_probe_cost.pdf`
- `m5_proberoute_policy_memo.md`

Summary:

| policy | status | n_queries | mean_utility | mean_utility_ci_low | mean_utility_ci_high | mean_net_utility | mean_net_utility_ci_low | mean_net_utility_ci_high | mean_quality | mean_model_cost | mean_probe_cost_proxy | fraction_probed | mean_oracle_regret | observability_gap_closed | observability_gap_closed_ci_low | observability_gap_closed_ci_high | mean_latency_sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| never_probe | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| always_probe | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 1.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| entropy_threshold | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| margin_threshold | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| voi_probe | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| oracle_probe | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
