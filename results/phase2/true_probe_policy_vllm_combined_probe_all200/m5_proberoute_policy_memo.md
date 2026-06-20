# Phase 2 ProbeRoute++ Policy

Command:

```bash
python experiments/54_proberoute_policy.py --output-dir results/phase2/true_probe_policy_vllm_combined_probe_all200 --before-beliefs results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_before_beliefs.csv --after-beliefs results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_after_beliefs.csv --state-model-utility results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_state_model_utility.csv --query-model-utility results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_query_model_utility.csv --probe-cost results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_cost.csv --predicted-gain results/phase2/true_probe_policy_inputs_vllm_combined_probe_all200/true_probe_predicted_gain.csv
```

M5 executed ProbeRoute++ policies through latent state beliefs with probe-cost accounting.

Outputs:

- `table_proberoute_policy.csv`
- `fig_gap_closed_vs_probe_cost.pdf`
- `m5_proberoute_policy_memo.md`

Summary:

| policy | status | n_queries | mean_utility | mean_utility_ci_low | mean_utility_ci_high | mean_net_utility | mean_net_utility_ci_low | mean_net_utility_ci_high | mean_quality | mean_model_cost | mean_probe_cost_proxy | fraction_probed | mean_oracle_regret | observability_gap_closed | observability_gap_closed_ci_low | observability_gap_closed_ci_high | mean_latency_sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.6904 | 0.5699 | 0.7920 |  |  | 0.1633 | 1.0000 | 0.2364 | -2.2315 | -3.8780 | -0.8426 | 0.0000 | Routed through state belief and state-model utility. |
| entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.7825 | 0.6564 | 0.8942 |  |  | 0.0712 | 0.4146 | 0.1444 | -0.9730 | -2.6955 | 0.5537 | 0.0000 | Routed through state belief and state-model utility. |
| margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8399 | 0.7159 | 0.9303 |  |  | 0.0138 | 0.0976 | 0.0869 | -0.1880 | -1.8822 | 1.0476 | 0.0000 | Routed through state belief and state-model utility. |
| voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
