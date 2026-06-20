# RouteCode Phase 2 Results

## Phase 2 ProbeRoute++ Policy

M5 executed ProbeRoute++ policies through latent state beliefs with probe-cost accounting.

Outputs:

- `table_proberoute_policy.csv`
- `fig_gap_closed_vs_probe_cost.pdf`
- `m5_proberoute_policy_memo.md`

| policy | status | n_queries | mean_utility | mean_utility_ci_low | mean_utility_ci_high | mean_net_utility | mean_net_utility_ci_low | mean_net_utility_ci_high | mean_quality | mean_model_cost | mean_probe_cost_proxy | fraction_probed | mean_oracle_regret | observability_gap_closed | observability_gap_closed_ci_low | observability_gap_closed_ci_high | mean_latency_sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| never_probe | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| always_probe | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 1.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| entropy_threshold | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| margin_threshold | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| voi_probe | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
| oracle_probe | executed | 41 | 0.9024 | 0.8049 | 0.9756 | 0.9024 | 0.8049 | 0.9756 |  |  | 0.0000 | 0.0000 | 0.0244 | 0.0000 | -4.0000 | 3.0000 | 0.0000 | Routed through state belief and state-model utility. |
