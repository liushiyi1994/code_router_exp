# RouteCode Phase 2 Results

## Phase 2 ProbeRoute++ Policy

M5 executed ProbeRoute++ policies through latent state beliefs with probe-cost accounting.

Outputs:

- `table_proberoute_policy.csv`
- `fig_gap_closed_vs_probe_cost.pdf`
- `m5_proberoute_policy_memo.md`

| policy | status | n_queries | mean_utility | mean_utility_ci_low | mean_utility_ci_high | mean_net_utility | mean_net_utility_ci_low | mean_net_utility_ci_high | mean_quality | mean_model_cost | mean_probe_cost_proxy | fraction_probed | mean_oracle_regret | observability_gap_closed | observability_gap_closed_ci_low | observability_gap_closed_ci_high | mean_latency_sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | -0.0590 | -0.2600 | 0.1017 |  |  | 0.9126 | 1.0000 | 0.9858 | -12.4728 | -15.2203 | -10.2770 | 0.0000 | Routed through state belief and state-model utility. |
| entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.4616 | 0.1746 | 0.6573 |  |  | 0.3921 | 0.4146 | 0.4652 | -5.3583 | -9.2799 | -2.6829 | 0.0000 | Routed through state belief and state-model utility. |
| margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.7778 | 0.6611 | 0.8814 |  |  | 0.0758 | 0.0976 | 0.1490 | -1.0365 | -2.6314 | 0.3795 | 0.0000 | Routed through state belief and state-model utility. |
| voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
