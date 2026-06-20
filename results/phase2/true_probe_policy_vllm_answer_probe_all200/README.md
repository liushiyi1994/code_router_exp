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
| always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.7727 | 0.6529 | 0.8716 |  |  | 0.0810 | 1.0000 | 0.1541 | -1.1065 | -2.7440 | 0.2448 | 0.0000 | Routed through state belief and state-model utility. |
| entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8206 | 0.7003 | 0.9219 |  |  | 0.0331 | 0.4146 | 0.1062 | -0.4520 | -2.0957 | 0.9323 | 0.0000 | Routed through state belief and state-model utility. |
| margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8464 | 0.7241 | 0.9416 |  |  | 0.0072 | 0.0976 | 0.0804 | -0.0990 | -1.7702 | 1.2017 | 0.0000 | Routed through state belief and state-model utility. |
| voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
