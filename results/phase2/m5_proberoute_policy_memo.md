# Phase 2 ProbeRoute++ Policy

Command:

```bash
python experiments/54_proberoute_policy.py --output-dir results/phase2 --before-beliefs results/phase2/aligned_offline/aligned_before_beliefs.csv --after-beliefs results/phase2/aligned_offline/aligned_after_beliefs.csv --state-model-utility results/phase2/aligned_offline/aligned_state_model_utility.csv --query-model-utility results/phase2/aligned_offline/aligned_query_model_utility.csv --probe-cost results/phase2/aligned_offline/aligned_probe_cost.csv --predicted-gain results/phase2/aligned_offline/aligned_predicted_gain.csv
```

M5 executed ProbeRoute++ policies through latent state beliefs with probe-cost accounting.

Outputs:

- `table_proberoute_policy.csv`
- `fig_gap_closed_vs_probe_cost.pdf`
- `m5_proberoute_policy_memo.md`

Summary:

| policy | status | n_queries | mean_utility | mean_utility_ci_low | mean_utility_ci_high | mean_net_utility | mean_net_utility_ci_low | mean_net_utility_ci_high | mean_quality | mean_model_cost | mean_probe_cost_proxy | fraction_probed | mean_oracle_regret | observability_gap_closed | observability_gap_closed_ci_low | observability_gap_closed_ci_high | mean_latency_sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7447 | 0.7120 | 0.7792 |  |  | 0.0001 | 1.0000 | 0.1518 | 0.0106 | -0.2029 | 0.2353 | 0.0000 | Routed through state belief and state-model utility. |
| entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7448 | 0.7069 | 0.7785 |  |  | 0.0000 | 0.2121 | 0.1517 | 0.0111 | -0.2361 | 0.2305 | 0.0000 | Routed through state belief and state-model utility. |
| margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7103 | 0.7810 |  |  | 0.0000 | 0.0845 | 0.1517 | 0.0112 | -0.2135 | 0.2471 | 0.0000 | Routed through state belief and state-model utility. |
| voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7060 | 0.7785 |  |  | 0.0000 | 0.3138 | 0.1518 | 0.0110 | -0.2421 | 0.2304 | 0.0000 | Routed through state belief and state-model utility. |
| oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7103 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2140 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |
