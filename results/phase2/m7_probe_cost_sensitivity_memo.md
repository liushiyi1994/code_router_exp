# Phase 2 Probe Cost Sensitivity

Command:

```bash
python experiments/63_probe_cost_sensitivity.py --output-dir results/phase2 --before-beliefs results/phase2/aligned_offline/aligned_before_beliefs.csv --after-beliefs results/phase2/aligned_offline/aligned_after_beliefs.csv --state-model-utility results/phase2/aligned_offline/aligned_state_model_utility.csv --query-model-utility results/phase2/aligned_offline/aligned_query_model_utility.csv --probe-cost results/phase2/aligned_offline/aligned_probe_cost.csv --predicted-gain results/phase2/aligned_offline/aligned_predicted_gain.csv --probe-cost-multipliers 0.0,0.5,1.0,2.0,5.0,10.0,50.0,100.0
```

This sweeps probe-cost multipliers for the same state-mediated ProbeRoute++ policy inputs.

Outputs:

- `table_probe_cost_sensitivity.csv`
- `table_probe_cost_sensitivity_summary.csv`
- `fig_probe_cost_sensitivity.pdf`
- `m7_probe_cost_sensitivity_memo.md`

Summary:

| probe_cost_multiplier | n_queries | best_policy_by_mean_net_utility | best_mean_net_utility | never_probe_mean_net_utility | never_probe_fraction_probed | never_probe_gap_closed | always_probe_mean_net_utility | always_probe_fraction_probed | always_probe_gap_closed | entropy_threshold_mean_net_utility | entropy_threshold_fraction_probed | entropy_threshold_gap_closed | margin_threshold_mean_net_utility | margin_threshold_fraction_probed | margin_threshold_gap_closed | voi_probe_mean_net_utility | voi_probe_fraction_probed | voi_probe_gap_closed | oracle_probe_mean_net_utility | oracle_probe_fraction_probed | oracle_probe_gap_closed | voi_minus_never_mean_net_utility | voi_minus_best_threshold_mean_net_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 580 | always_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7448 | 1.0000 | 0.0112 | 0.7448 | 0.2121 | 0.0112 | 0.7448 | 0.0845 | 0.0112 | 0.7448 | 0.5086 | 0.0112 | 0.7448 | 0.0017 | 0.0112 | 0.0017 | 0.0000 |
| 0.5000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7448 | 1.0000 | 0.0109 | 0.7448 | 0.2121 | 0.0112 | 0.7448 | 0.0845 | 0.0112 | 0.7448 | 0.3466 | 0.0111 | 0.7448 | 0.0017 | 0.0112 | 0.0017 | -0.0000 |
| 1.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7447 | 1.0000 | 0.0106 | 0.7448 | 0.2121 | 0.0111 | 0.7448 | 0.0845 | 0.0112 | 0.7448 | 0.3138 | 0.0110 | 0.7448 | 0.0017 | 0.0112 | 0.0017 | -0.0000 |
| 2.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7446 | 1.0000 | 0.0099 | 0.7448 | 0.2121 | 0.0110 | 0.7448 | 0.0845 | 0.0111 | 0.7448 | 0.2776 | 0.0109 | 0.7448 | 0.0017 | 0.0112 | 0.0017 | -0.0000 |
| 5.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7443 | 1.0000 | 0.0080 | 0.7447 | 0.2121 | 0.0105 | 0.7448 | 0.0845 | 0.0110 | 0.7447 | 0.2241 | 0.0105 | 0.7448 | 0.0017 | 0.0112 | 0.0016 | -0.0001 |
| 10.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7438 | 1.0000 | 0.0047 | 0.7446 | 0.2121 | 0.0099 | 0.7447 | 0.0845 | 0.0107 | 0.7446 | 0.2052 | 0.0099 | 0.7448 | 0.0017 | 0.0112 | 0.0015 | -0.0001 |
| 50.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7398 | 1.0000 | -0.0213 | 0.7438 | 0.2121 | 0.0043 | 0.7444 | 0.0845 | 0.0085 | 0.7443 | 0.1000 | 0.0080 | 0.7448 | 0.0017 | 0.0112 | 0.0012 | -0.0001 |
| 100.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7348 | 1.0000 | -0.0539 | 0.7427 | 0.2121 | -0.0026 | 0.7440 | 0.0845 | 0.0057 | 0.7442 | 0.0655 | 0.0070 | 0.7448 | 0.0017 | 0.0111 | 0.0011 | 0.0002 |

Policy Rows:

| probe_cost_multiplier | policy | status | n_queries | mean_utility | mean_utility_ci_low | mean_utility_ci_high | mean_net_utility | mean_net_utility_ci_low | mean_net_utility_ci_high | mean_quality | mean_model_cost | mean_probe_cost_proxy | fraction_probed | mean_oracle_regret | observability_gap_closed | observability_gap_closed_ci_low | observability_gap_closed_ci_high | mean_latency_sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7448 | 0.7121 | 0.7793 |  |  | 0.0000 | 1.0000 | 0.1517 | 0.0112 | -0.2022 | 0.2360 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7448 | 0.7069 | 0.7785 |  |  | 0.0000 | 0.2121 | 0.1517 | 0.0112 | -0.2360 | 0.2306 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7103 | 0.7810 |  |  | 0.0000 | 0.0845 | 0.1517 | 0.0112 | -0.2135 | 0.2472 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7060 | 0.7785 |  |  | 0.0000 | 0.5086 | 0.1517 | 0.0112 | -0.2419 | 0.2306 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7103 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2140 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7448 | 0.7120 | 0.7793 |  |  | 0.0001 | 1.0000 | 0.1518 | 0.0109 | -0.2026 | 0.2356 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7448 | 0.7069 | 0.7785 |  |  | 0.0000 | 0.2121 | 0.1517 | 0.0112 | -0.2360 | 0.2305 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7103 | 0.7810 |  |  | 0.0000 | 0.0845 | 0.1517 | 0.0112 | -0.2135 | 0.2472 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7060 | 0.7785 |  |  | 0.0000 | 0.3466 | 0.1517 | 0.0111 | -0.2420 | 0.2305 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7103 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2140 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7447 | 0.7120 | 0.7792 |  |  | 0.0001 | 1.0000 | 0.1518 | 0.0106 | -0.2029 | 0.2353 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7448 | 0.7069 | 0.7785 |  |  | 0.0000 | 0.2121 | 0.1517 | 0.0111 | -0.2361 | 0.2305 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7103 | 0.7810 |  |  | 0.0000 | 0.0845 | 0.1517 | 0.0112 | -0.2135 | 0.2471 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7060 | 0.7785 |  |  | 0.0000 | 0.3138 | 0.1518 | 0.0110 | -0.2421 | 0.2304 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7103 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2140 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |
| 2.0000 | never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| 2.0000 | always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7446 | 0.7119 | 0.7791 |  |  | 0.0002 | 1.0000 | 0.1519 | 0.0099 | -0.2036 | 0.2347 | 0.0000 | Routed through state belief and state-model utility. |
| 2.0000 | entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7448 | 0.7069 | 0.7784 |  |  | 0.0000 | 0.2121 | 0.1518 | 0.0110 | -0.2363 | 0.2303 | 0.0000 | Routed through state belief and state-model utility. |
| 2.0000 | margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7103 | 0.7810 |  |  | 0.0000 | 0.0845 | 0.1517 | 0.0111 | -0.2136 | 0.2471 | 0.0000 | Routed through state belief and state-model utility. |
| 2.0000 | voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7059 | 0.7784 |  |  | 0.0001 | 0.2776 | 0.1518 | 0.0109 | -0.2422 | 0.2303 | 0.0000 | Routed through state belief and state-model utility. |
| 2.0000 | oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7103 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2140 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |
| 5.0000 | never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| 5.0000 | always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7443 | 0.7116 | 0.7788 |  |  | 0.0005 | 1.0000 | 0.1522 | 0.0080 | -0.2055 | 0.2327 | 0.0000 | Routed through state belief and state-model utility. |
| 5.0000 | entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7447 | 0.7068 | 0.7784 |  |  | 0.0001 | 0.2121 | 0.1518 | 0.0105 | -0.2367 | 0.2299 | 0.0000 | Routed through state belief and state-model utility. |
| 5.0000 | margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7103 | 0.7810 |  |  | 0.0000 | 0.0845 | 0.1518 | 0.0110 | -0.2138 | 0.2469 | 0.0000 | Routed through state belief and state-model utility. |
| 5.0000 | voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7447 | 0.7059 | 0.7784 |  |  | 0.0001 | 0.2241 | 0.1518 | 0.0105 | -0.2426 | 0.2299 | 0.0000 | Routed through state belief and state-model utility. |
| 5.0000 | oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7103 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2141 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |
| 10.0000 | never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| 10.0000 | always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7438 | 0.7111 | 0.7783 |  |  | 0.0010 | 1.0000 | 0.1527 | 0.0047 | -0.2088 | 0.2294 | 0.0000 | Routed through state belief and state-model utility. |
| 10.0000 | entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7446 | 0.7067 | 0.7783 |  |  | 0.0002 | 0.2121 | 0.1519 | 0.0099 | -0.2374 | 0.2292 | 0.0000 | Routed through state belief and state-model utility. |
| 10.0000 | margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7447 | 0.7103 | 0.7809 |  |  | 0.0001 | 0.0845 | 0.1518 | 0.0107 | -0.2141 | 0.2466 | 0.0000 | Routed through state belief and state-model utility. |
| 10.0000 | voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7446 | 0.7058 | 0.7783 |  |  | 0.0002 | 0.2052 | 0.1519 | 0.0099 | -0.2433 | 0.2293 | 0.0000 | Routed through state belief and state-model utility. |
| 10.0000 | oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7103 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2141 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |
| 50.0000 | never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| 50.0000 | always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7398 | 0.7071 | 0.7743 |  |  | 0.0050 | 1.0000 | 0.1567 | -0.0213 | -0.2348 | 0.2034 | 0.0000 | Routed through state belief and state-model utility. |
| 50.0000 | entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7438 | 0.7058 | 0.7774 |  |  | 0.0011 | 0.2121 | 0.1528 | 0.0043 | -0.2434 | 0.2237 | 0.0000 | Routed through state belief and state-model utility. |
| 50.0000 | margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7444 | 0.7099 | 0.7806 |  |  | 0.0004 | 0.0845 | 0.1521 | 0.0085 | -0.2163 | 0.2441 | 0.0000 | Routed through state belief and state-model utility. |
| 50.0000 | voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7443 | 0.7055 | 0.7779 |  |  | 0.0005 | 0.1000 | 0.1522 | 0.0080 | -0.2451 | 0.2270 | 0.0000 | Routed through state belief and state-model utility. |
| 50.0000 | oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7102 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2142 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |
| 100.0000 | never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| 100.0000 | always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7348 | 0.7021 | 0.7693 |  |  | 0.0100 | 1.0000 | 0.1617 | -0.0539 | -0.2674 | 0.1708 | 0.0000 | Routed through state belief and state-model utility. |
| 100.0000 | entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7427 | 0.7046 | 0.7764 |  |  | 0.0021 | 0.2121 | 0.1538 | -0.0026 | -0.2508 | 0.2168 | 0.0000 | Routed through state belief and state-model utility. |
| 100.0000 | margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7440 | 0.7095 | 0.7801 |  |  | 0.0008 | 0.0845 | 0.1526 | 0.0057 | -0.2192 | 0.2411 | 0.0000 | Routed through state belief and state-model utility. |
| 100.0000 | voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7442 | 0.7054 | 0.7778 |  |  | 0.0007 | 0.0655 | 0.1524 | 0.0070 | -0.2457 | 0.2261 | 0.0000 | Routed through state belief and state-model utility. |
| 100.0000 | oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7102 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0111 | -0.2143 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |

Interpretation:

Across probe-cost multipliers, VOI minus the best threshold policy has mean net-utility delta `-0.0000` over `8` settings (`1` positive, `6` negative, `1` tied).
