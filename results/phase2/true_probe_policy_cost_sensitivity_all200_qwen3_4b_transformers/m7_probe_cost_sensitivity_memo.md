# Phase 2 Probe Cost Sensitivity

Command:

```bash
python experiments/63_probe_cost_sensitivity.py --output-dir results/phase2/true_probe_policy_cost_sensitivity_all200_qwen3_4b_transformers --before-beliefs results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_before_beliefs.csv --after-beliefs results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_after_beliefs.csv --state-model-utility results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_state_model_utility.csv --query-model-utility results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_query_model_utility.csv --probe-cost results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_cost.csv --predicted-gain results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_predicted_gain.csv --probe-cost-multipliers 0.0,0.01,0.05,0.1,0.25,0.5,1.0
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
| 0.0000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 1.0000 | 0.0000 | 0.8537 | 0.4146 | 0.0000 | 0.8537 | 0.0976 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0100 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8445 | 1.0000 | -0.1247 | 0.8497 | 0.4146 | -0.0536 | 0.8529 | 0.0976 | -0.0104 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0008 |
| 0.0500 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8080 | 1.0000 | -0.6236 | 0.8341 | 0.4146 | -0.2679 | 0.8499 | 0.0976 | -0.0518 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0038 |
| 0.1000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.7624 | 1.0000 | -1.2473 | 0.8145 | 0.4146 | -0.5358 | 0.8461 | 0.0976 | -0.1036 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0076 |
| 0.2500 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.6255 | 1.0000 | -3.1182 | 0.7556 | 0.4146 | -1.3396 | 0.8347 | 0.0976 | -0.2591 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0190 |
| 0.5000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.3973 | 1.0000 | -6.2364 | 0.6576 | 0.4146 | -2.6791 | 0.8157 | 0.0976 | -0.5182 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0379 |
| 1.0000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | -0.0590 | 1.0000 | -12.4728 | 0.4616 | 0.4146 | -5.3583 | 0.7778 | 0.0976 | -1.0365 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0758 |

Policy Rows:

| probe_cost_multiplier | policy | status | n_queries | mean_utility | mean_utility_ci_low | mean_utility_ci_high | mean_net_utility | mean_net_utility_ci_low | mean_net_utility_ci_high | mean_quality | mean_model_cost | mean_probe_cost_proxy | fraction_probed | mean_oracle_regret | observability_gap_closed | observability_gap_closed_ci_low | observability_gap_closed_ci_high | mean_latency_sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 1.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.4146 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0976 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0000 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.8445 | 0.7226 | 0.9425 |  |  | 0.0091 | 1.0000 | 0.0823 | -0.1247 | -1.7916 | 1.2140 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8497 | 0.7279 | 0.9482 |  |  | 0.0039 | 0.4146 | 0.0771 | -0.0536 | -1.7189 | 1.2915 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8529 | 0.7308 | 0.9501 |  |  | 0.0008 | 0.0976 | 0.0739 | -0.0104 | -1.6786 | 1.3177 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.8080 | 0.6860 | 0.9076 |  |  | 0.0456 | 1.0000 | 0.1188 | -0.6236 | -2.2912 | 0.7368 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8341 | 0.7126 | 0.9359 |  |  | 0.0196 | 0.4146 | 0.0928 | -0.2679 | -1.9278 | 1.1244 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8499 | 0.7273 | 0.9455 |  |  | 0.0038 | 0.0976 | 0.0770 | -0.0518 | -1.7262 | 1.2549 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.7624 | 0.6403 | 0.8639 |  |  | 0.0913 | 1.0000 | 0.1644 | -1.2473 | -2.9157 | 0.1403 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8145 | 0.6923 | 0.9206 |  |  | 0.0392 | 0.4146 | 0.1124 | -0.5358 | -2.2057 | 0.9154 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8461 | 0.7230 | 0.9397 |  |  | 0.0076 | 0.0976 | 0.0808 | -0.1036 | -1.7858 | 1.1765 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.6255 | 0.4972 | 0.7330 |  |  | 0.2282 | 1.0000 | 0.3013 | -3.1182 | -4.8720 | -1.6493 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.7556 | 0.6129 | 0.8760 |  |  | 0.0980 | 0.4146 | 0.1712 | -1.3396 | -3.2907 | 0.3055 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8347 | 0.7099 | 0.9225 |  |  | 0.0190 | 0.0976 | 0.0921 | -0.2591 | -1.9645 | 0.9413 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.3973 | 0.2478 | 0.5160 |  |  | 0.4563 | 1.0000 | 0.5295 | -6.2364 | -8.2799 | -4.6152 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.6576 | 0.4696 | 0.7983 |  |  | 0.1960 | 0.4146 | 0.2692 | -2.6791 | -5.2487 | -0.7564 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8157 | 0.6915 | 0.9051 |  |  | 0.0379 | 0.0976 | 0.1111 | -0.5182 | -2.2158 | 0.7034 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | -0.0590 | -0.2600 | 0.1017 |  |  | 0.9126 | 1.0000 | 0.9858 | -12.4728 | -15.2203 | -10.2770 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.4616 | 0.1746 | 0.6573 |  |  | 0.3921 | 0.4146 | 0.4652 | -5.3583 | -9.2799 | -2.6829 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.7778 | 0.6611 | 0.8814 |  |  | 0.0758 | 0.0976 | 0.1490 | -1.0365 | -2.6314 | 0.3795 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |

Interpretation:

Across probe-cost multipliers, VOI minus the best threshold policy has mean net-utility delta `0.0207` over `7` settings (`6` positive, `0` negative, `1` tied).
