# RouteCode Phase 2 Results

## Phase 2 Probe Cost Sensitivity

Command:

```bash
python experiments/63_probe_cost_sensitivity.py --output-dir results/phase2/true_probe_policy_cost_sensitivity_all200_qwen3_4b_transformers --before-beliefs results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_before_beliefs.csv --after-beliefs results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_after_beliefs.csv --state-model-utility results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_state_model_utility.csv --query-model-utility results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_query_model_utility.csv --probe-cost results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_cost.csv --predicted-gain results/phase2/true_probe_policy_inputs_all200_qwen3_4b_transformers/true_probe_predicted_gain.csv --probe-cost-multipliers 0.0,0.01,0.05,0.1,0.25,0.5,1.0
```

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

Interpretation:

Across probe-cost multipliers, VOI minus the best threshold policy has mean net-utility delta `0.0207` over `7` settings (`6` positive, `0` negative, `1` tied).
