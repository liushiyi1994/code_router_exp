# RouteCode Phase 2 Results

## Phase 2 Probe Cost Sensitivity

Command:

```bash
python experiments/63_probe_cost_sensitivity.py --output-dir results/phase2/true_probe_policy_cost_sensitivity_vllm_qwen3_4b_all200 --before-beliefs results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_before_beliefs.csv --after-beliefs results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_after_beliefs.csv --state-model-utility results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_state_model_utility.csv --query-model-utility results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_query_model_utility.csv --probe-cost results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_cost.csv --predicted-gain results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_predicted_gain.csv --probe-cost-multipliers 0.0,0.01,0.05,0.1,0.25,0.5,1.0
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
| 0.0100 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8512 | 1.0000 | -0.0336 | 0.8526 | 0.4146 | -0.0149 | 0.8535 | 0.0976 | -0.0028 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| 0.0500 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8414 | 1.0000 | -0.1678 | 0.8482 | 0.4146 | -0.0747 | 0.8526 | 0.0976 | -0.0138 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
| 0.1000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8291 | 1.0000 | -0.3356 | 0.8427 | 0.4146 | -0.1494 | 0.8516 | 0.0976 | -0.0277 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0020 |
| 0.2500 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.7923 | 1.0000 | -0.8391 | 0.8263 | 0.4146 | -0.3735 | 0.8486 | 0.0976 | -0.0692 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0051 |
| 0.5000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.7309 | 1.0000 | -1.6782 | 0.7990 | 0.4146 | -0.7470 | 0.8435 | 0.0976 | -0.1385 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0101 |
| 1.0000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.6081 | 1.0000 | -3.3564 | 0.7443 | 0.4146 | -1.4940 | 0.8334 | 0.0976 | -0.2770 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |

Interpretation:

Across probe-cost multipliers, VOI minus the best threshold policy has mean net-utility delta `0.0055` over `7` settings (`6` positive, `0` negative, `1` tied).
