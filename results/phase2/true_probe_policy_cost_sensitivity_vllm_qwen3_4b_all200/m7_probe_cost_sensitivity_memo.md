# Phase 2 Probe Cost Sensitivity

Command:

```bash
python experiments/63_probe_cost_sensitivity.py --output-dir results/phase2/true_probe_policy_cost_sensitivity_vllm_qwen3_4b_all200 --before-beliefs results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_before_beliefs.csv --after-beliefs results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_after_beliefs.csv --state-model-utility results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_state_model_utility.csv --query-model-utility results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_query_model_utility.csv --probe-cost results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_cost.csv --predicted-gain results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_predicted_gain.csv --probe-cost-multipliers 0.0,0.01,0.05,0.1,0.25,0.5,1.0
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
| 0.0100 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8512 | 1.0000 | -0.0336 | 0.8526 | 0.4146 | -0.0149 | 0.8535 | 0.0976 | -0.0028 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| 0.0500 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8414 | 1.0000 | -0.1678 | 0.8482 | 0.4146 | -0.0747 | 0.8526 | 0.0976 | -0.0138 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
| 0.1000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.8291 | 1.0000 | -0.3356 | 0.8427 | 0.4146 | -0.1494 | 0.8516 | 0.0976 | -0.0277 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0020 |
| 0.2500 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.7923 | 1.0000 | -0.8391 | 0.8263 | 0.4146 | -0.3735 | 0.8486 | 0.0976 | -0.0692 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0051 |
| 0.5000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.7309 | 1.0000 | -1.6782 | 0.7990 | 0.4146 | -0.7470 | 0.8435 | 0.0976 | -0.1385 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0101 |
| 1.0000 | 41 | never_probe | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.6081 | 1.0000 | -3.3564 | 0.7443 | 0.4146 | -1.4940 | 0.8334 | 0.0976 | -0.2770 | 0.8537 | 0.0000 | 0.0000 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |

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
| 0.0100 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.8512 | 0.7293 | 0.9489 |  |  | 0.0025 | 1.0000 | 0.0756 | -0.0336 | -1.6996 | 1.3012 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8526 | 0.7307 | 0.9503 |  |  | 0.0011 | 0.4146 | 0.0743 | -0.0149 | -1.6811 | 1.3213 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8535 | 0.7315 | 0.9509 |  |  | 0.0002 | 0.0976 | 0.0734 | -0.0028 | -1.6699 | 1.3291 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0100 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.8414 | 0.7197 | 0.9395 |  |  | 0.0123 | 1.0000 | 0.0855 | -0.1678 | -1.8312 | 1.1725 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8482 | 0.7264 | 0.9468 |  |  | 0.0055 | 0.4146 | 0.0786 | -0.0747 | -1.7387 | 1.2730 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8526 | 0.7305 | 0.9497 |  |  | 0.0010 | 0.0976 | 0.0742 | -0.0138 | -1.6827 | 1.3121 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.0500 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.8291 | 0.7076 | 0.9277 |  |  | 0.0246 | 1.0000 | 0.0977 | -0.3356 | -1.9958 | 1.0118 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8427 | 0.7212 | 0.9424 |  |  | 0.0109 | 0.4146 | 0.0841 | -0.1494 | -1.8108 | 1.2126 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8516 | 0.7294 | 0.9481 |  |  | 0.0020 | 0.0976 | 0.0752 | -0.0277 | -1.6986 | 1.2909 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.1000 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.7923 | 0.6715 | 0.8924 |  |  | 0.0614 | 1.0000 | 0.1346 | -0.8391 | -2.4895 | 0.5294 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8263 | 0.7053 | 0.9291 |  |  | 0.0273 | 0.4146 | 0.1005 | -0.3735 | -2.0271 | 1.0316 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8486 | 0.7259 | 0.9435 |  |  | 0.0051 | 0.0976 | 0.0782 | -0.0692 | -1.7466 | 1.2273 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.2500 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.7309 | 0.6113 | 0.8336 |  |  | 0.1228 | 1.0000 | 0.1960 | -1.6782 | -3.3123 | -0.2745 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.7990 | 0.6780 | 0.9071 |  |  | 0.0547 | 0.4146 | 0.1278 | -0.7470 | -2.4002 | 0.7298 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8435 | 0.7200 | 0.9357 |  |  | 0.0101 | 0.0976 | 0.0833 | -0.1385 | -1.8265 | 1.1213 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 0.5000 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | never_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | always_probe | executed | 41 | 0.8537 | 0.7189 | 0.9512 | 0.6081 | 0.4761 | 0.7134 |  |  | 0.2456 | 1.0000 | 0.3188 | -3.3564 | -5.1606 | -1.9164 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | entropy_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.7443 | 0.6011 | 0.8630 |  |  | 0.1093 | 0.4146 | 0.1825 | -1.4940 | -3.4518 | 0.1283 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | margin_threshold | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8334 | 0.7083 | 0.9220 |  |  | 0.0203 | 0.0976 | 0.0934 | -0.2770 | -1.9864 | 0.9334 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | voi_probe | executed | 41 | 0.8537 | 0.7433 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |
| 1.0000 | oracle_probe | executed | 41 | 0.8537 | 0.7561 | 0.9512 | 0.8537 | 0.7317 | 0.9512 |  |  | 0.0000 | 0.0000 | 0.0732 | 0.0000 | -1.6667 | 1.3333 | 0.0000 | Routed through state belief and state-model utility. |

Interpretation:

Across probe-cost multipliers, VOI minus the best threshold policy has mean net-utility delta `0.0055` over `7` settings (`6` positive, `0` negative, `1` tied).
