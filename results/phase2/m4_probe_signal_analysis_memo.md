# Phase 2 Probe Signal Analysis

Command:

```bash
python experiments/53_probe_signal_analysis.py --probe-features results/phase2/aligned_offline/aligned_probe_features.parquet --output-dir results/phase2 --state-targets results/phase2/aligned_offline/aligned_state_targets.csv --query-features results/phase2/aligned_offline/aligned_query_features.csv
```

M4 executed on aligned probe features and route-state targets.

Outputs:

- `table_probe_signal_analysis.csv`
- `fig_probe_signal_gain.pdf`
- `m4_probe_signal_analysis_memo.md`

Summary:

| method | status | n_queries | n_train | n_test | state_prediction_accuracy | state_prediction_accuracy_ci_low | state_prediction_accuracy_ci_high | routing_utility | observability_gap_closed | mean_probe_cost_proxy | regret_prediction_auc | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| query_only_state_predictor | executed | 2318 | 1738 | 580 | 0.8724 | 0.8431 | 0.8992 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
| probe_only_state_predictor | executed | 2318 | 1738 | 580 | 0.3603 | 0.3224 | 0.4017 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
| query_plus_probe_state_predictor | executed | 2318 | 1738 | 580 | 0.8793 | 0.8534 | 0.9061 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
| query_plus_knn_uncertainty_state_predictor | executed | 2318 | 1738 | 580 | 0.8724 | 0.8448 | 0.9000 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
| query_plus_confidence_state_predictor | executed | 2318 | 1738 | 580 | 0.8724 | 0.8448 | 0.9000 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
