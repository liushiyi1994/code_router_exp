# vLLM Probe Variant Comparison

All variants use Qwen3-4B served by vLLM on the all200 exact manifest.

| probe_variant | n_aligned_queries | n_test | query_only_state_accuracy | query_plus_probe_state_accuracy | state_accuracy_delta | query_plus_probe_ci_low | query_plus_probe_ci_high | mean_probe_cost_proxy | selected_model_changes | policy_rows | mean_before_utility | mean_after_utility | mean_utility_delta | nonzero_predicted_gain_rows | never_probe_mean_net_utility | voi_probe_mean_net_utility | voi_probe_fraction_probed | always_probe_mean_net_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confidence_probe | 164 | 41 | 0.6098 | 0.6341 | 0.0244 | 0.4878 | 0.7805 | 0.2266 | 2 | 41 | 0.8537 | 0.8537 | 0.0000 | 0 | 0.8537 | 0.8537 | 0.0000 | 0.6081 |
| answer_output_probe | 164 | 41 | 0.6098 | 0.6585 | 0.0488 | 0.5122 | 0.7805 | 0.0770 | 1 | 41 | 0.8537 | 0.8537 | 0.0000 | 0 | 0.8537 | 0.8537 | 0.0000 | 0.7727 |
| combined_probe | 164 | 41 | 0.6098 | 0.6341 | 0.0244 | 0.4878 | 0.7805 | 0.1518 | 1 | 41 | 0.8537 | 0.8537 | 0.0000 | 0 | 0.8537 | 0.8537 | 0.0000 | 0.6904 |
