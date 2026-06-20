# True Local Probe Decision Value

This decision value diagnostic measures whether true local probe features change the latent-state routing decision before probe-cost accounting.

Summary:

| n_queries | selected_model_changes | selected_model_change_rate | mean_before_utility | mean_after_utility | mean_utility_delta | positive_utility_delta_rows | negative_utility_delta_rows | mean_predicted_gain | nonzero_predicted_gain_rows | mean_probe_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 41.0000 | 2.0000 | 0.0488 | 0.8537 | 0.8537 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2456 |

The selected model changed on `2/41` held-out utility rows.
Mean realized utility moved from `0.8537` to `0.8537`.
Nonzero predicted-gain rows: `0/41`.

Outputs:

| artifact | path |
| --- | --- |
| summary | results/phase2/true_probe_decision_value_vllm_qwen3_4b_all200/table_true_probe_decision_value.csv |
| by_query | results/phase2/true_probe_decision_value_vllm_qwen3_4b_all200/table_true_probe_decision_value_by_query.csv |

Changed-decision rows:

| query_id | before_selected_model | after_selected_model | selected_changed | before_utility | after_utility | utility_delta | predicted_gain | probe_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math500:test:8 | MiniCPM4.1-8B | Qwen3-8B | True | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1750 |
| math500:test:85 | MiniCPM4.1-8B | Qwen3-8B | True | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3708 |
