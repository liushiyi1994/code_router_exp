# Probe Action Observability

This diagnostic checks whether a probe changes latent-state beliefs in ways that change the final selected model.

| n_queries | top_state_changes | top_state_change_rate | top_state_action_changes | selected_model_changes | selected_model_change_rate | action_equivalent_top_state_changes | before_oracle_model_match_rate | after_oracle_model_match_rate | mean_before_utility | mean_after_utility | mean_utility_delta | mean_before_regret | mean_after_regret | mean_regret_delta | mean_belief_l1_shift | mean_belief_top_prob_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 41.0000 | 1.0000 | 0.0244 | 1.0000 | 2.0000 | 0.0488 | 1.0000 | 0.4634 | 0.5122 | 0.8537 | 0.8537 | 0.0000 | 0.0732 | 0.0732 | 0.0000 | 0.1025 | -0.0031 |

Top-state changes: `1/41`; selected-model changes: `2/41`.
Action-equivalent top-state changes: `1`.

Outputs:

| artifact | path |
| --- | --- |
| summary | results/phase2/probe_action_observability_vllm_qwen3_4b_all200/table_probe_action_observability.csv |
| by_query | results/phase2/probe_action_observability_vllm_qwen3_4b_all200/table_probe_action_observability_by_query.csv |

Rows with selected-model changes:

| query_id | before_top_state | after_top_state | top_state_changed | before_top_state_action | after_top_state_action | top_state_action_changed | before_selected_model | after_selected_model | selected_model_changed | action_equivalent_top_state_change | oracle_selected_model | before_matches_oracle_model | after_matches_oracle_model | before_utility | after_utility | oracle_utility | utility_delta | before_regret | after_regret | regret_delta | belief_l1_shift | belief_top_prob_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math500:test:8 | z4 | z4 | False | MiniCPM4.1-8B | MiniCPM4.1-8B | False | MiniCPM4.1-8B | Qwen3-8B | True | False | Qwen3-8B | False | True | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1186 | -0.0306 |
| math500:test:85 | z11 | z11 | False | MiniCPM4.1-8B | MiniCPM4.1-8B | False | MiniCPM4.1-8B | Qwen3-8B | True | False | Qwen3-8B | False | True | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4554 | -0.2234 |
