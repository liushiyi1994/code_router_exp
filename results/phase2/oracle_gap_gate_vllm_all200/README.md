# RouteCode Phase 2 Oracle-Gap Gate

## Oracle-Gap Gate

Threshold: relative gap to query oracle <= `0.0300`.

Best row: `dataset_model_rule:exact_math_qwen_intern` with mean utility `0.9268` versus oracle `0.9268` (relative gap `0.0000`).
Best deployable row: `dataset_model_rule:exact_math_qwen_intern` with mean utility `0.9268` versus oracle `0.9268` (relative gap `0.0000`).
Best current Phase 2 policy row: `target_rate_routecode:always_probe` with relative gap `0.0263`.

Important: rows marked `deployable = False` are diagnostic upper bounds. Rows whose `selection_basis` says `policy_slice` are useful as candidates, but should be validated on a held-out selection protocol before being reported as final.

| candidate | candidate_type | selection_basis | deployable | n_queries | mean_utility | oracle_mean_utility | abs_gap_to_oracle | relative_gap_to_oracle | threshold | within_threshold | regret_count | notes | val_relative_gap_to_oracle | test_relative_gap_to_oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_model_rule:exact_math_qwen_intern | benchmark_label_route_rule | targeted_exact_math_benchmark_label_rule | True | 41 | 0.9268 | 0.9268 | 0.0000 | 0.0000 | 0.0300 | True | 0.0000 | Benchmark-label route rule: query metadata -> route label -> selected model. | 0.0833 | 0.0265 |
| routecode_state_oracle_upper:k32:alpha0 | routecode_state_oracle_upper | policy_slice_candidate_not_val_selected; diagnostic_uses_eval_utility_for_label_assignment | False | 41 | 0.9268 | 0.9268 | 0.0000 | 0.0000 | 0.0300 | True | 0.0000 | Diagnostic upper bound: assigns labels using eval utility, not deployable. | 0.2471 | 0.2500 |
| target_rate_routecode:always_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True |  | Existing M5 policy row with probe-cost accounting. |  |  |
| target_rate_routecode:entropy_threshold | proberoute_policy | current_phase2_policy_table | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True |  | Existing M5 policy row with probe-cost accounting. |  |  |
| target_rate_routecode:margin_threshold | proberoute_policy | current_phase2_policy_table | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True |  | Existing M5 policy row with probe-cost accounting. |  |  |
| target_rate_routecode:never_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True |  | Existing M5 policy row with probe-cost accounting. |  |  |
| target_rate_routecode:oracle_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True |  | Existing M5 policy row with probe-cost accounting. |  |  |
| target_rate_routecode:voi_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True |  | Existing M5 policy row with probe-cost accounting. |  |  |
| routecode_embedding_predicted:k32:alpha0 | routecode_embedding_centroid | policy_slice_candidate_not_val_selected | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True | 1.0000 | Train-fit RouteCode labels predicted from query embeddings. | 0.2471 | 0.2500 |
| target_rate_routecode_inputs:after_belief_expected | belief_expected_action | current_phase2_policy_inputs | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True | 1.0000 | Expected model utility under predicted state belief. |  |  |
| target_rate_routecode_inputs:after_hard_top_state | hard_top_state_action | current_phase2_policy_inputs | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True | 1.0000 | Best action for top predicted state only; no belief averaging. |  |  |
| target_rate_routecode_inputs:before_belief_expected | belief_expected_action | current_phase2_policy_inputs | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True | 1.0000 | Expected model utility under predicted state belief. |  |  |
| target_rate_routecode_inputs:before_hard_top_state | hard_top_state_action | current_phase2_policy_inputs | True | 41 | 0.9024 | 0.9268 | 0.0244 | 0.0263 | 0.0300 | True | 1.0000 | Best action for top predicted state only; no belief averaging. |  |  |
| answer_policy:never_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| answer_policy:oracle_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| answer_policy:voi_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| combined_policy:never_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| combined_policy:oracle_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| combined_policy:voi_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| confidence_policy:never_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| confidence_policy:oracle_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| confidence_policy:voi_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| answer_inputs:after_belief_expected | belief_expected_action | current_phase2_policy_inputs | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False | 3.0000 | Expected model utility under predicted state belief. |  |  |
| answer_inputs:before_belief_expected | belief_expected_action | current_phase2_policy_inputs | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False | 3.0000 | Expected model utility under predicted state belief. |  |  |
| combined_inputs:after_belief_expected | belief_expected_action | current_phase2_policy_inputs | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False | 3.0000 | Expected model utility under predicted state belief. |  |  |
| combined_inputs:before_belief_expected | belief_expected_action | current_phase2_policy_inputs | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False | 3.0000 | Expected model utility under predicted state belief. |  |  |
| confidence_inputs:after_belief_expected | belief_expected_action | current_phase2_policy_inputs | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False | 3.0000 | Expected model utility under predicted state belief. |  |  |
| confidence_inputs:before_belief_expected | belief_expected_action | current_phase2_policy_inputs | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False | 3.0000 | Expected model utility under predicted state belief. |  |  |
| routecode_embedding_predicted:k16:alpha3 | routecode_embedding_centroid | current_phase2_d2_config | True | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False | 3.0000 | Train-fit RouteCode labels predicted from query embeddings. | 0.1467 | 0.1673 |
| routecode_state_oracle_upper:k16:alpha3 | routecode_state_oracle_upper | current_phase2_d2_config; diagnostic_uses_eval_utility_for_label_assignment | False | 41 | 0.8537 | 0.9268 | 0.0732 | 0.0789 | 0.0300 | False | 3.0000 | Diagnostic upper bound: assigns labels using eval utility, not deployable. | 0.1467 | 0.1673 |
| answer_policy:margin_threshold | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8464 | 0.9268 | 0.0804 | 0.0868 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| combined_policy:margin_threshold | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8399 | 0.9268 | 0.0869 | 0.0938 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| confidence_policy:margin_threshold | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8334 | 0.9268 | 0.0934 | 0.1008 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| answer_inputs:after_hard_top_state | hard_top_state_action | current_phase2_policy_inputs | True | 41 | 0.8293 | 0.9268 | 0.0976 | 0.1053 | 0.0300 | False | 4.0000 | Best action for top predicted state only; no belief averaging. |  |  |
| answer_inputs:before_hard_top_state | hard_top_state_action | current_phase2_policy_inputs | True | 41 | 0.8293 | 0.9268 | 0.0976 | 0.1053 | 0.0300 | False | 4.0000 | Best action for top predicted state only; no belief averaging. |  |  |
| combined_inputs:after_hard_top_state | hard_top_state_action | current_phase2_policy_inputs | True | 41 | 0.8293 | 0.9268 | 0.0976 | 0.1053 | 0.0300 | False | 4.0000 | Best action for top predicted state only; no belief averaging. |  |  |
| combined_inputs:before_hard_top_state | hard_top_state_action | current_phase2_policy_inputs | True | 41 | 0.8293 | 0.9268 | 0.0976 | 0.1053 | 0.0300 | False | 4.0000 | Best action for top predicted state only; no belief averaging. |  |  |
| confidence_inputs:after_hard_top_state | hard_top_state_action | current_phase2_policy_inputs | True | 41 | 0.8293 | 0.9268 | 0.0976 | 0.1053 | 0.0300 | False | 4.0000 | Best action for top predicted state only; no belief averaging. |  |  |
| confidence_inputs:before_hard_top_state | hard_top_state_action | current_phase2_policy_inputs | True | 41 | 0.8293 | 0.9268 | 0.0976 | 0.1053 | 0.0300 | False | 4.0000 | Best action for top predicted state only; no belief averaging. |  |  |
| answer_policy:entropy_threshold | proberoute_policy | current_phase2_policy_table | True | 41 | 0.8206 | 0.9268 | 0.1062 | 0.1146 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| combined_policy:entropy_threshold | proberoute_policy | current_phase2_policy_table | True | 41 | 0.7825 | 0.9268 | 0.1444 | 0.1558 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| answer_policy:always_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.7727 | 0.9268 | 0.1541 | 0.1663 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| confidence_policy:entropy_threshold | proberoute_policy | current_phase2_policy_table | True | 41 | 0.7443 | 0.9268 | 0.1825 | 0.1969 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| combined_policy:always_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.6904 | 0.9268 | 0.2364 | 0.2551 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
| confidence_policy:always_probe | proberoute_policy | current_phase2_policy_table | True | 41 | 0.6081 | 0.9268 | 0.3188 | 0.3439 | 0.0300 | False |  | Existing M5 policy row with probe-cost accounting. |  |  |
