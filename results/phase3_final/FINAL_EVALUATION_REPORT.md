# Phase 3 Final Evaluation Report

This report is the current Phase 3 final evaluation package. The main routing, strata, onboarding, sensitivity, and ablation sections are cache-backed; the live-calibration section records local vLLM smoke calls and provider-readiness checks.

## Main Broad100 Result

- Method: `et_flip_leaf4_thr0.8502_capNone`
- Mean quality: `0.8547`
- Mean utility: `0.8238`
- Quality gap to oracle: `0.0174`
- Oracle utility ratio: `0.9735`
- Frontier-call rate: `0.1919`

## Literature Baseline Status

- `routellm_mf`: `broad100_adapter_executed`
- `graphrouter`: `fallback_broad100_adapter_executed`
- `avengerspro`: `broad100_adapter_executed`

## Calibration Strata

- `utility_cluster_k8_diagnostic`: test utility variance `0.0861`
- `benchmark_label`: test utility variance `0.1596`
- `calibration_aware_routecode_state_k8`: test utility variance `0.1697`
- `routecode_state_k8`: test utility variance `0.1809`
- `text_cluster_k8`: test utility variance `0.1835`
- `random_k8`: test utility variance `0.2475`

## New-Model Onboarding

- `calibration_aware_route_state` at budget `640`: utility `0.7395`
- `active_route_state_calibration` at budget `640`: utility `0.6723`
- `uniform_route_state_calibration` at budget `640`: utility `0.6723`
- `random_query_route_state_calibration` at budget `640`: utility `0.6723`
- `dataset_stratified_calibration` at budget `640`: utility `0.6651`
- `direct_probe_regressor_retrain` at budget `640`: utility `0.6637`
- `embedding_cluster_calibration` at budget `640`: utility `0.6041`

## Frozen State vs Retrain

- `frozen_state_calibration_aware`: best utility `0.7395` at budget `640`
- `random_query_state_calibration`: best utility `0.6752` at budget `160`
- `frozen_state_active`: best utility `0.6741` at budget `320`
- `frozen_state_uniform`: best utility `0.6739` at budget `320`
- `direct_router_retrain_proxy`: best utility `0.6699` at budget `160`
- `dataset_stratified_calibration`: best utility `0.6651` at budget `640`
- `embedding_cluster_calibration`: best utility `0.6041` at budget `640`

## Cost Sensitivity

- lambda `0.00`, price x`0.5`: utility `0.7791`, frontier rate `0.9186`
- lambda `0.00`, price x`1.0`: utility `0.7791`, frontier rate `0.9186`
- lambda `0.00`, price x`2.0`: utility `0.7791`, frontier rate `0.9186`
- lambda `0.00`, price x`5.0`: utility `0.7791`, frontier rate `0.9186`
- lambda `0.10`, price x`0.5`: utility `0.7724`, frontier rate `0.6163`
- lambda `0.10`, price x`1.0`: utility `0.7569`, frontier rate `0.5523`
- lambda `0.10`, price x`2.0`: utility `0.7231`, frontier rate `0.5523`
- lambda `0.10`, price x`5.0`: utility `0.6670`, frontier rate `0.5523`
- lambda `0.35`, price x`0.5`: utility `0.7315`, frontier rate `0.5523`
- lambda `0.35`, price x`1.0`: utility `0.6723`, frontier rate `0.5523`
- lambda `0.35`, price x`2.0`: utility `0.6818`, frontier rate `0.3198`
- lambda `0.35`, price x`5.0`: utility `0.6667`, frontier rate `0.3198`

## Ablation

- `full_method`: utility `0.8238`, delta vs full `0.0000`
- `compact_routecode_state_policy`: utility `0.8136`, delta vs full `-0.0103`
- `large_only_action_pool_diagnostic`: utility `0.7689`, delta vs full `-0.0550`
- `no_verifiable_tool_actions`: utility `0.7360`, delta vs full `-0.0878`
- `local_only_action_pool_diagnostic`: utility `0.6919`, delta vs full `-0.1320`
- `random_query_calibration_budget160`: utility `0.6752`, delta vs full `-0.1486`
- `direct_probe_action_predictor_no_state`: utility `0.6733`, delta vs full `-0.1505`
- `active_state_calibration_budget160`: utility `0.6731`, delta vs full `-0.1507`
- `no_active_calibration_uniform_budget160`: utility `0.6728`, delta vs full `-0.1510`
- `no_probe_local_behavior_features`: utility `0.6652`, delta vs full `-0.1586`
- `direct_action_probability_no_state`: utility `0.6208`, delta vs full `-0.2030`

## Real Local/Frontier Calibration

- `historical_live_broad100_qwen32_thinking_mcq` / `gemini-3.5-flash`: status `skipped`, calls `200`, mean quality `nan`
- `historical_live_broad100_qwen32_thinking_mcq` / `gpt-5.5`: status `skipped`, calls `200`, mean quality `nan`
- `historical_live_broad100_qwen32_thinking_mcq` / `qwen3-32b-awq-thinking-local`: status `success`, calls `200`, mean quality `0.0450`
- `current_live_qwen06_smoke` / `qwen3-0.6b-probe-live-smoke`: status `success`, calls `8`, mean quality `0.0000`
- `frontier_live_call_readiness` / `gpt-5.5`: status `blocked_no_api_key`, calls `0`, mean quality `nan`
- `frontier_live_call_readiness` / `gemini-3.5-flash`: status `blocked_no_api_key`, calls `0`, mean quality `nan`

## Predicted Utility-State Calibration Update

This update learns utility states on train and predicts them from observable cached probe features. It is the current best evidence for calibration strata and state-based new-model onboarding.

- `predicted_states_as_calibration_strata`: `supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_only_k24;test_variance=0.1308;best_label_or_text=0.1596
- `active_acquisition_advantage`: `weakly_supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;active=0.6953;random=0.6945;uniform=0.6909;margin=0.0008
- `predicted_state_new_model_onboarding`: `supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;best_state=0.6953;direct_retrain_proxy=0.6658;state_minus_direct=0.0295

Key held-out test strata variance rows:

- `predicted_utility_state_rf_probe_only_k24`: `0.1308`
- `benchmark_label`: `0.1596`
- `text_cluster_k8`: `0.1835`

Key onboarding rows at budget 320:

- `active_predicted_utility_state`: utility `0.6953`, quality `0.7474`
- `random_query_predicted_utility_state`: utility `0.6945`, quality `0.7485`
- `uniform_predicted_utility_state`: utility `0.6909`, quality `0.7448`
- `direct_probe_regressor_retrain`: utility `0.6658`, quality `0.7283`

## Live Broad100 Predicted Utility-State Update

This update runs the same predicted utility-state logic on the live Stage0 matrix with GPT, Gemini, and local vLLM rows.

- `live_predicted_states_as_calibration_strata`: `supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k16;test_variance=0.1366;best_label_or_text=0.1666
- `live_predicted_state_new_model_onboarding`: `weakly_supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;best_state=0.5656;direct_retrain_proxy=0.5644;state_minus_direct=0.0012
- `live_active_acquisition_advantage`: `weakly_supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;active=0.5656;random=0.5640;uniform=0.5650;margin=0.0006
- `live_frontier_active_onboarding_low_budget`: `supported_on_live_broad100_stage0`; validation_selected_budget=40;test_active=0.5627;test_best_competitor=0.5510;margin=0.0117;heldout_models=gpt-5.5,gemini-3.5-flash
- `live_frontier_budget_efficiency`: `supported_on_live_broad100_stage0`; active_budget=40;direct_eval_reduction_lower_bound=8.0x;random_eval_reduction_lower_bound=8.0x;target_active_utility=0.5627

Live held-out test strata variance rows:

- `predicted_utility_state_rf_probe_plus_benchmark_k16`: `0.1366`
- `benchmark_label`: `0.1666`
- `text_cluster_k8`: `0.1923`

Live onboarding rows at budget 320:

- `active_predicted_utility_state` / `validation_selected:traffic_active`: utility `0.5656`, quality `0.6483`
- `uniform_predicted_utility_state` / `uniform_group`: utility `0.5650`, quality `0.6476`
- `direct_probe_regressor_retrain` / `direct_regressor`: utility `0.5644`, quality `0.5869`
- `random_query_predicted_utility_state` / `random_query`: utility `0.5640`, quality `0.6466`

Live frontier onboarding slice:

- Validation-selected budget `40`: active validation margin `0.0118`
- Test at budget `40`: active `0.5627`, best competitor `0.5510`, margin `0.0117`

Live frontier budget-to-match:

- `uniform` matches at `80` evals; active uses `40` evals; reduction lower bound `2.0x`
- `random` matches at `320` evals; active uses `40` evals; reduction lower bound `8.0x`
- `direct` does not match by `320` evals; active uses `40` evals; reduction lower bound `8.0x`

## Claim Table

- `final_main_broad100_oracle_level`: `supported_on_cached_broad100`; quality_gap=0.0174;utility_ratio=0.9735;frontier_rate=0.1919
- `final_literature_baseline_coverage`: `complete`; pending_or_not_included=0;total=3
- `states_as_calibration_strata`: `not_supported_on_cached_broad100`; best_routecode_test_variance=0.1697;best_label_or_text_variance=0.1596
- `state_calibration_new_model_onboarding`: `not_supported_on_cached_broad100`; budget=640;active_utility=0.6723;random_utility=0.6723
- `frozen_state_vs_direct_retrain_proxy`: `table_generated`; rows=42
- `state_action_table_price_adaptation`: `table_generated`; rows=240
- `final_ablation_coverage`: `table_generated`; rows=11
- `real_local_frontier_new_model_calibration`: `partial_local_live_only`; local_success_rows=2;frontier_blocked_rows=2;best_live_local_quality=0.0450
- `predicted_states_as_calibration_strata`: `supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_only_k24;test_variance=0.1308;best_label_or_text=0.1596
- `active_acquisition_advantage`: `weakly_supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;active=0.6953;random=0.6945;uniform=0.6909;margin=0.0008
- `predicted_state_new_model_onboarding`: `supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;best_state=0.6953;direct_retrain_proxy=0.6658;state_minus_direct=0.0295
- `live_predicted_states_as_calibration_strata`: `supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k16;test_variance=0.1366;best_label_or_text=0.1666
- `live_predicted_state_new_model_onboarding`: `weakly_supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;best_state=0.5656;direct_retrain_proxy=0.5644;state_minus_direct=0.0012
- `live_active_acquisition_advantage`: `weakly_supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;active=0.5656;random=0.5640;uniform=0.5650;margin=0.0006
- `live_frontier_active_onboarding_low_budget`: `supported_on_live_broad100_stage0`; validation_selected_budget=40;test_active=0.5627;test_best_competitor=0.5510;margin=0.0117;heldout_models=gpt-5.5,gemini-3.5-flash
- `live_frontier_budget_efficiency`: `supported_on_live_broad100_stage0`; active_budget=40;direct_eval_reduction_lower_bound=8.0x;random_eval_reduction_lower_bound=8.0x;target_active_utility=0.5627
- `phase3_broad100_current_best_oracle_level_target`: `supported`; quality_gap=0.0174;utility_ratio=0.9735;frontier_rate=0.1919
- `phase3_broad100_routecode_state_policy_target`: `supported_with_lower_utility`; quality_gap=0.0233;utility_ratio=0.9614;frontier_rate=0.2384
- `phase3_no_tool_full_oracle_target`: `not_supported_feasibility_bound`; no_tool_oracle_quality_gap_to_full=0.0465;no_tool_oracle_utility_ratio_to_full=0.9338
- `phase3_exact_math_controlled_targets`: `supported`; quality_gap=0.0152;utility_ratio=0.9739;frontier_rate=0.1061;normalized_remote_cost=0.0463;p95_latency_ratio_vs_all_gpt=0.4799
- `phase3_state_level_new_model_calibration`: `supported_on_cached_exact_math`; active_evals=4;active_quality=0.8485;direct_best_quality=0.7273
- `phase3_budget_and_model_constraints`: `supported`; top_level_max_model_cost=4.9765;broad_stage0_max_model_cost=0.2512
- `phase3_controlled_verifiability_action_pool_scope`: `supported`; broad100_quality_gap=0.0174;broad100_utility_ratio=0.9735;exact_quality_gap=0.0152;active_evals=4;top_level_max_model_cost=4.9765

## Remaining Optional Or Follow-Up Work

- Approved GPT/Gemini live calibration calls with API keys, budget, token logging, and refreshed pricing.
