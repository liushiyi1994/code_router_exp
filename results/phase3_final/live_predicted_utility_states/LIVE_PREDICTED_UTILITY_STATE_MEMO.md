# Live Broad100 Predicted Utility-State Calibration

This experiment reruns the predicted utility-state calibration/onboarding checks on the live Stage0 outcome matrix.

## Commands

- `PYTHONPATH=src python experiments/241_phase3_live_predicted_utility_state_calibration.py --config configs/probecode_final_eval.yaml`

## Inputs

- Live outputs: `results/controlled/live_broad100_stage0/model_outputs.parquet`
- Probe/split features: `results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv`
- Lambda cost: `0.35`

## Live Model Coverage

- `gpt-5.5`: selected, coverage `1.0000`, quality `0.6512`, cost `$6.2018`
- `gemini-3.5-flash`: selected, coverage `1.0000`, quality `0.3663`, cost `$0.4951`
- `qwen3-14b-awq-local`: selected, coverage `0.9988`, quality `0.4820`, cost `$0.0000`
- `qwen3-8b-local`: selected, coverage `0.9988`, quality `0.3260`, cost `$0.0000`
- `qwen3-32b-awq-local`: selected, coverage `0.9523`, quality `0.4982`, cost `$0.0000`
- `qwen3-4b-local`: selected, coverage `0.9465`, quality `0.3575`, cost `$0.0000`

Selected strata method: `predicted_utility_state_rf_probe_plus_benchmark_k16`.
Selected onboarding method: `predicted_utility_state_rf_probe_plus_benchmark_k6`.

## Held-Out Test Strata Variance

- `utility_cluster_k8_diagnostic`: utility variance `0.0667`
- `predicted_utility_state_rf_probe_plus_benchmark_k16`: utility variance `0.1366`
- `predicted_utility_state_rf_probe_plus_benchmark_k8`: utility variance `0.1433`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k16`: utility variance `0.1440`
- `predicted_utility_state_rf_probe_only_k16`: utility variance `0.1489`
- `predicted_utility_state_rf_probe_only_k8`: utility variance `0.1502`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k12`: utility variance `0.1548`
- `predicted_utility_state_extratrees_probe_only_k8`: utility variance `0.1550`
- `predicted_utility_state_extratrees_probe_only_k16`: utility variance `0.1554`
- `predicted_utility_state_rf_probe_plus_benchmark_k12`: utility variance `0.1564`

## Validation Onboarding-State Selection

- `predicted_utility_state_rf_probe_plus_benchmark_k6`: train-to-val utility estimation error `0.0853`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k4`: train-to-val utility estimation error `0.0892`
- `predicted_utility_state_rf_probe_only_k4`: train-to-val utility estimation error `0.0909`
- `predicted_utility_state_rf_probe_plus_benchmark_k4`: train-to-val utility estimation error `0.0927`
- `predicted_utility_state_rf_probe_only_k6`: train-to-val utility estimation error `0.0965`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k6`: train-to-val utility estimation error `0.0966`
- `predicted_utility_state_extratrees_probe_only_k4`: train-to-val utility estimation error `0.0966`
- `predicted_utility_state_extratrees_probe_only_k6`: train-to-val utility estimation error `0.0968`

## Validation Active-Acquisition Selection

- Budget `20`: `pilot_gain_active` selected with validation utility `0.5983`
- Budget `40`: `pilot_gain_active` selected with validation utility `0.6081`
- Budget `80`: `pilot_gain_active` selected with validation utility `0.6093`
- Budget `160`: `pilot_gain_active` selected with validation utility `0.6117`
- Budget `320`: `traffic_active` selected with validation utility `0.6120`

## Frontier Onboarding Slice

This slice treats GPT and Gemini as held-out new models and selects the calibration budget using validation frontier utility margin.

Validation-selected frontier budget: `40`.

- Validation budget `40`: active `0.6114`, best competitor `0.5996`, margin `0.0118`
- Validation budget `20`: active `0.5950`, best competitor `0.5915`, margin `0.0035`
- Validation budget `80`: active `0.6114`, best competitor `0.6114`, margin `0.0000`
- Validation budget `160`: active `0.6114`, best competitor `0.6114`, margin `0.0000`
- Validation budget `320`: active `0.6114`, best competitor `0.6114`, margin `0.0000`

Test at selected budget `40`: active `0.5627`, best competitor `0.5510`, margin `0.0117`.

Budget-to-match summary:
- `uniform` matches at `80` evals; active uses `40` evals; reduction lower bound `2.0x`
- `random` matches at `320` evals; active uses `40` evals; reduction lower bound `8.0x`
- `direct` does not match by `320` evals; active uses `40` evals; reduction lower bound `8.0x`

## Onboarding At Budget 320

- `active_predicted_utility_state`: utility `0.5656`, quality `0.6483`
- `uniform_predicted_utility_state`: utility `0.5650`, quality `0.6476`
- `direct_probe_regressor_retrain`: utility `0.5644`, quality `0.5869`
- `random_query_predicted_utility_state`: utility `0.5640`, quality `0.6466`

## Claim Status

- `live_predicted_states_as_calibration_strata`: `supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k16;test_variance=0.1366;best_label_or_text=0.1666
- `live_predicted_state_new_model_onboarding`: `weakly_supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;best_state=0.5656;direct_retrain_proxy=0.5644;state_minus_direct=0.0012
- `live_active_acquisition_advantage`: `weakly_supported_on_live_broad100_stage0`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;active=0.5656;random=0.5640;uniform=0.5650;margin=0.0006
- `live_frontier_active_onboarding_low_budget`: `supported_on_live_broad100_stage0`; validation_selected_budget=40;test_active=0.5627;test_best_competitor=0.5510;margin=0.0117;heldout_models=gpt-5.5,gemini-3.5-flash
- `live_frontier_budget_efficiency`: `supported_on_live_broad100_stage0`; active_budget=40;direct_eval_reduction_lower_bound=8.0x;random_eval_reduction_lower_bound=8.0x;target_active_utility=0.5627

## Caveats

- This is live/cached Stage0 evidence, not a new uncached full provider run.
- Gemini fresh retry hit rate limits in the separate provider-readiness run; the Stage0 Gemini rows here are reused from the existing cache.
- Active acquisition is reported separately because state-based calibration can succeed even if the acquisition rule still needs improvement.
