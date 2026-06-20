# Predicted Utility-State Calibration Experiment

This experiment tests whether RouteCode states become better calibration strata when the state is learned from utility vectors on train and then predicted from cheap observable probe features.

## Commands

- `PYTHONPATH=src python experiments/240_phase3_predicted_utility_state_calibration.py --config configs/probecode_final_eval.yaml`

## Inputs

- Outcome matrix: `results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet`
- Probe features: `results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv`
- No fresh model calls are made by this script.

## Method

1. Learn KMeans states from train query utility vectors only.
2. Train a RandomForest or ExtraTrees predictor from observable probe features to those train utility states.
3. Select the deployable strata variant using validation within-state utility variance.
4. Select the onboarding variant using train-to-validation utility estimation error.
5. Report held-out test strata quality and simulated held-out-model onboarding.

Selected strata method: `predicted_utility_state_rf_probe_only_k24`.
Selected onboarding method: `predicted_utility_state_rf_probe_plus_benchmark_k6`.

## State Prediction Diagnostics

- `test`: adjusted Rand `0.5076`, raw cluster accuracy `0.7647`
- `train`: adjusted Rand `0.6828`, raw cluster accuracy `0.8476`
- `val`: adjusted Rand `0.5658`, raw cluster accuracy `0.7763`

Top validation state-prediction variants by adjusted Rand:

- `predicted_utility_state_extratrees_probe_only_k6`: ARI `0.5767`
- `predicted_utility_state_rf_probe_plus_benchmark_k6`: ARI `0.5658`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k6`: ARI `0.5588`
- `predicted_utility_state_rf_probe_only_k6`: ARI `0.5575`
- `predicted_utility_state_rf_probe_plus_benchmark_k8`: ARI `0.5532`

## Validation Onboarding-State Selection

- `predicted_utility_state_rf_probe_plus_benchmark_k6`: train-to-val utility estimation error `0.0732`
- `predicted_utility_state_rf_probe_only_k6`: train-to-val utility estimation error `0.0743`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k6`: train-to-val utility estimation error `0.0754`
- `predicted_utility_state_extratrees_probe_only_k6`: train-to-val utility estimation error `0.0801`
- `predicted_utility_state_extratrees_probe_only_k8`: train-to-val utility estimation error `0.0903`
- `predicted_utility_state_rf_probe_plus_benchmark_k8`: train-to-val utility estimation error `0.0919`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k8`: train-to-val utility estimation error `0.0923`
- `predicted_utility_state_rf_probe_only_k8`: train-to-val utility estimation error `0.0939`

## Test Calibration-Strata Results

- `utility_cluster_k8_diagnostic`: utility variance `0.0861`, groups `8`
- `predicted_utility_state_rf_probe_plus_benchmark_k8`: utility variance `0.1213`, groups `8`
- `predicted_utility_state_rf_probe_only_k8`: utility variance `0.1241`, groups `8`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k24`: utility variance `0.1255`, groups `24`
- `predicted_utility_state_rf_probe_plus_benchmark_k24`: utility variance `0.1267`, groups `23`
- `predicted_utility_state_rf_probe_only_k24`: utility variance `0.1308`, groups `23`
- `predicted_utility_state_rf_probe_plus_benchmark_k16`: utility variance `0.1311`, groups `16`
- `predicted_utility_state_extratrees_probe_only_k24`: utility variance `0.1314`, groups `24`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k16`: utility variance `0.1318`, groups `16`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k8`: utility variance `0.1327`, groups `8`
- `predicted_utility_state_extratrees_probe_only_k16`: utility variance `0.1333`, groups `16`
- `predicted_utility_state_rf_probe_only_k16`: utility variance `0.1334`, groups `16`

Selected predicted state test variance: `0.1308`.

## Estimation And Best-Model Checks

- `utility_cluster_k8_diagnostic`: weighted abs utility estimation error `0.0466`
- `predicted_utility_state_rf_probe_only_k6`: weighted abs utility estimation error `0.0604`
- `predicted_utility_state_rf_probe_plus_benchmark_k6`: weighted abs utility estimation error `0.0635`
- `predicted_utility_state_rf_probe_plus_benchmark_k8`: weighted abs utility estimation error `0.0655`
- `predicted_utility_state_rf_probe_only_k8`: weighted abs utility estimation error `0.0702`
- `routecode_state_k8`: weighted abs utility estimation error `0.0709`
- `predicted_utility_state_extratrees_probe_only_k6`: weighted abs utility estimation error `0.0741`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k6`: weighted abs utility estimation error `0.0764`

- `calibration_aware_routecode_state_k8`: traffic-weighted best-model match `0.9186`
- `predicted_utility_state_rf_probe_only_k6`: traffic-weighted best-model match `0.8198`
- `predicted_utility_state_rf_probe_plus_benchmark_k6`: traffic-weighted best-model match `0.8198`
- `utility_cluster_k8_diagnostic`: traffic-weighted best-model match `0.7412`
- `routecode_state_k8`: traffic-weighted best-model match `0.6802`
- `predicted_utility_state_rf_probe_only_k8`: traffic-weighted best-model match `0.6628`
- `predicted_utility_state_extratrees_probe_only_k16`: traffic-weighted best-model match `0.6279`
- `predicted_utility_state_extratrees_probe_plus_benchmark_k6`: traffic-weighted best-model match `0.6221`

## Onboarding Results At Budget 320

- `active_predicted_utility_state`: utility `0.6953`, quality `0.7474`, evals `320.0`
- `random_query_predicted_utility_state`: utility `0.6945`, quality `0.7485`, evals `320.0`
- `uniform_predicted_utility_state`: utility `0.6909`, quality `0.7448`, evals `320.0`
- `direct_probe_regressor_retrain`: utility `0.6658`, quality `0.7283`, evals `320.0`

## Claim Status

- `predicted_states_as_calibration_strata`: `supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_only_k24;test_variance=0.1308;best_label_or_text=0.1596
- `active_acquisition_advantage`: `weakly_supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;active=0.6953;random=0.6945;uniform=0.6909;margin=0.0008
- `predicted_state_new_model_onboarding`: `supported_on_cached_broad100`; selected=predicted_utility_state_rf_probe_plus_benchmark_k6;budget=320;best_state=0.6953;direct_retrain_proxy=0.6658;state_minus_direct=0.0295

## Caveats

- This is cached Broad100 evidence, not a fresh GPT/Gemini calibration deployment.
- Utility-cluster labels are learned from train outcomes, but validation/test assignments here are predicted from observable cached probe features.
- The onboarding method is selected on validation estimation error, not test onboarding utility.
- If the active acquisition row only narrowly beats random/uniform, the claim should be written as weak evidence, not a strong 3-5x sample-efficiency result.
