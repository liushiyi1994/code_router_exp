# Simulated New-Model Onboarding

This cache-only experiment treats each cached action as a held-out new model/action.

## Inputs

- Outcome matrix: `results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet`
- Probe features: `results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv`
- RouteCode state method: `gb_depth2_thr0.9844_state_k8`

## Mean Results At Budget 640

- `calibration_aware_route_state`: utility `0.7395`, regret to full `0.0000`, evals `510.6`
- `active_route_state_calibration`: utility `0.6723`, regret to full `0.0000`, evals `510.6`
- `uniform_route_state_calibration`: utility `0.6723`, regret to full `0.0000`, evals `510.6`
- `random_query_route_state_calibration`: utility `0.6723`, regret to full `0.0000`, evals `510.6`
- `dataset_stratified_calibration`: utility `0.6651`, regret to full `0.0000`, evals `510.6`
- `direct_probe_regressor_retrain`: utility `0.6637`, regret to full `0.0007`, evals `516.0`
- `embedding_cluster_calibration`: utility `0.6041`, regret to full `0.0000`, evals `510.6`

## Model-Type Coverage

- `cheap_local` / `calibration_aware_route_state`: utility `0.7395`
- `cheap_local` / `active_route_state_calibration`: utility `0.6723`
- `cheap_local` / `random_query_route_state_calibration`: utility `0.6723`
- `cheap_local` / `uniform_route_state_calibration`: utility `0.6723`
- `cheap_local` / `dataset_stratified_calibration`: utility `0.6645`
- `cheap_local` / `direct_probe_regressor_retrain`: utility `0.6637`
- `cheap_local` / `embedding_cluster_calibration`: utility `0.6035`
- `frontier` / `calibration_aware_route_state`: utility `0.7395`
- `frontier` / `active_route_state_calibration`: utility `0.6723`
- `frontier` / `random_query_route_state_calibration`: utility `0.6723`
- `frontier` / `uniform_route_state_calibration`: utility `0.6723`
- `frontier` / `dataset_stratified_calibration`: utility `0.6645`
- `frontier` / `direct_probe_regressor_retrain`: utility `0.6637`
- `frontier` / `embedding_cluster_calibration`: utility `0.6035`
- `medium_local` / `calibration_aware_route_state`: utility `0.7395`
- `medium_local` / `active_route_state_calibration`: utility `0.6723`
- `medium_local` / `random_query_route_state_calibration`: utility `0.6723`
- `medium_local` / `uniform_route_state_calibration`: utility `0.6723`
- `medium_local` / `dataset_stratified_calibration`: utility `0.6674`
- `medium_local` / `direct_probe_regressor_retrain`: utility `0.6637`

## Caveat

This is simulated from cached outcomes. It measures sample efficiency for state-table calibration, not live API deployment cost.
