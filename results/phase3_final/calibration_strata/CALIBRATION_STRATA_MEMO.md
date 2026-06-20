# Calibration Strata Experiment

This no-call experiment tests whether learned states are useful groups for estimating model utility.

## Inputs

- Outcome matrix: `results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet`
- RouteCode assignments: `results/controlled/broad100_learned_verifiability_probe_state/table_learned_verifiability_assignments.csv`
- Compact state method: `gb_depth2_thr0.9844_state_k8`

## Test Within-State Utility Variance

- `utility_cluster_k8_diagnostic`: variance `0.0861`, groups `8`
- `benchmark_label`: variance `0.1596`, groups `9`
- `calibration_aware_routecode_state_k8`: variance `0.1697`, groups `11`
- `routecode_state_k8`: variance `0.1809`, groups `6`
- `text_cluster_k8`: variance `0.1835`, groups `8`
- `random_k8`: variance `0.2475`, groups `8`

## New-Model Estimation Error Proxy

- `utility_cluster_k8_diagnostic`: weighted abs utility error `0.0466`
- `routecode_state_k8`: weighted abs utility error `0.0709`
- `calibration_aware_routecode_state_k8`: weighted abs utility error `0.0808`
- `random_k8`: weighted abs utility error `0.0899`
- `benchmark_label`: weighted abs utility error `0.0993`
- `text_cluster_k8`: weighted abs utility error `0.1144`

## Best-Model Identification

- `calibration_aware_routecode_state_k8`: traffic-weighted match `0.9186`
- `utility_cluster_k8_diagnostic`: traffic-weighted match `0.7412`
- `routecode_state_k8`: traffic-weighted match `0.6802`
- `benchmark_label`: traffic-weighted match `0.5349`
- `text_cluster_k8`: traffic-weighted match `0.4535`
- `random_k8`: traffic-weighted match `0.4186`

## Caveat

`utility_cluster_*` is diagnostic because assigning validation/test queries to a utility cluster uses hidden outcome vectors. The deployable learned-state rows are the RouteCode/probe-state rows.
