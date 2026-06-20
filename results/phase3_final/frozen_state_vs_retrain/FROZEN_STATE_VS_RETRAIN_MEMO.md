# Frozen State Router vs Direct Router Retraining

Source onboarding table: `results/phase3_final/new_model_onboarding/table_new_model_onboarding.csv`

This is a cache-backed proxy comparison. The direct-router row is a probe-feature utility regressor retrained with the same new-model budget.

## Best Mean Utility By Family

- `frozen_state_calibration_aware`: utility `0.7395` at budget `640`, delta vs direct same budget `0.0758`
- `random_query_state_calibration`: utility `0.6752` at budget `160`, delta vs direct same budget `0.0053`
- `frozen_state_active`: utility `0.6741` at budget `320`, delta vs direct same budget `0.0083`
- `frozen_state_uniform`: utility `0.6739` at budget `320`, delta vs direct same budget `0.0081`
- `direct_router_retrain_proxy`: utility `0.6699` at budget `160`, delta vs direct same budget `0.0000`
- `dataset_stratified_calibration`: utility `0.6651` at budget `640`, delta vs direct same budget `0.0014`
- `embedding_cluster_calibration`: utility `0.6041` at budget `640`, delta vs direct same budget `-0.0596`

## Budget To Match Direct At Max Budget

- `frozen_state_calibration_aware`: `20`; best utility `0.7395`
- `random_query_state_calibration`: `40`; best utility `0.6752`
- `frozen_state_active`: `80`; best utility `0.6741`
- `frozen_state_uniform`: `80`; best utility `0.6739`
- `direct_router_retrain_proxy`: `40`; best utility `0.6699`
- `dataset_stratified_calibration`: `640`; best utility `0.6651`
- `embedding_cluster_calibration`: `not matched`; best utility `0.6041`

## Caveat

This does not replace a full learned-router baseline on live data. It is the first no-call evidence for whether frozen states can avoid full retraining.
