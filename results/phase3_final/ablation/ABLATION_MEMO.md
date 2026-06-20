# Final Phase 3 Ablation

This is a cache-backed ablation consolidation for the Broad100 final package. It makes no model calls.

## Rows

- `full_method`: utility `0.8238`, quality `0.8547`, delta vs full `0.0000`; validation selected ProbeCode-StateCal current best
- `compact_routecode_state_policy`: utility `0.8136`, quality `0.8488`, delta vs full `-0.0103`; Uses compact learned RouteCode state policy without the residual flip that creates the final current-best method.
- `large_only_action_pool_diagnostic`: utility `0.7689`, quality `0.8140`, delta vs full `-0.0550`; Per-query large/frontier-style upper bound within the cached large action pool.
- `no_verifiable_tool_actions`: utility `0.7360`, quality `0.7674`, delta vs full `-0.0878`; Removes deterministic/verifiable tool-style local actions from the selected learned-verifiability package.
- `local_only_action_pool_diagnostic`: utility `0.6919`, quality `0.6919`, delta vs full `-0.1320`; Per-query local-only upper bound within the cached local/verifiable action pool.
- `random_query_calibration_budget160`: utility `0.6752`, quality `0.7847`, delta vs full `-0.1486`; Random query calibration at the same budget.
- `direct_probe_action_predictor_no_state`: utility `0.6733`, quality `0.7500`, delta vs full `-0.1505`; Direct utility predictor over probe features, without an explicit RouteCode state table.
- `active_state_calibration_budget160`: utility `0.6731`, quality `0.7819`, delta vs full `-0.1507`; Active state calibration at 160 new-model evaluations.
- `no_active_calibration_uniform_budget160`: utility `0.6728`, quality `0.7851`, delta vs full `-0.1510`; Uniform state calibration at the same budget.
- `no_probe_local_behavior_features`: utility `0.6652`, quality `0.7791`, delta vs full `-0.1586`; Uses train benchmark lookup only; this removes local probe-behavior features and is a diagnostic label baseline.
- `direct_action_probability_no_state`: utility `0.6208`, quality `0.6802`, delta vs full `-0.2030`; Decision-aware direct action predictor without discrete state abstraction.

## Interpretation

- Removing verifiable/tool-style actions causes a large utility drop in the cached Broad100 package.
- Direct no-state predictors remain well below the final current-best method.
- The onboarding rows show active calibration did not beat uniform/random in the cached Broad100 simulation at this budget.
