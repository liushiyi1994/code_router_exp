# Probe-Observable Action State

This experiment evaluates a 1-bit RouteCode state learned from the cost-aware
utility matrix:

```text
local_enough vs frontier_needed
```

The state learner uses the utility matrix to assign train labels and build a
train-only state-to-model table. The deployable predictor uses query semantics
and cached local-model behavior only. No gold answer, quality score, utility
value, benchmark label, or frontier output is used as a predictor feature.

Command:

```bash
PYTHONPATH=src python experiments/248_phase3_probe_observable_action_state.py \
  --outputs results/phase3_final/live_predicted_utility_states/live_outputs_with_splits_and_utility.parquet \
  --output-dir results/phase3_probe_observable_action_state \
  --embedding-model BAAI/bge-small-en-v1.5
```

State counts:

```text
state_label
local_enough       718
frontier_needed     96
```

Train state-to-model table:

```text
{'frontier_needed': 'gpt-5.5', 'local_enough': 'qwen3-32b-awq-local'}
```

Target hit: `True`

## Best Test Rows

| method | state accuracy | balanced accuracy | mean utility | oracle utility ratio | frontier call rate | probe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true_action_state_oracle | 1.0000 | 1.0000 | 0.5597 | 0.7650 | 0.1176 | 0.0000 |
| always_probe::semantic_pca8_plus_probe::histgb | 0.9118 | 0.7117 | 0.5037 | 0.6885 | 0.0765 | 1.0000 |
| always_probe::probe_only::histgb | 0.9059 | 0.7083 | 0.4997 | 0.6830 | 0.0824 | 1.0000 |
| always_probe::probe_only::extratrees | 0.9000 | 0.6833 | 0.4974 | 0.6798 | 0.0765 | 1.0000 |
| always_probe::semantic_pca8_plus_probe::extratrees | 0.9000 | 0.6617 | 0.4945 | 0.6759 | 0.0647 | 1.0000 |
| active_probe::rf::probe_only::extratrees::rate_0.2 | 0.8941 | 0.6150 | 0.4895 | 0.6690 | 0.0471 | 0.1176 |
| active_probe::rf::probe_only::extratrees::rate_0.3 | 0.8941 | 0.6150 | 0.4895 | 0.6690 | 0.0471 | 0.1176 |
| active_probe::rf::probe_only::extratrees::rate_0.5 | 0.8941 | 0.6150 | 0.4895 | 0.6690 | 0.0471 | 0.1176 |
| active_probe::rf::probe_only::extratrees::rate_0.75 | 0.8941 | 0.6150 | 0.4895 | 0.6690 | 0.0471 | 0.1176 |
| active_probe::rf::probe_only::extratrees::rate_1 | 0.8941 | 0.6150 | 0.4895 | 0.6690 | 0.0471 | 0.1176 |
| active_probe::extratrees::probe_only::rf::rate_0.2 | 0.8941 | 0.6150 | 0.4875 | 0.6664 | 0.0471 | 0.1471 |
| active_probe::extratrees::probe_only::rf::rate_0.3 | 0.8941 | 0.6150 | 0.4875 | 0.6664 | 0.0471 | 0.1471 |
| active_probe::extratrees::probe_only::rf::rate_0.5 | 0.8941 | 0.6150 | 0.4875 | 0.6664 | 0.0471 | 0.1471 |
| active_probe::extratrees::probe_only::rf::rate_0.75 | 0.8941 | 0.6150 | 0.4875 | 0.6664 | 0.0471 | 0.1471 |
| active_probe::extratrees::probe_only::rf::rate_1 | 0.8941 | 0.6150 | 0.4875 | 0.6664 | 0.0471 | 0.1471 |
| active_probe::extratrees::probe_only::rf::rate_0.1 | 0.8941 | 0.5717 | 0.4855 | 0.6635 | 0.0235 | 0.0412 |
| active_probe::logreg::probe_only::extratrees::rate_0.5 | 0.8882 | 0.6333 | 0.4878 | 0.6667 | 0.0647 | 0.2824 |
| active_probe::logreg::probe_only::extratrees::rate_0.75 | 0.8882 | 0.6333 | 0.4878 | 0.6667 | 0.0647 | 0.2824 |
| active_probe::logreg::probe_only::extratrees::rate_1 | 0.8882 | 0.6333 | 0.4878 | 0.6667 | 0.0647 | 0.2824 |
| active_probe::rf::semantic_pca8_plus_probe::extratrees::rate_0.2 | 0.8882 | 0.6117 | 0.4860 | 0.6643 | 0.0529 | 0.1176 |

Artifacts:

- `table_probe_observable_action_state.csv`
- `table_probe_observable_action_features.csv`
- `table_probe_observable_action_assignments.csv`
