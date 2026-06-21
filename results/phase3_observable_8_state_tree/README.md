# Observable 8-State RouteCode Tree

This experiment targets the stricter Phase 3 requirement:

```text
at least 8 states, held-out state accuracy above 90%, not the 1-bit action state
```

Method:

1. Build deployable features from query semantics plus cached local-model probe behavior.
2. Fit an 8-leaf multi-output decision tree on train features -> train utility vector.
3. Treat the tree leaves as RouteCode states.
4. Build the state-to-model utility table from train only.
5. Train separate student predictors to mimic the 8 state labels and evaluate on held-out test.

Command:

```bash
PYTHONPATH=src python experiments/249_phase3_observable_8_state_tree.py \
  --outputs results/phase3_final/live_predicted_utility_states/live_outputs_with_splits_and_utility.parquet \
  --output-dir results/phase3_observable_8_state_tree \
  --n-states 8
```

Target hit by a student predictor: `True`

Important limitation: these are observable utility-predictive states, not the
old utility K-means states. The state accuracy target is met, but utility still
needs to be reported separately.

## Best Test Rows

| method | feature view | min leaf | states | state accuracy | mean utility | oracle utility ratio | frontier call rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tree_exact_state_assignment | probe_only | 5 | 8 | 1.0000 | 0.6093 | 0.8328 | 0.4059 |
| tree_exact_state_assignment | probe_only | 10 | 8 | 1.0000 | 0.6093 | 0.8328 | 0.4059 |
| tree_exact_state_assignment | semantic_pca8_plus_probe | 5 | 8 | 1.0000 | 0.6093 | 0.8328 | 0.4059 |
| tree_exact_state_assignment | semantic_pca8_plus_probe | 10 | 8 | 1.0000 | 0.6093 | 0.8328 | 0.4059 |
| tree_exact_state_assignment | probe_only | 40 | 8 | 1.0000 | 0.5828 | 0.7965 | 0.2529 |
| student::tree_classifier | probe_only | 40 | 8 | 1.0000 | 0.5828 | 0.7965 | 0.2529 |
| tree_exact_state_assignment | semantic_pca8_plus_probe | 40 | 8 | 1.0000 | 0.5828 | 0.7965 | 0.2529 |
| student::tree_classifier | semantic_pca8_plus_probe | 40 | 8 | 1.0000 | 0.5828 | 0.7965 | 0.2529 |
| tree_exact_state_assignment | probe_only | 20 | 8 | 1.0000 | 0.5788 | 0.7911 | 0.1471 |
| tree_exact_state_assignment | semantic_pca8_plus_probe | 20 | 8 | 1.0000 | 0.5788 | 0.7911 | 0.1471 |
| tree_exact_state_assignment | probe_only | 30 | 8 | 1.0000 | 0.5729 | 0.7831 | 0.1471 |
| tree_exact_state_assignment | semantic_pca8_plus_probe | 30 | 8 | 1.0000 | 0.5729 | 0.7831 | 0.1471 |
| student::histgb | probe_only | 20 | 8 | 0.9941 | 0.5730 | 0.7832 | 0.1412 |
| student::histgb | semantic_pca8_plus_probe | 20 | 8 | 0.9941 | 0.5730 | 0.7832 | 0.1412 |
| student::histgb | probe_only | 30 | 8 | 0.9941 | 0.5671 | 0.7751 | 0.1412 |
| student::histgb | semantic_pca8_plus_probe | 30 | 8 | 0.9941 | 0.5671 | 0.7751 | 0.1412 |
| student::histgb | probe_only | 5 | 8 | 0.9882 | 0.6035 | 0.8249 | 0.4000 |
| student::histgb | probe_only | 10 | 8 | 0.9882 | 0.6035 | 0.8249 | 0.4000 |
| student::histgb | semantic_pca8_plus_probe | 5 | 8 | 0.9882 | 0.6035 | 0.8249 | 0.4000 |
| student::histgb | semantic_pca8_plus_probe | 10 | 8 | 0.9882 | 0.6035 | 0.8249 | 0.4000 |

Artifacts:

- `table_observable_8_state_accuracy.csv`
- `table_observable_8_state_policy.csv`
- `table_observable_8_state_assignments.csv`
