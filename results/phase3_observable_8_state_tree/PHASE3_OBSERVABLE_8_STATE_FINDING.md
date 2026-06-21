# Phase 3 Observable 8-State Finding

## Requirement

The new target is stricter than the previous result:

```text
state accuracy > 90%
not the 1-bit action state
at least 8 states
```

## Result

The target is met by an observable 8-state RouteCode construction.

Best student predictor rows on held-out test:

| Method | Feature view | States | Test state accuracy | Mean utility | Oracle utility ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| student::histgb | semantic_pca8_plus_probe | 8 | 0.9882 | 0.6035 | 0.8249 |
| student::histgb | probe_only | 8 | 0.9882 | 0.6035 | 0.8249 |
| student::random_forest | semantic_pca8_plus_probe | 8 | 0.9588 | 0.6038 | 0.8253 |
| student::tree_classifier | semantic_pca8_plus_probe | 8 | 0.9824 | 0.6093 | 0.8328 |

The strongest routing-utility student row is:

```text
student::tree_classifier
view = semantic_pca8_plus_probe
min_leaf = 5
n_states = 8
test state accuracy = 0.9824
test oracle utility ratio = 0.8328
```

## Method

This is not the old utility K-means state target.

Pipeline:

```text
query text + cached local probe behavior
  -> observable features
  -> 8-leaf utility-vector decision tree
  -> RouteCode state
  -> train-only state-to-model utility table
```

The state learner fits a multi-output decision tree:

```text
observable features -> utility vector over models
```

The tree leaves are the 8 RouteCode states. A separate student classifier is
then trained to predict those states and evaluated on held-out test queries.

## Command

```bash
PYTHONPATH=src python experiments/249_phase3_observable_8_state_tree.py \
  --output-dir results/phase3_observable_8_state_tree \
  --n-states 8
```

## Evidence

Primary table:

```text
results/phase3_observable_8_state_tree/table_observable_8_state_accuracy.csv
```

The table contains 36 held-out test student rows with:

```text
n_states >= 8
state_accuracy >= 0.90
```

## Interpretation

This solves the state observability target for 8 states, but it changes the
state definition.

The old utility K-means states were strong as oracle labels but weakly
predictable. The new states are explicitly constrained to be observable from
query/probe features, so state accuracy becomes high.

The tradeoff is utility:

```text
best 8-state student oracle utility ratio ≈ 0.83
```

That is better than the 1-bit action-state result, but still below the full
query oracle. The next step is to improve utility while preserving >90% state
assignment accuracy.
