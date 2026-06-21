# Phase 3 Observability Finding

## Question

Can we get RouteCode state prediction above 90% accuracy with query semantics
plus active local probing?

## Short Answer

Yes, but only after changing the target state.

The previous utility-cluster states remain too hard to predict:

- K=2 utility states: best cached active/local-probe result was about 81.8%.
- K=8 utility states: best cached active/local-probe result was about 63.5%.
- Earlier text-only stronger heads also missed the target: K=8 stayed around
  40-44%, and K=16 stayed around 20-35%.

The working state is now a 1-bit, utility-derived action state:

```text
local_enough vs frontier_needed
```

This state is learned from the cost-aware utility matrix. The predictor uses
only query semantics and cached local-model behavior.

## Best Working Row

Command:

```bash
PYTHONPATH=src python experiments/248_phase3_probe_observable_action_state.py \
  --output-dir results/phase3_probe_observable_action_state
```

Best held-out test result:

| Method | State accuracy | Balanced accuracy | Frontier precision | Frontier recall | Mean utility | Oracle utility ratio | Probe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_probe::semantic_pca8_plus_probe::histgb | 0.9118 | 0.7117 | 0.6923 | 0.4500 | 0.5037 | 0.6885 | 1.0000 |

This is the first saved result above 90% state accuracy with a deployable
feature set.

## Important Limitation

This does not solve full RouteCode routing yet.

The true 1-bit action-state oracle only reaches 0.7650 of query-oracle utility,
because the two states are coarse. The high-accuracy predictor reaches 0.6885
of query-oracle utility. So we solved observability for a coarse state, not
high-utility fine routing.

## What This Means

More text-only data may improve K=8/K=16 state prediction somewhat, but the
current evidence says it is unlikely to jump from 30-60% to 90% without changing
the state definition or adding stronger probes.

The next method should be hierarchical:

```text
query semantics + local probe
  -> high-confidence 1-bit action state
  -> optional substate only when needed
  -> state-to-model table
```

This keeps the first state observable while preserving a path back toward higher
utility with finer substates.
