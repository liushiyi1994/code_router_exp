# Phase D5 Adaptive-Refinement Gate Memo

Command: `python experiments/03_residual_concentration.py --config configs/llmrouterbench_broad20.yaml`

This memo checks whether residual failures are concentrated and predictable enough to justify implementing adaptive refinement. It is a gate, not an adaptive-refinement result.

## Residual Concentration

| top_fraction | n_queries | top_regret | total_regret | regret_mass_fraction |
| --- | --- | --- | --- | --- |
| 0.0500 | 141.0000 | 141.0000 | 881.0000 | 0.1600 |
| 0.1000 | 281.0000 | 281.0000 | 881.0000 | 0.3190 |
| 0.2000 | 562.0000 | 562.0000 | 881.0000 | 0.6379 |

## Best Deployable Risk Signals

| score | top_fraction | n_flagged | regret_mass_fraction | positive_regret_recall | auc_regret_positive |
| --- | --- | --- | --- | --- | --- |
| low_route_label_confidence | 0.0500 | 141 | 0.0556 | 0.0556 | 0.5370 |
| low_route_label_confidence | 0.1000 | 281 | 0.1056 | 0.1056 | 0.5370 |
| low_route_label_confidence | 0.2000 | 562 | 0.2191 | 0.2191 | 0.5370 |

## Current Decision

- Best deployable signals capture `0.0556` of regret in the top 5% and `0.1056` in the top 10% of flagged queries.
- Adaptive refinement should remain deferred; the current gate is not strong enough for a core claim.
- The oracle-margin diagnostic is included to show an upper-bound-style non-deployable signal; it should not be treated as a deployable trigger.
