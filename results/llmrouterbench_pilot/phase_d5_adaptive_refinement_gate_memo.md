# Phase D5 Adaptive-Refinement Gate Memo

Command: `python experiments/03_residual_concentration.py --config configs/llmrouterbench_pilot.yaml`

This memo checks whether residual failures are concentrated and predictable enough to justify implementing adaptive refinement. It is a gate, not an adaptive-refinement result.

## Residual Concentration

| top_fraction | n_queries | top_regret | total_regret | regret_mass_fraction |
| --- | --- | --- | --- | --- |
| 0.0500 | 29.0000 | 29.0000 | 164.0000 | 0.1768 |
| 0.1000 | 58.0000 | 58.0000 | 164.0000 | 0.3537 |
| 0.2000 | 116.0000 | 116.0000 | 164.0000 | 0.7073 |

## Best Deployable Risk Signals

| score | top_fraction | n_flagged | regret_mass_fraction | positive_regret_recall | auc_regret_positive |
| --- | --- | --- | --- | --- | --- |
| low_route_label_confidence | 0.0500 | 29 | 0.0549 | 0.0549 | 0.5547 |
| low_route_label_confidence | 0.1000 | 58 | 0.1220 | 0.1220 | 0.5547 |
| low_route_label_confidence | 0.2000 | 116 | 0.2439 | 0.2439 | 0.5547 |

## Current Decision

- Best deployable signals capture `0.0549` of regret in the top 5% and `0.1220` in the top 10% of flagged queries.
- Adaptive refinement should remain deferred; the current gate is not strong enough for a core claim.
- The oracle-margin diagnostic is included to show an upper-bound-style non-deployable signal; it should not be treated as a deployable trigger.
