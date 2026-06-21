# Model-Holdout Repair State Clustering Memo

## What Changed

Implemented `model_holdout_repaired` in `src/routecode/states/utility_states_v2.py`.

The repaired state learner now uses:

- relative routing features: centered utility, regret, rank, soft preference, margins;
- calibration features: raw utility columns and centered utility columns;
- split repair for states with high model-holdout variance/error;
- optional budget behavior:
  - default: merge repaired states back to requested `K`;
  - `--allow-extra-repair-states`: keep repair splits for lower calibration variance.

## Commands

Equal-budget repair:

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py \
  --output-dir results/phase3_routecode_v2_model_holdout_repair \
  --state-methods raw_kmeans relative_kmeans calibration_refined model_holdout_repaired \
  --k-values 16 24 \
  --predictors knn \
  --active-label-budgets 64 \
  --model-holdout-variance-threshold 0.025 \
  --model-holdout-error-threshold 0.10 \
  --model-holdout-min-state-size 8 \
  --model-holdout-max-split-fraction 1.0
```

Extra-state calibration repair:

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py \
  --output-dir results/phase3_routecode_v2_model_holdout_repair_extra_states \
  --state-methods raw_kmeans relative_kmeans calibration_refined model_holdout_repaired \
  --k-values 16 24 \
  --predictors knn \
  --active-label-budgets 64 \
  --model-holdout-variance-threshold 0.025 \
  --model-holdout-error-threshold 0.10 \
  --model-holdout-min-state-size 8 \
  --model-holdout-max-split-fraction 1.0 \
  --allow-extra-repair-states
```

## Main Result

On held-out Broad100 test queries, the extra-state repair gives the clearest calibration-strata improvement:

| method | requested K | states used on test | mean model utility variance | selected-model variance | mean abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| relative_kmeans | 24 | 22 | 0.020630 | 0.000761 | 0.043210 |
| calibration_refined | 24 | 22 | 0.020630 | 0.000761 | 0.043210 |
| raw_kmeans | 24 | 23 | 0.022062 | 0.017582 | 0.045334 |
| model_holdout_repaired extra-states | 24 | 26 | 0.015143 | 0.000737 | 0.032937 |

This is a 26.6% reduction in traffic-weighted mean model utility variance versus relative K=24.

Equal-budget repair has a different tradeoff:

- `model_holdout_repaired` K=24 reaches diagnostic true-state test utility `0.731635`, equal to the query oracle in this cached matrix.
- But it does not lower all-model calibration variance versus relative K=24: `0.023139` vs `0.020630`.
- It does slightly lower selected-model variance: `0.000737` vs `0.000761`.

## Interpretation

The repaired states can improve calibration strata when allowed to keep split states. For a strict fixed-K codebook, the repair is better as an action-table repair than as an all-model calibration-variance repair.

This means the next calibration experiment should report two regimes:

- fixed-rate RouteCode: preserve K and evaluate utility/rate-distortion;
- calibration RouteCode: allow a small number of repair splits and evaluate new-model onboarding sample efficiency.

## Artifacts

- `table_v2_state_policy.csv`
- `table_v2_query_state_predictor_diagnostics.csv`
- `table_v2_state_assignments.csv`
- `table_v2_state_cards.csv`
- `table_model_holdout_repair_variance_comparison.csv`
- `README.md`
