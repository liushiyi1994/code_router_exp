# RouteCode Phase 3 Final Results

This folder is the dedicated final-results package for the current RouteCode /
ProbeCode Phase 3 evaluation.

The current method should be described as:

```text
query + cheap local/probe behavior -> predicted utility state -> cost-aware action
```

The strongest result is not a clean query-only router. The supported result is a
ProbeCode-StateCal system that uses observable local/probe behavior, compact
states, and a cost-aware action table over local, verifiable, GPT, and Gemini
actions.

## Bottom Line

Main Broad100 held-out test result:

| Method | Quality | Utility | Oracle Utility Ratio | Frontier Rate |
| --- | ---: | ---: | ---: | ---: |
| Cost-aware oracle | 0.8721 | 0.8463 | 1.0000 | 0.1395 |
| ProbeCode-StateCal | 0.8547 | 0.8238 | 0.9735 | 0.1919 |

This passes the configured Phase 3 gate:

- quality gap to oracle <= 0.03;
- oracle utility ratio >= 0.97;
- frontier-call rate <= 0.40.

## Benchmark Scope

The Broad100 final package uses 9 benchmark families:

```text
aime
bbh
gpqa
gsm8k
humaneval
livemathbench
math500
mbpp
mmlupro
```

The current state model is learned and selected using train/validation splits
drawn from these benchmark families, and final routing results are reported on
held-out test queries from the same families.

That means the current final eval supports within-family held-out routing and
calibration. It does not yet prove that the same states generalize to entirely
new benchmark families. A new-benchmark or leave-one-benchmark-family-out
evaluation is the next required generalization test.

## What The State Comes From

There are two related state objects in the current package.

1. `gb_depth2_thr0.9844_state_k8`: the compact learned RouteCode/probe state
   used by the main Broad100 method card. It is fit from Broad100 train-split
   cached outcomes and observable local/probe behavior, then selected using
   validation.

2. `predicted_utility_state_rf_probe_plus_benchmark_k16` and
   `predicted_utility_state_rf_probe_plus_benchmark_k6`: the predicted utility
   states used for the strongest calibration-strata and new-model-onboarding
   claims. Utility states are learned from train-split utility patterns, then a
   predictor maps observable probe features to state ids. K/state variants are
   chosen on validation and reported on held-out test.

In both cases, state learning uses Broad100 data from the 9 families above. The
test rows are query-heldout, not benchmark-family-heldout.

## Main Files

Start here:

- `FINAL_EVALUATION_REPORT.md`: full current result summary.
- `TWO_CLAIM_LIVE_COMPLETION_AUDIT.md`: status for the two calibration claims.
- `EXPERIMENT_PROTOCOL.md`: commands, data flow, and benchmark-scope caveats.
- `final_method/METHOD_CARD.md`: compact method card and action mix.

Main tables:

- `table_final_main_eval.csv`: accuracy, utility, cost, latency, and baselines.
- `table_final_baselines.csv`: final comparison rows.
- `table_final_claims.csv`: claim ledger and caveats.
- `table_final_ablation.csv`: ablations.
- `table_final_sensitivity.csv`: lambda/price sensitivity.
- `table_final_live_predicted_utility_state_claims.csv`: live Stage0 state claims.
- `table_final_live_frontier_onboarding_test.csv`: GPT/Gemini onboarding test slice.
- `table_final_live_frontier_budget_efficiency.csv`: budget-to-match evidence.

Main figures:

- `fig_final_quality_cost_frontier.pdf`
- `fig_final_oracle_gap.pdf`
- `fig_final_calibration_efficiency.pdf`
- `live_predicted_utility_states/fig_predicted_state_variance.pdf`
- `live_predicted_utility_states/fig_predicted_state_onboarding.pdf`
- `ablation/fig_ablation_utility.pdf`
- `sensitivity/fig_price_sensitivity.pdf`

## Current Claim Status

Supported:

- ProbeCode-StateCal reaches the configured Broad100 oracle-level gate.
- Predicted utility states are better live calibration strata than benchmark
  labels or text clusters.
- Active predicted-state calibration helps the GPT/Gemini frontier onboarding
  slice at low budget.
- Frozen states can adapt the state-to-action table when costs change.

Weak or scoped:

- All-model average active acquisition is only weakly supported.
- The current result is not a clean no-tool routing result.
- The current final test is not a new-benchmark-family test.
- `results/phase3_new_benchmark_live/` now contains a first 15-query
  new-benchmark live smoke on SimpleQA Verified and LiveBench. It shows a
  local-vs-GPT oracle opportunity, but it is not enough to claim state
  generalization.

Not supported:

- The older claim that small inferred query-only labels recover most oracle
  routing performance.
- Adaptive refinement from the current residual-risk signal.
