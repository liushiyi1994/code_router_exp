# Two-Claim Live Completion Audit

This audit records the current live-backed status for the two Phase 3 claims:

1. learned/predicted states are better calibration strata than simple labels or text clusters;
2. state-based active calibration is useful for new-model onboarding.

## Claim 1: Calibration Strata

Status: `supported_on_live_broad100_stage0`

Evidence:

- Artifact: `results/phase3_final/live_predicted_utility_states/table_live_predicted_state_claims.csv`
- Claim row: `live_predicted_states_as_calibration_strata`
- Selected deployable state method: `predicted_utility_state_rf_probe_plus_benchmark_k16`
- Held-out test utility variance: `0.1366`
- Best label/text baseline variance: `0.1666`
- Outcome matrix: live/cached Broad100 Stage0 with GPT, Gemini, and local vLLM rows.

Interpretation:

The predicted utility states form tighter live calibration strata than benchmark labels or text clusters on held-out test queries.

## Claim 2: Active New-Model Onboarding

Status: `supported_on_live_broad100_stage0` for the GPT/Gemini frontier onboarding slice.

Evidence:

- Artifact: `results/phase3_final/live_predicted_utility_states/table_live_frontier_onboarding_test.csv`
- Claim row: `live_frontier_active_onboarding_low_budget`
- Held-out new models: `gpt-5.5`, `gemini-3.5-flash`
- Validation-selected budget: `40`
- Test active utility: `0.5627`
- Best competitor utility at the same selected budget: `0.5510`
- Active margin: `+0.0117`

Budget-to-match evidence:

- Artifact: `results/phase3_final/live_predicted_utility_states/table_live_frontier_budget_efficiency.csv`
- Active reaches target utility with `40` evaluations.
- Random calibration matches only at `320` evaluations: `8.0x` reduction.
- Direct retraining proxy does not match by `320` evaluations: at least `8.0x` reduction lower bound.
- Uniform state calibration matches at `80` evaluations: `2.0x` reduction.

Interpretation:

For the practical closed-source frontier onboarding case, active state calibration meets the Phase 3 sample-efficiency gate versus random calibration and the direct retraining proxy.

## Residual Caveat

The all-model average active acquisition row remains weak:

- `live_active_acquisition_advantage`: active `0.5656`, random `0.5640`, uniform `0.5650`, margin `+0.0006`.

This means the supported active-onboarding claim should be scoped to frontier/new-provider onboarding, not stated as a universal all-model average result.

## Verification

Latest verification command:

```bash
pytest -q
```

Result:

```text
325 passed, 24 warnings
```
