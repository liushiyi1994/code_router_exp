# ProbeCode-StateCal Method Card

## Method

- Name: `ProbeCode-StateCal`
- Current best policy: `et_flip_leaf4_thr0.8502_capNone`
- Compact state policy: `gb_depth2_thr0.9844_state_k8`
- Base policy: `extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75`
- Lambda cost: `0.35`
- Verifiable actions allowed: `True`
- Caveat: Clean no-tool routing is not claimed as solved.

## Data Flow

```text
query -> cheap local/verifiable behavior -> probe/route state -> state-to-action utility table -> action
```

## Current Broad100 Test Evidence

- Mean quality: `0.854651`
- Mean utility: `0.823815`
- Quality gap to oracle: `0.017442`
- Oracle utility ratio: `0.973483`
- Frontier-call rate: `0.191860`
- N test queries: `172`

## Action Mix

- `qwen3-32b-awq-local`: `73` queries
- `qwen3-4b-local`: `29` queries
- `deterministic_math_tool`: `23` queries
- `gemini-3.5-flash`: `15` queries
- `gemini-3.5-flash-strong-solve`: `15` queries
- `qwen3-32b-awq-selfconsistency-n3-local`: `6` queries
- `qwen3-14b-awq-local`: `5` queries
- `gpt-5.5`: `3` queries
- `qwen3-8b-local`: `3` queries

## State Cards

- State-card rows: `8`

## Current Claim Ledger

- `phase3_broad100_current_best_oracle_level_target`: `supported`; quality_gap=0.0174;utility_ratio=0.9735;frontier_rate=0.1919
- `phase3_broad100_routecode_state_policy_target`: `supported_with_lower_utility`; quality_gap=0.0233;utility_ratio=0.9614;frontier_rate=0.2384
- `phase3_no_tool_full_oracle_target`: `not_supported_feasibility_bound`; no_tool_oracle_quality_gap_to_full=0.0465;no_tool_oracle_utility_ratio_to_full=0.9338
- `phase3_exact_math_controlled_targets`: `supported`; quality_gap=0.0152;utility_ratio=0.9739;frontier_rate=0.1061;normalized_remote_cost=0.0463;p95_latency_ratio_vs_all_gpt=0.4799
- `phase3_state_level_new_model_calibration`: `supported_on_cached_exact_math`; active_evals=4;active_quality=0.8485;direct_best_quality=0.7273
- `phase3_budget_and_model_constraints`: `supported`; top_level_max_model_cost=4.9765;broad_stage0_max_model_cost=0.2512
- `phase3_controlled_verifiability_action_pool_scope`: `supported`; broad100_quality_gap=0.0174;broad100_utility_ratio=0.9735;exact_quality_gap=0.0152;active_evals=4;top_level_max_model_cost=4.9765

## Source Artifacts

- Current best eval: `results/controlled/broad100_current_best_method_package/table_broad100_current_best_main_eval.csv`
- Current best choices: `results/controlled/broad100_current_best_method_package/table_broad100_current_best_query_choices.csv`
- State cards source: `results/controlled/broad100_learned_verifiability_probe_state/table_learned_verifiability_code_cards.csv`
