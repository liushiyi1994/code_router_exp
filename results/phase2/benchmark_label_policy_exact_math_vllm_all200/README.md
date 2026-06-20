# Benchmark-Label Policy Export

## Export

This is an operational benchmark-label route rule, not the core latent-state RouteCode/ProbeRoute++ method.

Policy: `exact_math_qwen_intern`
Mapping: `aime -> Intern-S1-mini`, `math500 -> Qwen3-8B`
Queries: `41`
Mean utility: `0.9268`
Oracle mean utility: `0.9268`
Relative gap to oracle: `0.0000`
Within threshold: `True` at threshold `0.0300`
Regret rows: `0/41`

Outputs:

- `table_policy_summary.csv`
- `table_policy_selections.csv`

Selection table columns include `query_id`, `route_label`, `selected_model`, `selected_utility`, `oracle_model`, `oracle_utility`, and `oracle_regret`.

Route-label distribution:

| route_label | n_queries |
| --- | --- |
| math500 | 27 |
| aime | 14 |
