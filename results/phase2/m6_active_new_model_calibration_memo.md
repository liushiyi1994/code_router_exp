# Phase 2 Active New-Model Calibration

Command:

```bash
python experiments/55_active_new_model_calibration.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2 --max-holdout-models 1 --r-values 1,2,4,8
```

This compares route-state calibration strategies under matched new-model evaluation budgets.

Outputs:

- `table_active_new_model_calibration.csv`
- `fig_new_model_calibration_curve.pdf`
- `m6_active_new_model_calibration_memo.md`

Best Rows:

| method | new_model_id | examples_per_label | new_model_evaluations | mean_utility | utility_ci_low | utility_ci_high | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| active_route_state_calibration | Qwen3-8B | 4 | 61 | 0.7397 | 0.7155 | 0.7725 | 0.3158 |
| uniform_route_state_calibration | Qwen3-8B | 4 | 61 | 0.7379 | 0.7121 | 0.7707 | 0.3083 |
| dataset_stratified_calibration | Qwen3-8B | 2 | 31 | 0.7362 | 0.7121 | 0.7708 | 0.3008 |
| random_route_state_calibration | Qwen3-8B | 8 | 121 | 0.7345 | 0.7085 | 0.7716 | 0.2932 |
| embedding_cluster_calibration | Qwen3-8B | 4 | 61 | 0.7310 | 0.7059 | 0.7673 | 0.2782 |
| routecode_no_new_model | Qwen3-8B | 0 | 0 | 0.7190 | 0.6861 | 0.7475 | 0.2256 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 1 | 16 | 0.6069 | 0.5741 | 0.6544 | -0.2632 |
