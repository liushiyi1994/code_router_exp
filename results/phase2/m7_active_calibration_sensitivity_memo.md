# Phase 2 Active Calibration Sensitivity

Command:

```bash
python experiments/62_active_calibration_sensitivity.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2 --max-holdout-models 3 --k-values 8,16,32 --alpha-values 3.0 --r-values 1,4,8 --seeds 0,1
```

This sweeps active new-model calibration state-learning settings under matched evaluation budgets.

Outputs:

- `table_active_calibration_sensitivity.csv`
- `table_active_calibration_sensitivity_summary.csv`
- `table_active_calibration_sensitivity_deltas.csv`
- `m7_active_calibration_sensitivity_memo.md`

Sensitivity Delta Summary:

| sensitivity_name | sensitivity_k | sensitivity_alpha | baseline | paired_rows | active_minus_baseline_mean | active_minus_baseline_std | active_minus_baseline_min | active_minus_baseline_max | positive | negative | tied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k_16_alpha_3p0 | 16 | 3.0000 | uniform_route_state_calibration | 18 | -0.0232 | 0.0506 | -0.1448 | 0.0638 | 5 | 11 | 2 |
| k_32_alpha_3p0 | 32 | 3.0000 | uniform_route_state_calibration | 18 | -0.0128 | 0.0402 | -0.0897 | 0.0517 | 7 | 11 | 0 |
| k_8_alpha_3p0 | 8 | 3.0000 | uniform_route_state_calibration | 18 | -0.0206 | 0.0543 | -0.0897 | 0.1190 | 3 | 14 | 1 |
| k_16_alpha_3p0 | 16 | 3.0000 | random_route_state_calibration | 18 | -0.0098 | 0.0597 | -0.1379 | 0.0966 | 9 | 9 | 0 |
| k_32_alpha_3p0 | 32 | 3.0000 | random_route_state_calibration | 18 | -0.0130 | 0.0352 | -0.1207 | 0.0241 | 7 | 11 | 0 |
| k_8_alpha_3p0 | 8 | 3.0000 | random_route_state_calibration | 18 | -0.0026 | 0.0711 | -0.1224 | 0.1793 | 7 | 11 | 0 |
| k_16_alpha_3p0 | 16 | 3.0000 | dataset_stratified_calibration | 18 | -0.0259 | 0.0441 | -0.1431 | 0.0414 | 6 | 12 | 0 |
| k_32_alpha_3p0 | 32 | 3.0000 | dataset_stratified_calibration | 18 | -0.0313 | 0.0346 | -0.0810 | 0.0190 | 5 | 13 | 0 |
| k_8_alpha_3p0 | 8 | 3.0000 | dataset_stratified_calibration | 18 | -0.0310 | 0.0734 | -0.2069 | 0.0655 | 6 | 10 | 2 |
| k_16_alpha_3p0 | 16 | 3.0000 | embedding_cluster_calibration | 18 | -0.0286 | 0.0373 | -0.1086 | 0.0259 | 4 | 14 | 0 |
| k_32_alpha_3p0 | 32 | 3.0000 | embedding_cluster_calibration | 18 | -0.0299 | 0.0527 | -0.1172 | 0.1069 | 4 | 13 | 1 |
| k_8_alpha_3p0 | 8 | 3.0000 | embedding_cluster_calibration | 18 | -0.0354 | 0.0700 | -0.1879 | 0.0897 | 5 | 11 | 2 |

Best Active Rows:

| sensitivity_name | sensitivity_k | sensitivity_alpha | new_model_id | examples_per_label | replicates | mean_utility_mean | mean_utility_std | new_model_evaluations_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k_8_alpha_3p0 | 8 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 4 | 2 | 0.7629 | 0.0037 | 32.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | Qwen2.5-Coder-7B-Instruct | 4 | 2 | 0.7603 | 0.0024 | 63.0000 |
| k_8_alpha_3p0 | 8 | 3.0000 | Qwen3-8B | 4 | 2 | 0.7534 | 0.0195 | 32.0000 |
| k_8_alpha_3p0 | 8 | 3.0000 | Qwen3-8B | 8 | 2 | 0.7491 | 0.0110 | 64.0000 |
| k_32_alpha_3p0 | 32 | 3.0000 | Qwen3-8B | 8 | 2 | 0.7474 | 0.0061 | 243.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 4 | 2 | 0.7431 | 0.0098 | 64.0000 |
| k_32_alpha_3p0 | 32 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 8 | 2 | 0.7431 | 0.0122 | 241.5000 |
| k_32_alpha_3p0 | 32 | 3.0000 | Qwen2.5-Coder-7B-Instruct | 8 | 2 | 0.7422 | 0.0158 | 235.0000 |
| k_8_alpha_3p0 | 8 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 8 | 2 | 0.7388 | 0.0305 | 64.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | Qwen3-8B | 8 | 2 | 0.7336 | 0.0085 | 125.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | Qwen2.5-Coder-7B-Instruct | 8 | 2 | 0.7310 | 0.0390 | 125.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 8 | 2 | 0.7276 | 0.0536 | 125.0000 |

Interpretation:

Across active-vs-random sensitivity cells, active calibration has mean cell delta `-0.0085` over `3` cells (`0` positive, `3` negative, `0` tied).
