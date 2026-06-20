# Phase 2 Active Calibration Replicates

Command:

```bash
python experiments/61_active_calibration_replicates.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2 --max-holdout-models 6 --seeds 0,1,2 --r-values 1,2,4,8
```

This repeats the active new-model calibration comparison across seeds and held-out models.

Outputs:

- `table_active_calibration_replicates.csv`
- `table_active_calibration_replicate_summary.csv`
- `table_active_calibration_active_vs_uniform_deltas.csv`
- `table_active_calibration_active_vs_random_deltas.csv`
- `table_active_calibration_active_vs_dataset_deltas.csv`
- `table_active_calibration_active_vs_embedding_deltas.csv`
- `m6_active_calibration_replicates_memo.md`

Replicate Summary:

| method | new_model_id | examples_per_label | replicates | mean_utility_mean | mean_utility_std | mean_utility_min | mean_utility_max | recovered_gap_vs_oracle_mean | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| active_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.6839 | 0.0342 | 0.6466 | 0.7138 | 0.0727 | 16.0000 | 16 | 16 |
| active_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.6793 | 0.0194 | 0.6672 | 0.7017 | 0.0526 | 32.0000 | 32 | 32 |
| active_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.7270 | 0.0155 | 0.7172 | 0.7448 | 0.2607 | 63.3333 | 62 | 64 |
| active_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.6730 | 0.0697 | 0.5983 | 0.7362 | 0.0251 | 126.0000 | 122 | 128 |
| active_route_state_calibration | Intern-S1-mini | 1 | 3 | 0.6575 | 0.0512 | 0.6034 | 0.7052 | -0.0426 | 16.0000 | 16 | 16 |
| active_route_state_calibration | Intern-S1-mini | 2 | 3 | 0.7195 | 0.0070 | 0.7155 | 0.7276 | 0.2281 | 32.0000 | 32 | 32 |
| active_route_state_calibration | Intern-S1-mini | 4 | 3 | 0.7098 | 0.0318 | 0.6828 | 0.7448 | 0.1855 | 64.0000 | 64 | 64 |
| active_route_state_calibration | Intern-S1-mini | 8 | 3 | 0.7333 | 0.0078 | 0.7259 | 0.7414 | 0.2882 | 127.0000 | 126 | 128 |
| active_route_state_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.6161 | 0.0202 | 0.5931 | 0.6310 | -0.2231 | 16.0000 | 16 | 16 |
| active_route_state_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.7011 | 0.0297 | 0.6672 | 0.7224 | 0.1479 | 31.3333 | 30 | 32 |
| active_route_state_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.6874 | 0.0552 | 0.6241 | 0.7259 | 0.0877 | 61.0000 | 58 | 64 |
| active_route_state_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7109 | 0.0101 | 0.7034 | 0.7224 | 0.1905 | 118.3333 | 113 | 125 |
| active_route_state_calibration | MiniCPM4.1-8B | 1 | 3 | 0.7305 | 0.0088 | 0.7207 | 0.7379 | 0.2757 | 16.0000 | 16 | 16 |
| active_route_state_calibration | MiniCPM4.1-8B | 2 | 3 | 0.7305 | 0.0115 | 0.7172 | 0.7379 | 0.2757 | 31.3333 | 31 | 32 |
| active_route_state_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7253 | 0.0095 | 0.7155 | 0.7345 | 0.2531 | 61.3333 | 59 | 64 |
| active_route_state_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7379 | 0.0096 | 0.7293 | 0.7483 | 0.3083 | 119.3333 | 115 | 125 |
| active_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.6420 | 0.0248 | 0.6138 | 0.6603 | -0.1103 | 16.0000 | 16 | 16 |
| active_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.7057 | 0.0214 | 0.6810 | 0.7190 | 0.1679 | 31.6667 | 31 | 32 |
| active_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.7282 | 0.0173 | 0.7086 | 0.7414 | 0.2657 | 62.6667 | 61 | 64 |
| active_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7190 | 0.0135 | 0.7103 | 0.7345 | 0.2256 | 123.3333 | 121 | 127 |
| active_route_state_calibration | Qwen3-8B | 1 | 3 | 0.7046 | 0.0174 | 0.6845 | 0.7155 | 0.1629 | 16.0000 | 16 | 16 |
| active_route_state_calibration | Qwen3-8B | 2 | 3 | 0.7144 | 0.0263 | 0.6914 | 0.7431 | 0.2055 | 32.0000 | 32 | 32 |
| active_route_state_calibration | Qwen3-8B | 4 | 3 | 0.7282 | 0.0202 | 0.7052 | 0.7431 | 0.2657 | 63.3333 | 62 | 64 |
| active_route_state_calibration | Qwen3-8B | 8 | 3 | 0.7149 | 0.0275 | 0.6845 | 0.7379 | 0.2080 | 126.0000 | 122 | 128 |
| dataset_stratified_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.7259 | 0.0194 | 0.7138 | 0.7483 | 0.2556 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.7253 | 0.0127 | 0.7155 | 0.7397 | 0.2531 | 32.0000 | 32 | 32 |
| dataset_stratified_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.7132 | 0.0268 | 0.6914 | 0.7431 | 0.2005 | 63.3333 | 62 | 64 |
| dataset_stratified_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.7264 | 0.0127 | 0.7121 | 0.7362 | 0.2581 | 126.0000 | 122 | 128 |
| dataset_stratified_calibration | Intern-S1-mini | 1 | 3 | 0.6960 | 0.0354 | 0.6569 | 0.7259 | 0.1253 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | Intern-S1-mini | 2 | 3 | 0.7195 | 0.0286 | 0.6897 | 0.7466 | 0.2281 | 32.0000 | 32 | 32 |
| dataset_stratified_calibration | Intern-S1-mini | 4 | 3 | 0.7374 | 0.0088 | 0.7276 | 0.7448 | 0.3058 | 64.0000 | 64 | 64 |
| dataset_stratified_calibration | Intern-S1-mini | 8 | 3 | 0.7339 | 0.0078 | 0.7259 | 0.7414 | 0.2907 | 127.0000 | 126 | 128 |
| dataset_stratified_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.7057 | 0.0177 | 0.6862 | 0.7207 | 0.1679 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.7086 | 0.0086 | 0.7000 | 0.7172 | 0.1805 | 31.3333 | 30 | 32 |
| dataset_stratified_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.7138 | 0.0113 | 0.7034 | 0.7259 | 0.2030 | 61.0000 | 58 | 64 |
| dataset_stratified_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7144 | 0.0277 | 0.6828 | 0.7345 | 0.2055 | 118.3333 | 113 | 125 |
| dataset_stratified_calibration | MiniCPM4.1-8B | 1 | 3 | 0.6983 | 0.0702 | 0.6172 | 0.7397 | 0.1353 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | MiniCPM4.1-8B | 2 | 3 | 0.7052 | 0.0352 | 0.6655 | 0.7328 | 0.1654 | 31.3333 | 31 | 32 |
| dataset_stratified_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7385 | 0.0055 | 0.7345 | 0.7448 | 0.3108 | 61.3333 | 59 | 64 |
| dataset_stratified_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7420 | 0.0081 | 0.7328 | 0.7483 | 0.3258 | 119.3333 | 115 | 125 |
| dataset_stratified_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.6908 | 0.0145 | 0.6741 | 0.7000 | 0.1028 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.7109 | 0.0115 | 0.6983 | 0.7207 | 0.1905 | 31.6667 | 31 | 32 |
| dataset_stratified_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.7190 | 0.0086 | 0.7103 | 0.7276 | 0.2256 | 62.6667 | 61 | 64 |
| dataset_stratified_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7316 | 0.0115 | 0.7241 | 0.7448 | 0.2807 | 123.3333 | 121 | 127 |
| dataset_stratified_calibration | Qwen3-8B | 1 | 3 | 0.6960 | 0.0132 | 0.6845 | 0.7103 | 0.1253 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | Qwen3-8B | 2 | 3 | 0.7282 | 0.0261 | 0.6983 | 0.7466 | 0.2657 | 32.0000 | 32 | 32 |
| dataset_stratified_calibration | Qwen3-8B | 4 | 3 | 0.7034 | 0.0170 | 0.6897 | 0.7224 | 0.1579 | 63.3333 | 62 | 64 |
| dataset_stratified_calibration | Qwen3-8B | 8 | 3 | 0.7282 | 0.0125 | 0.7138 | 0.7362 | 0.2657 | 126.0000 | 122 | 128 |
| direct_retraining_budgeted_logistic_active_budget | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 32.0000 | 32 | 32 |
| direct_retraining_budgeted_logistic_active_budget | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 63.3333 | 62 | 64 |
| direct_retraining_budgeted_logistic_active_budget | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 126.0000 | 122 | 128 |
| direct_retraining_budgeted_logistic_active_budget | Intern-S1-mini | 1 | 3 | 0.6810 | 0.0000 | 0.6810 | 0.6810 | 0.0602 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | Intern-S1-mini | 2 | 3 | 0.6810 | 0.0000 | 0.6810 | 0.6810 | 0.0602 | 32.0000 | 32 | 32 |
| direct_retraining_budgeted_logistic_active_budget | Intern-S1-mini | 4 | 3 | 0.6810 | 0.0000 | 0.6810 | 0.6810 | 0.0602 | 64.0000 | 64 | 64 |
| direct_retraining_budgeted_logistic_active_budget | Intern-S1-mini | 8 | 3 | 0.6805 | 0.0010 | 0.6793 | 0.6810 | 0.0576 | 127.0000 | 126 | 128 |
| direct_retraining_budgeted_logistic_active_budget | Llama-3.1-8B-Instruct | 1 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | Llama-3.1-8B-Instruct | 2 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 31.3333 | 30 | 32 |
| direct_retraining_budgeted_logistic_active_budget | Llama-3.1-8B-Instruct | 4 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 61.0000 | 58 | 64 |
| direct_retraining_budgeted_logistic_active_budget | Llama-3.1-8B-Instruct | 8 | 3 | 0.6770 | 0.0010 | 0.6759 | 0.6776 | 0.0426 | 118.3333 | 113 | 125 |
| direct_retraining_budgeted_logistic_active_budget | MiniCPM4.1-8B | 1 | 3 | 0.6690 | 0.0000 | 0.6690 | 0.6690 | 0.0075 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | MiniCPM4.1-8B | 2 | 3 | 0.6678 | 0.0020 | 0.6655 | 0.6690 | 0.0025 | 31.3333 | 31 | 32 |
| direct_retraining_budgeted_logistic_active_budget | MiniCPM4.1-8B | 4 | 3 | 0.6678 | 0.0020 | 0.6655 | 0.6690 | 0.0025 | 61.3333 | 59 | 64 |
| direct_retraining_budgeted_logistic_active_budget | MiniCPM4.1-8B | 8 | 3 | 0.6667 | 0.0020 | 0.6655 | 0.6690 | -0.0025 | 119.3333 | 115 | 125 |
| direct_retraining_budgeted_logistic_active_budget | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.6667 | 0.0010 | 0.6655 | 0.6672 | -0.0025 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.6667 | 0.0010 | 0.6655 | 0.6672 | -0.0025 | 31.6667 | 31 | 32 |
| direct_retraining_budgeted_logistic_active_budget | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.6661 | 0.0010 | 0.6655 | 0.6672 | -0.0050 | 62.6667 | 61 | 64 |
| direct_retraining_budgeted_logistic_active_budget | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.6655 | 0.0034 | 0.6621 | 0.6690 | -0.0075 | 123.3333 | 121 | 127 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 1 | 3 | 0.6069 | 0.0000 | 0.6069 | 0.6069 | -0.2632 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 2 | 3 | 0.6069 | 0.0000 | 0.6069 | 0.6069 | -0.2632 | 32.0000 | 32 | 32 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 4 | 3 | 0.6052 | 0.0030 | 0.6017 | 0.6069 | -0.2707 | 63.3333 | 62 | 64 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 8 | 3 | 0.6052 | 0.0030 | 0.6017 | 0.6069 | -0.2707 | 126.0000 | 122 | 128 |
| embedding_cluster_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.6856 | 0.0407 | 0.6431 | 0.7241 | 0.0802 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.6816 | 0.0259 | 0.6569 | 0.7086 | 0.0627 | 32.0000 | 32 | 32 |
| embedding_cluster_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.6920 | 0.0313 | 0.6586 | 0.7207 | 0.1078 | 63.3333 | 62 | 64 |
| embedding_cluster_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.6730 | 0.0369 | 0.6500 | 0.7155 | 0.0251 | 126.0000 | 122 | 128 |
| embedding_cluster_calibration | Intern-S1-mini | 1 | 3 | 0.6971 | 0.0339 | 0.6759 | 0.7362 | 0.1303 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | Intern-S1-mini | 2 | 3 | 0.7155 | 0.0299 | 0.6828 | 0.7414 | 0.2105 | 32.0000 | 32 | 32 |
| embedding_cluster_calibration | Intern-S1-mini | 4 | 3 | 0.7368 | 0.0098 | 0.7259 | 0.7448 | 0.3033 | 64.0000 | 64 | 64 |
| embedding_cluster_calibration | Intern-S1-mini | 8 | 3 | 0.7224 | 0.0149 | 0.7052 | 0.7310 | 0.2406 | 127.0000 | 126 | 128 |
| embedding_cluster_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.6954 | 0.0145 | 0.6862 | 0.7121 | 0.1228 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.7201 | 0.0140 | 0.7103 | 0.7362 | 0.2306 | 31.3333 | 30 | 32 |
| embedding_cluster_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.7011 | 0.0252 | 0.6741 | 0.7241 | 0.1479 | 61.0000 | 58 | 64 |
| embedding_cluster_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7144 | 0.0078 | 0.7069 | 0.7224 | 0.2055 | 118.3333 | 113 | 125 |
| embedding_cluster_calibration | MiniCPM4.1-8B | 1 | 3 | 0.7230 | 0.0183 | 0.7034 | 0.7397 | 0.2431 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | MiniCPM4.1-8B | 2 | 3 | 0.7213 | 0.0207 | 0.7000 | 0.7414 | 0.2356 | 31.3333 | 31 | 32 |
| embedding_cluster_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7402 | 0.0112 | 0.7293 | 0.7517 | 0.3183 | 61.3333 | 59 | 64 |
| embedding_cluster_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7379 | 0.0034 | 0.7345 | 0.7414 | 0.3083 | 119.3333 | 115 | 125 |
| embedding_cluster_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.7086 | 0.0130 | 0.6966 | 0.7224 | 0.1805 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.7069 | 0.0137 | 0.6966 | 0.7224 | 0.1729 | 31.6667 | 31 | 32 |
| embedding_cluster_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.7172 | 0.0124 | 0.7034 | 0.7276 | 0.2180 | 62.6667 | 61 | 64 |
| embedding_cluster_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7213 | 0.0277 | 0.6897 | 0.7414 | 0.2356 | 123.3333 | 121 | 127 |
| embedding_cluster_calibration | Qwen3-8B | 1 | 3 | 0.6822 | 0.0181 | 0.6638 | 0.7000 | 0.0652 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | Qwen3-8B | 2 | 3 | 0.7339 | 0.0105 | 0.7224 | 0.7431 | 0.2907 | 32.0000 | 32 | 32 |
| embedding_cluster_calibration | Qwen3-8B | 4 | 3 | 0.7230 | 0.0177 | 0.7034 | 0.7379 | 0.2431 | 63.3333 | 62 | 64 |
| embedding_cluster_calibration | Qwen3-8B | 8 | 3 | 0.7218 | 0.0131 | 0.7069 | 0.7310 | 0.2381 | 126.0000 | 122 | 128 |
| random_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.7155 | 0.0290 | 0.6931 | 0.7483 | 0.2105 | 16.0000 | 16 | 16 |
| random_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.7029 | 0.0413 | 0.6552 | 0.7276 | 0.1554 | 32.0000 | 32 | 32 |
| random_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.7023 | 0.0160 | 0.6845 | 0.7155 | 0.1529 | 63.3333 | 62 | 64 |
| random_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.7276 | 0.0199 | 0.7052 | 0.7431 | 0.2632 | 126.0000 | 122 | 128 |
| random_route_state_calibration | Intern-S1-mini | 1 | 3 | 0.7345 | 0.0119 | 0.7276 | 0.7483 | 0.2932 | 16.0000 | 16 | 16 |
| random_route_state_calibration | Intern-S1-mini | 2 | 3 | 0.7368 | 0.0156 | 0.7190 | 0.7483 | 0.3033 | 32.0000 | 32 | 32 |
| random_route_state_calibration | Intern-S1-mini | 4 | 3 | 0.7362 | 0.0062 | 0.7293 | 0.7414 | 0.3008 | 64.0000 | 64 | 64 |
| random_route_state_calibration | Intern-S1-mini | 8 | 3 | 0.7414 | 0.0034 | 0.7379 | 0.7448 | 0.3233 | 127.0000 | 126 | 128 |
| random_route_state_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.6822 | 0.0101 | 0.6707 | 0.6897 | 0.0652 | 16.0000 | 16 | 16 |
| random_route_state_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.7132 | 0.0265 | 0.6828 | 0.7310 | 0.2005 | 31.3333 | 30 | 32 |
| random_route_state_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.7069 | 0.0062 | 0.7017 | 0.7138 | 0.1729 | 61.0000 | 58 | 64 |
| random_route_state_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7069 | 0.0141 | 0.6948 | 0.7224 | 0.1729 | 118.3333 | 113 | 125 |
| random_route_state_calibration | MiniCPM4.1-8B | 1 | 3 | 0.7293 | 0.0105 | 0.7172 | 0.7362 | 0.2707 | 16.0000 | 16 | 16 |
| random_route_state_calibration | MiniCPM4.1-8B | 2 | 3 | 0.7305 | 0.0174 | 0.7103 | 0.7414 | 0.2757 | 31.3333 | 31 | 32 |
| random_route_state_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7287 | 0.0199 | 0.7172 | 0.7517 | 0.2682 | 61.3333 | 59 | 64 |
| random_route_state_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7420 | 0.0098 | 0.7310 | 0.7500 | 0.3258 | 119.3333 | 115 | 125 |
| random_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.7000 | 0.0069 | 0.6931 | 0.7069 | 0.1429 | 16.0000 | 16 | 16 |
| random_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.6994 | 0.0263 | 0.6707 | 0.7224 | 0.1404 | 31.6667 | 31 | 32 |
| random_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.7213 | 0.0070 | 0.7138 | 0.7276 | 0.2356 | 62.6667 | 61 | 64 |
| random_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7299 | 0.0095 | 0.7207 | 0.7397 | 0.2732 | 123.3333 | 121 | 127 |
| random_route_state_calibration | Qwen3-8B | 1 | 3 | 0.7259 | 0.0225 | 0.7000 | 0.7414 | 0.2556 | 16.0000 | 16 | 16 |
| random_route_state_calibration | Qwen3-8B | 2 | 3 | 0.7195 | 0.0144 | 0.7034 | 0.7310 | 0.2281 | 32.0000 | 32 | 32 |
| random_route_state_calibration | Qwen3-8B | 4 | 3 | 0.7310 | 0.0179 | 0.7103 | 0.7414 | 0.2782 | 63.3333 | 62 | 64 |
| random_route_state_calibration | Qwen3-8B | 8 | 3 | 0.7287 | 0.0156 | 0.7121 | 0.7431 | 0.2682 | 126.0000 | 122 | 128 |
| routecode_no_new_model | DeepSeek-R1-Distill-Qwen-7B | 0 | 3 | 0.7374 | 0.0122 | 0.7241 | 0.7483 | 0.3058 | 0.0000 | 0 | 0 |
| routecode_no_new_model | Intern-S1-mini | 0 | 3 | 0.7414 | 0.0052 | 0.7362 | 0.7466 | 0.3233 | 0.0000 | 0 | 0 |
| routecode_no_new_model | Llama-3.1-8B-Instruct | 0 | 3 | 0.7299 | 0.0070 | 0.7259 | 0.7379 | 0.2732 | 0.0000 | 0 | 0 |
| routecode_no_new_model | MiniCPM4.1-8B | 0 | 3 | 0.7425 | 0.0072 | 0.7345 | 0.7483 | 0.3283 | 0.0000 | 0 | 0 |
| routecode_no_new_model | Qwen2.5-Coder-7B-Instruct | 0 | 3 | 0.6828 | 0.0030 | 0.6793 | 0.6845 | 0.0677 | 0.0000 | 0 | 0 |
| routecode_no_new_model | Qwen3-8B | 0 | 3 | 0.7305 | 0.0072 | 0.7224 | 0.7362 | 0.2757 | 0.0000 | 0 | 0 |
| uniform_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.6241 | 0.0268 | 0.5966 | 0.6500 | -0.1880 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.7040 | 0.0127 | 0.6897 | 0.7138 | 0.1604 | 32.0000 | 32 | 32 |
| uniform_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.6948 | 0.0210 | 0.6707 | 0.7086 | 0.1203 | 63.3333 | 62 | 64 |
| uniform_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.7213 | 0.0010 | 0.7207 | 0.7224 | 0.2356 | 126.0000 | 122 | 128 |
| uniform_route_state_calibration | Intern-S1-mini | 1 | 3 | 0.6770 | 0.0742 | 0.5914 | 0.7207 | 0.0426 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | Intern-S1-mini | 2 | 3 | 0.7000 | 0.0192 | 0.6793 | 0.7172 | 0.1429 | 32.0000 | 32 | 32 |
| uniform_route_state_calibration | Intern-S1-mini | 4 | 3 | 0.6943 | 0.0140 | 0.6845 | 0.7103 | 0.1178 | 64.0000 | 64 | 64 |
| uniform_route_state_calibration | Intern-S1-mini | 8 | 3 | 0.7264 | 0.0147 | 0.7121 | 0.7414 | 0.2581 | 127.0000 | 126 | 128 |
| uniform_route_state_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.5862 | 0.0340 | 0.5483 | 0.6138 | -0.3534 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.6925 | 0.0548 | 0.6293 | 0.7259 | 0.1103 | 31.3333 | 30 | 32 |
| uniform_route_state_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.7098 | 0.0020 | 0.7086 | 0.7121 | 0.1855 | 61.0000 | 58 | 64 |
| uniform_route_state_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7259 | 0.0121 | 0.7138 | 0.7379 | 0.2556 | 118.3333 | 113 | 125 |
| uniform_route_state_calibration | MiniCPM4.1-8B | 1 | 3 | 0.6937 | 0.0426 | 0.6466 | 0.7293 | 0.1153 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | MiniCPM4.1-8B | 2 | 3 | 0.6672 | 0.0553 | 0.6345 | 0.7310 | 0.0000 | 31.3333 | 31 | 32 |
| uniform_route_state_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7276 | 0.0243 | 0.7017 | 0.7500 | 0.2632 | 61.3333 | 59 | 64 |
| uniform_route_state_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7328 | 0.0017 | 0.7310 | 0.7345 | 0.2857 | 119.3333 | 115 | 125 |
| uniform_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.6420 | 0.0316 | 0.6172 | 0.6776 | -0.1103 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.6672 | 0.0466 | 0.6224 | 0.7155 | 0.0000 | 31.6667 | 31 | 32 |
| uniform_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.6822 | 0.0251 | 0.6586 | 0.7086 | 0.0652 | 62.6667 | 61 | 64 |
| uniform_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7241 | 0.0216 | 0.7034 | 0.7466 | 0.2481 | 123.3333 | 121 | 127 |
| uniform_route_state_calibration | Qwen3-8B | 1 | 3 | 0.7138 | 0.0216 | 0.6914 | 0.7345 | 0.2030 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | Qwen3-8B | 2 | 3 | 0.7339 | 0.0208 | 0.7121 | 0.7534 | 0.2907 | 32.0000 | 32 | 32 |
| uniform_route_state_calibration | Qwen3-8B | 4 | 3 | 0.7282 | 0.0259 | 0.6983 | 0.7431 | 0.2657 | 63.3333 | 62 | 64 |
| uniform_route_state_calibration | Qwen3-8B | 8 | 3 | 0.7149 | 0.0131 | 0.7000 | 0.7241 | 0.2080 | 126.0000 | 122 | 128 |

Active vs Uniform Deltas:

| new_model_id | examples_per_label | replicates | active_minus_uniform_mean_utility_mean | active_minus_uniform_mean_utility_std | active_minus_uniform_mean_utility_min | active_minus_uniform_mean_utility_max | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.0598 | 0.0605 | -0.0034 | 0.1172 | 16.0000 | 16 | 16 |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | -0.0247 | 0.0320 | -0.0466 | 0.0121 | 32.0000 | 32 | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.0322 | 0.0364 | 0.0086 | 0.0741 | 63.3333 | 62 | 64 |
| DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | -0.0483 | 0.0689 | -0.1224 | 0.0138 | 126.0000 | 122 | 128 |
| Intern-S1-mini | 1 | 3 | -0.0195 | 0.0940 | -0.1155 | 0.0724 | 16.0000 | 16 | 16 |
| Intern-S1-mini | 2 | 3 | 0.0195 | 0.0258 | -0.0017 | 0.0483 | 32.0000 | 32 | 32 |
| Intern-S1-mini | 4 | 3 | 0.0155 | 0.0389 | -0.0086 | 0.0603 | 64.0000 | 64 | 64 |
| Intern-S1-mini | 8 | 3 | 0.0069 | 0.0119 | 0.0000 | 0.0207 | 127.0000 | 126 | 128 |
| Llama-3.1-8B-Instruct | 1 | 3 | 0.0299 | 0.0484 | -0.0207 | 0.0759 | 16.0000 | 16 | 16 |
| Llama-3.1-8B-Instruct | 2 | 3 | 0.0086 | 0.0261 | -0.0121 | 0.0379 | 31.3333 | 30 | 32 |
| Llama-3.1-8B-Instruct | 4 | 3 | -0.0224 | 0.0544 | -0.0845 | 0.0172 | 61.0000 | 58 | 64 |
| Llama-3.1-8B-Instruct | 8 | 3 | -0.0149 | 0.0043 | -0.0190 | -0.0103 | 118.3333 | 113 | 125 |
| MiniCPM4.1-8B | 1 | 3 | 0.0368 | 0.0506 | -0.0086 | 0.0914 | 16.0000 | 16 | 16 |
| MiniCPM4.1-8B | 2 | 3 | 0.0632 | 0.0667 | -0.0138 | 0.1017 | 31.3333 | 31 | 32 |
| MiniCPM4.1-8B | 4 | 3 | -0.0023 | 0.0229 | -0.0155 | 0.0241 | 61.3333 | 59 | 64 |
| MiniCPM4.1-8B | 8 | 3 | 0.0052 | 0.0079 | -0.0017 | 0.0138 | 119.3333 | 115 | 125 |
| Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.0000 | 0.0192 | -0.0172 | 0.0207 | 16.0000 | 16 | 16 |
| Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.0385 | 0.0493 | 0.0034 | 0.0948 | 31.6667 | 31 | 32 |
| Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.0460 | 0.0319 | 0.0259 | 0.0828 | 62.6667 | 61 | 64 |
| Qwen2.5-Coder-7B-Instruct | 8 | 3 | -0.0052 | 0.0255 | -0.0345 | 0.0121 | 123.3333 | 121 | 127 |
| Qwen3-8B | 1 | 3 | -0.0092 | 0.0088 | -0.0190 | -0.0017 | 16.0000 | 16 | 16 |
| Qwen3-8B | 2 | 3 | -0.0195 | 0.0087 | -0.0276 | -0.0103 | 32.0000 | 32 | 32 |
| Qwen3-8B | 4 | 3 | 0.0000 | 0.0069 | -0.0069 | 0.0069 | 63.3333 | 62 | 64 |
| Qwen3-8B | 8 | 3 | -0.0000 | 0.0344 | -0.0397 | 0.0224 | 126.0000 | 122 | 128 |

Active vs Random Deltas:

| new_model_id | examples_per_label | replicates | active_minus_random_mean_utility_mean | active_minus_random_mean_utility_std | active_minus_random_mean_utility_min | active_minus_random_mean_utility_max | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | -0.0316 | 0.0286 | -0.0586 | -0.0017 | 16.0000 | 16 | 16 |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | -0.0236 | 0.0607 | -0.0603 | 0.0466 | 32.0000 | 32 | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.0247 | 0.0111 | 0.0121 | 0.0328 | 63.3333 | 62 | 64 |
| DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | -0.0546 | 0.0789 | -0.1448 | 0.0017 | 126.0000 | 122 | 128 |
| Intern-S1-mini | 1 | 3 | -0.0770 | 0.0513 | -0.1241 | -0.0224 | 16.0000 | 16 | 16 |
| Intern-S1-mini | 2 | 3 | -0.0172 | 0.0225 | -0.0328 | 0.0086 | 32.0000 | 32 | 32 |
| Intern-S1-mini | 4 | 3 | -0.0264 | 0.0264 | -0.0466 | 0.0034 | 64.0000 | 64 | 64 |
| Intern-S1-mini | 8 | 3 | -0.0080 | 0.0043 | -0.0121 | -0.0034 | 127.0000 | 126 | 128 |
| Llama-3.1-8B-Instruct | 1 | 3 | -0.0661 | 0.0267 | -0.0966 | -0.0466 | 16.0000 | 16 | 16 |
| Llama-3.1-8B-Instruct | 2 | 3 | -0.0121 | 0.0449 | -0.0586 | 0.0310 | 31.3333 | 30 | 32 |
| Llama-3.1-8B-Instruct | 4 | 3 | -0.0195 | 0.0533 | -0.0810 | 0.0121 | 61.0000 | 58 | 64 |
| Llama-3.1-8B-Instruct | 8 | 3 | 0.0040 | 0.0177 | -0.0155 | 0.0190 | 118.3333 | 113 | 125 |
| MiniCPM4.1-8B | 1 | 3 | 0.0011 | 0.0147 | -0.0138 | 0.0155 | 16.0000 | 16 | 16 |
| MiniCPM4.1-8B | 2 | 3 | -0.0000 | 0.0250 | -0.0241 | 0.0259 | 31.3333 | 31 | 32 |
| MiniCPM4.1-8B | 4 | 3 | -0.0034 | 0.0287 | -0.0362 | 0.0172 | 61.3333 | 59 | 64 |
| MiniCPM4.1-8B | 8 | 3 | -0.0040 | 0.0088 | -0.0138 | 0.0034 | 119.3333 | 115 | 125 |
| Qwen2.5-Coder-7B-Instruct | 1 | 3 | -0.0580 | 0.0184 | -0.0793 | -0.0466 | 16.0000 | 16 | 16 |
| Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.0063 | 0.0451 | -0.0414 | 0.0483 | 31.6667 | 31 | 32 |
| Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.0069 | 0.0105 | -0.0052 | 0.0138 | 62.6667 | 61 | 64 |
| Qwen2.5-Coder-7B-Instruct | 8 | 3 | -0.0109 | 0.0164 | -0.0276 | 0.0052 | 123.3333 | 121 | 127 |
| Qwen3-8B | 1 | 3 | -0.0213 | 0.0061 | -0.0276 | -0.0155 | 16.0000 | 16 | 16 |
| Qwen3-8B | 2 | 3 | -0.0052 | 0.0306 | -0.0397 | 0.0190 | 32.0000 | 32 | 32 |
| Qwen3-8B | 4 | 3 | -0.0029 | 0.0345 | -0.0362 | 0.0328 | 63.3333 | 62 | 64 |
| Qwen3-8B | 8 | 3 | -0.0138 | 0.0182 | -0.0276 | 0.0069 | 126.0000 | 122 | 128 |

Active vs Dataset Deltas:

| new_model_id | examples_per_label | replicates | active_minus_dataset_mean_utility_mean | active_minus_dataset_mean_utility_std | active_minus_dataset_mean_utility_min | active_minus_dataset_mean_utility_max | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | -0.0420 | 0.0242 | -0.0690 | -0.0224 | 16.0000 | 16 | 16 |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | -0.0460 | 0.0292 | -0.0707 | -0.0138 | 32.0000 | 32 | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.0138 | 0.0121 | 0.0017 | 0.0259 | 63.3333 | 62 | 64 |
| DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | -0.0534 | 0.0701 | -0.1328 | 0.0000 | 126.0000 | 122 | 128 |
| Intern-S1-mini | 1 | 3 | -0.0385 | 0.0166 | -0.0534 | -0.0207 | 16.0000 | 16 | 16 |
| Intern-S1-mini | 2 | 3 | 0.0000 | 0.0288 | -0.0310 | 0.0259 | 32.0000 | 32 | 32 |
| Intern-S1-mini | 4 | 3 | -0.0276 | 0.0284 | -0.0448 | 0.0052 | 64.0000 | 64 | 64 |
| Intern-S1-mini | 8 | 3 | -0.0006 | 0.0078 | -0.0086 | 0.0069 | 127.0000 | 126 | 128 |
| Llama-3.1-8B-Instruct | 1 | 3 | -0.0897 | 0.0316 | -0.1172 | -0.0552 | 16.0000 | 16 | 16 |
| Llama-3.1-8B-Instruct | 2 | 3 | -0.0075 | 0.0219 | -0.0328 | 0.0052 | 31.3333 | 30 | 32 |
| Llama-3.1-8B-Instruct | 4 | 3 | -0.0264 | 0.0562 | -0.0879 | 0.0224 | 61.0000 | 58 | 64 |
| Llama-3.1-8B-Instruct | 8 | 3 | -0.0034 | 0.0244 | -0.0224 | 0.0241 | 118.3333 | 113 | 125 |
| MiniCPM4.1-8B | 1 | 3 | 0.0322 | 0.0618 | -0.0069 | 0.1034 | 16.0000 | 16 | 16 |
| MiniCPM4.1-8B | 2 | 3 | 0.0253 | 0.0394 | 0.0000 | 0.0707 | 31.3333 | 31 | 32 |
| MiniCPM4.1-8B | 4 | 3 | -0.0132 | 0.0065 | -0.0207 | -0.0086 | 61.3333 | 59 | 64 |
| MiniCPM4.1-8B | 8 | 3 | -0.0040 | 0.0078 | -0.0121 | 0.0034 | 119.3333 | 115 | 125 |
| Qwen2.5-Coder-7B-Instruct | 1 | 3 | -0.0489 | 0.0320 | -0.0845 | -0.0224 | 16.0000 | 16 | 16 |
| Qwen2.5-Coder-7B-Instruct | 2 | 3 | -0.0052 | 0.0108 | -0.0172 | 0.0034 | 31.6667 | 31 | 32 |
| Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.0092 | 0.0208 | -0.0103 | 0.0310 | 62.6667 | 61 | 64 |
| Qwen2.5-Coder-7B-Instruct | 8 | 3 | -0.0126 | 0.0207 | -0.0328 | 0.0086 | 123.3333 | 121 | 127 |
| Qwen3-8B | 1 | 3 | 0.0086 | 0.0121 | 0.0000 | 0.0224 | 16.0000 | 16 | 16 |
| Qwen3-8B | 2 | 3 | -0.0138 | 0.0150 | -0.0310 | -0.0034 | 32.0000 | 32 | 32 |
| Qwen3-8B | 4 | 3 | 0.0247 | 0.0117 | 0.0155 | 0.0379 | 63.3333 | 62 | 64 |
| Qwen3-8B | 8 | 3 | -0.0132 | 0.0334 | -0.0517 | 0.0086 | 126.0000 | 122 | 128 |

Active vs Embedding Deltas:

| new_model_id | examples_per_label | replicates | active_minus_embedding_mean_utility_mean | active_minus_embedding_mean_utility_std | active_minus_embedding_mean_utility_min | active_minus_embedding_mean_utility_max | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | -0.0017 | 0.0629 | -0.0431 | 0.0707 | 16.0000 | 16 | 16 |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | -0.0023 | 0.0431 | -0.0397 | 0.0448 | 32.0000 | 32 | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.0351 | 0.0204 | 0.0224 | 0.0586 | 63.3333 | 62 | 64 |
| DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.0000 | 0.0756 | -0.0552 | 0.0862 | 126.0000 | 122 | 128 |
| Intern-S1-mini | 1 | 3 | -0.0397 | 0.0294 | -0.0724 | -0.0155 | 16.0000 | 16 | 16 |
| Intern-S1-mini | 2 | 3 | 0.0040 | 0.0366 | -0.0259 | 0.0448 | 32.0000 | 32 | 32 |
| Intern-S1-mini | 4 | 3 | -0.0270 | 0.0286 | -0.0569 | 0.0000 | 64.0000 | 64 | 64 |
| Intern-S1-mini | 8 | 3 | 0.0109 | 0.0222 | -0.0052 | 0.0362 | 127.0000 | 126 | 128 |
| Llama-3.1-8B-Instruct | 1 | 3 | -0.0793 | 0.0147 | -0.0931 | -0.0638 | 16.0000 | 16 | 16 |
| Llama-3.1-8B-Instruct | 2 | 3 | -0.0190 | 0.0254 | -0.0466 | 0.0034 | 31.3333 | 30 | 32 |
| Llama-3.1-8B-Instruct | 4 | 3 | -0.0138 | 0.0610 | -0.0810 | 0.0379 | 61.0000 | 58 | 64 |
| Llama-3.1-8B-Instruct | 8 | 3 | -0.0034 | 0.0121 | -0.0155 | 0.0086 | 118.3333 | 113 | 125 |
| MiniCPM4.1-8B | 1 | 3 | 0.0075 | 0.0190 | -0.0052 | 0.0293 | 16.0000 | 16 | 16 |
| MiniCPM4.1-8B | 2 | 3 | 0.0092 | 0.0249 | -0.0052 | 0.0379 | 31.3333 | 31 | 32 |
| MiniCPM4.1-8B | 4 | 3 | -0.0149 | 0.0105 | -0.0241 | -0.0034 | 61.3333 | 59 | 64 |
| MiniCPM4.1-8B | 8 | 3 | 0.0000 | 0.0079 | -0.0086 | 0.0069 | 119.3333 | 115 | 125 |
| Qwen2.5-Coder-7B-Instruct | 1 | 3 | -0.0667 | 0.0287 | -0.0931 | -0.0362 | 16.0000 | 16 | 16 |
| Qwen2.5-Coder-7B-Instruct | 2 | 3 | -0.0011 | 0.0218 | -0.0207 | 0.0224 | 31.6667 | 31 | 32 |
| Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.0109 | 0.0252 | -0.0121 | 0.0379 | 62.6667 | 61 | 64 |
| Qwen2.5-Coder-7B-Instruct | 8 | 3 | -0.0023 | 0.0252 | -0.0293 | 0.0207 | 123.3333 | 121 | 127 |
| Qwen3-8B | 1 | 3 | 0.0224 | 0.0096 | 0.0138 | 0.0328 | 16.0000 | 16 | 16 |
| Qwen3-8B | 2 | 3 | -0.0195 | 0.0230 | -0.0345 | 0.0069 | 32.0000 | 32 | 32 |
| Qwen3-8B | 4 | 3 | 0.0052 | 0.0340 | -0.0328 | 0.0328 | 63.3333 | 62 | 64 |
| Qwen3-8B | 8 | 3 | -0.0069 | 0.0147 | -0.0224 | 0.0069 | 126.0000 | 122 | 128 |

Interpretation:

Across paired active-vs-uniform rows, active calibration has mean utility delta `0.0082` over `72` pairs (`35` positive, `32` negative, `5` tied).

Across paired active-vs-random rows, active calibration has mean utility delta `-0.0172` over `72` pairs (`27` positive, `45` negative, `0` tied).

Across paired active-vs-dataset rows, active calibration has mean utility delta `-0.0138` over `72` pairs (`26` positive, `41` negative, `5` tied).

Across paired active-vs-embedding rows, active calibration has mean utility delta `-0.0080` over `72` pairs (`29` positive, `42` negative, `1` tied).
