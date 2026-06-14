# Phase D4/E5 New-Model Calibration Memo

Command: `python experiments/07_new_model_calibration.py --config configs/llmrouterbench_pilot.yaml`

Route labels: predictability-constrained RouteCode, K = 16, alpha = 3, beta = 0.

This is a simulated held-out-model calibration using existing outcome tables. It makes no external API calls.

## Mean Across Held-Out Models

| method | examples_per_label | mean_utility | recovered_gap_vs_oracle | calibration_query_count |
| --- | --- | --- | --- | --- |
| direct_retraining_budgeted_gradient_boosting | 1 | 0.6575 | -0.0426 | 16.0000 |
| direct_retraining_budgeted_gradient_boosting | 2 | 0.6578 | -0.0414 | 31.8333 |
| direct_retraining_budgeted_gradient_boosting | 4 | 0.6572 | -0.0439 | 63.3333 |
| direct_retraining_budgeted_gradient_boosting | 8 | 0.6578 | -0.0414 | 124.8333 |
| direct_retraining_budgeted_gradient_boosting | 16 | 0.6580 | -0.0401 | 231.8333 |
| direct_retraining_budgeted_gradient_boosting | 32 | 0.6563 | -0.0476 | 411.8333 |
| direct_retraining_budgeted_gradient_boosting | 64 | 0.6580 | -0.0401 | 658.0000 |
| direct_retraining_budgeted_knn | 1 | 0.6546 | -0.0551 | 16.0000 |
| direct_retraining_budgeted_knn | 2 | 0.6546 | -0.0551 | 31.8333 |
| direct_retraining_budgeted_knn | 4 | 0.6546 | -0.0551 | 63.3333 |
| direct_retraining_budgeted_knn | 8 | 0.6546 | -0.0551 | 124.8333 |
| direct_retraining_budgeted_knn | 16 | 0.6546 | -0.0551 | 231.8333 |
| direct_retraining_budgeted_knn | 32 | 0.6546 | -0.0551 | 411.8333 |
| direct_retraining_budgeted_knn | 64 | 0.6546 | -0.0551 | 658.0000 |
| direct_retraining_budgeted_logistic | 1 | 0.6626 | -0.0201 | 16.0000 |
| direct_retraining_budgeted_logistic | 2 | 0.6626 | -0.0201 | 31.8333 |
| direct_retraining_budgeted_logistic | 4 | 0.6624 | -0.0213 | 63.3333 |
| direct_retraining_budgeted_logistic | 8 | 0.6624 | -0.0213 | 124.8333 |
| direct_retraining_budgeted_logistic | 16 | 0.6621 | -0.0226 | 231.8333 |
| direct_retraining_budgeted_logistic | 32 | 0.6624 | -0.0213 | 411.8333 |
| direct_retraining_budgeted_logistic | 64 | 0.6624 | -0.0213 | 658.0000 |
| direct_retraining_budgeted_mlp | 1 | 0.6644 | -0.0125 | 16.0000 |
| direct_retraining_budgeted_mlp | 2 | 0.6672 | 0.0000 | 31.8333 |
| direct_retraining_budgeted_mlp | 4 | 0.6641 | -0.0138 | 63.3333 |
| direct_retraining_budgeted_mlp | 8 | 0.6609 | -0.0276 | 124.8333 |
| direct_retraining_budgeted_mlp | 16 | 0.6598 | -0.0326 | 231.8333 |
| direct_retraining_budgeted_mlp | 32 | 0.6575 | -0.0426 | 411.8333 |
| direct_retraining_budgeted_mlp | 64 | 0.6557 | -0.0501 | 658.0000 |
| direct_retraining_budgeted_svm | 1 | 0.6563 | -0.0476 | 16.0000 |
| direct_retraining_budgeted_svm | 2 | 0.6566 | -0.0464 | 31.8333 |
| direct_retraining_budgeted_svm | 4 | 0.6560 | -0.0489 | 63.3333 |
| direct_retraining_budgeted_svm | 8 | 0.6563 | -0.0476 | 124.8333 |
| direct_retraining_budgeted_svm | 16 | 0.6560 | -0.0489 | 231.8333 |
| direct_retraining_budgeted_svm | 32 | 0.6543 | -0.0564 | 411.8333 |
| direct_retraining_budgeted_svm | 64 | 0.6549 | -0.0539 | 658.0000 |
| routecode_label_calibration | 1 | 0.6856 | 0.0802 | 16.0000 |
| routecode_label_calibration | 2 | 0.6980 | 0.1341 | 31.8333 |
| routecode_label_calibration | 4 | 0.7305 | 0.2757 | 63.3333 |
| routecode_label_calibration | 8 | 0.7124 | 0.1967 | 124.8333 |
| routecode_label_calibration | 16 | 0.7305 | 0.2757 | 231.8333 |
| routecode_label_calibration | 32 | 0.7374 | 0.3058 | 411.8333 |
| routecode_label_calibration | 64 | 0.7319 | 0.2820 | 658.0000 |

## Current Readout

- Held-out/new models: `DeepSeek-R1-Distill-Qwen-7B`, `Intern-S1-mini`, `Llama-3.1-8B-Instruct`, `MiniCPM4.1-8B`, `Qwen2.5-Coder-7B-Instruct`, `Qwen3-8B`.
- Direct retraining baselines: `gradient_boosting`, `knn`, `logistic`, `mlp`, `svm`.
- Best budgeted row: `routecode_label_calibration` for `Qwen2.5-Coder-7B-Instruct` at r `64`, mean utility `0.7431` with `607` new-model evaluations.
- Interpret this as a sample-efficiency diagnostic only after comparing against the direct retraining curve.
- A strong claim requires RouteCode to reach competitive utility with fewer new-model evaluations than direct retraining across held-out models.
