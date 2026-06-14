# Phase D4/E5 New-Model Calibration Memo

Command: `python experiments/07_new_model_calibration.py --config configs/synthetic.yaml`

Route labels: predictability-constrained RouteCode, K = 16, alpha = 3, beta = 0.

This is a simulated held-out-model calibration using existing outcome tables. It makes no external API calls.

## Mean Across Held-Out Models

| method | examples_per_label | mean_utility | recovered_gap_vs_oracle | calibration_query_count |
| --- | --- | --- | --- | --- |
| direct_retraining_budgeted_gradient_boosting | 1 | 0.5945 | 0.7633 | 16.0000 |
| direct_retraining_budgeted_gradient_boosting | 2 | 0.5947 | 0.7638 | 32.0000 |
| direct_retraining_budgeted_gradient_boosting | 4 | 0.5949 | 0.7645 | 64.0000 |
| direct_retraining_budgeted_gradient_boosting | 8 | 0.5977 | 0.7725 | 128.0000 |
| direct_retraining_budgeted_gradient_boosting | 16 | 0.5998 | 0.7787 | 256.0000 |
| direct_retraining_budgeted_gradient_boosting | 32 | 0.6102 | 0.8087 | 497.0000 |
| direct_retraining_budgeted_gradient_boosting | 64 | 0.6387 | 0.8916 | 900.6667 |
| direct_retraining_budgeted_knn | 1 | 0.6017 | 0.7841 | 16.0000 |
| direct_retraining_budgeted_knn | 2 | 0.6015 | 0.7835 | 32.0000 |
| direct_retraining_budgeted_knn | 4 | 0.6015 | 0.7836 | 64.0000 |
| direct_retraining_budgeted_knn | 8 | 0.6024 | 0.7863 | 128.0000 |
| direct_retraining_budgeted_knn | 16 | 0.6040 | 0.7908 | 256.0000 |
| direct_retraining_budgeted_knn | 32 | 0.6153 | 0.8237 | 497.0000 |
| direct_retraining_budgeted_knn | 64 | 0.6464 | 0.9140 | 900.6667 |
| direct_retraining_budgeted_logistic | 1 | 0.5950 | 0.7648 | 16.0000 |
| direct_retraining_budgeted_logistic | 2 | 0.5953 | 0.7654 | 32.0000 |
| direct_retraining_budgeted_logistic | 4 | 0.5951 | 0.7650 | 64.0000 |
| direct_retraining_budgeted_logistic | 8 | 0.5957 | 0.7668 | 128.0000 |
| direct_retraining_budgeted_logistic | 16 | 0.5965 | 0.7691 | 256.0000 |
| direct_retraining_budgeted_logistic | 32 | 0.6056 | 0.7955 | 497.0000 |
| direct_retraining_budgeted_logistic | 64 | 0.6363 | 0.8844 | 900.6667 |
| direct_retraining_budgeted_mlp | 1 | 0.5784 | 0.7167 | 16.0000 |
| direct_retraining_budgeted_mlp | 2 | 0.5785 | 0.7169 | 32.0000 |
| direct_retraining_budgeted_mlp | 4 | 0.5784 | 0.7166 | 64.0000 |
| direct_retraining_budgeted_mlp | 8 | 0.5786 | 0.7171 | 128.0000 |
| direct_retraining_budgeted_mlp | 16 | 0.5858 | 0.7379 | 256.0000 |
| direct_retraining_budgeted_mlp | 32 | 0.5937 | 0.7608 | 497.0000 |
| direct_retraining_budgeted_mlp | 64 | 0.6120 | 0.8141 | 900.6667 |
| direct_retraining_budgeted_svm | 1 | 0.5960 | 0.7676 | 16.0000 |
| direct_retraining_budgeted_svm | 2 | 0.5961 | 0.7678 | 32.0000 |
| direct_retraining_budgeted_svm | 4 | 0.5956 | 0.7665 | 64.0000 |
| direct_retraining_budgeted_svm | 8 | 0.5958 | 0.7669 | 128.0000 |
| direct_retraining_budgeted_svm | 16 | 0.5966 | 0.7693 | 256.0000 |
| direct_retraining_budgeted_svm | 32 | 0.6034 | 0.7889 | 497.0000 |
| direct_retraining_budgeted_svm | 64 | 0.6364 | 0.8848 | 900.6667 |
| routecode_label_calibration | 1 | 0.6661 | 0.9711 | 16.0000 |
| routecode_label_calibration | 2 | 0.6660 | 0.9706 | 32.0000 |
| routecode_label_calibration | 4 | 0.6660 | 0.9706 | 64.0000 |
| routecode_label_calibration | 8 | 0.6660 | 0.9706 | 128.0000 |
| routecode_label_calibration | 16 | 0.6660 | 0.9706 | 256.0000 |
| routecode_label_calibration | 32 | 0.6660 | 0.9706 | 497.0000 |
| routecode_label_calibration | 64 | 0.6660 | 0.9706 | 900.6667 |

## Current Readout

- Held-out/new models: `code_7b`, `frontier_expensive`, `general_8b`, `math_7b`, `reasoner_13b`, `tiny_cheap`.
- Direct retraining baselines: `gradient_boosting`, `knn`, `logistic`, `mlp`, `svm`.
- Best budgeted row: `routecode_label_calibration` for `reasoner_13b` at r `1`, mean utility `0.6669` with `16` new-model evaluations.
- Interpret this as a sample-efficiency diagnostic only after comparing against the direct retraining curve.
- A strong claim requires RouteCode to reach competitive utility with fewer new-model evaluations than direct retraining across held-out models.
