# RouteCode Synthetic Demo

This run is a zero-API synthetic pilot. It is intended to verify code paths, leakage controls, metrics, and plot generation before any real benchmark download.

## Commands

```bash
python experiments/00_data_audit.py --config configs/synthetic.yaml
python experiments/01_compression_ladder.py --config configs/synthetic.yaml
python experiments/02_rate_distortion_curve.py --config configs/synthetic.yaml
python experiments/06_predictability_constrained.py --config configs/synthetic.yaml
python experiments/07_new_model_calibration.py --config configs/synthetic.yaml
python experiments/08_ablation_summary.py --config configs/synthetic.yaml
python experiments/09_sensitivity_suite.py --config configs/synthetic.yaml
python experiments/10_external_baseline_surrogates.py --config configs/synthetic.yaml
python experiments/11_code_card_interpretability.py --config configs/synthetic.yaml
pytest -q
```

## Outputs

- `table_routability.csv`: best single, cheapest, dataset-label lookup, and query-oracle audit.
- `table_recovered_gap.csv`: compression ladder with bootstrap confidence intervals.
- `table_rate_distortion.csv`: semantic-cluster and RouteCode curves for K = 1, 2, 4, 8, 16, 32, 64, 128.
- `code_cards.md`, `code_cards.json`, and `fig_code_label_heatmap.pdf`: train-set summaries for learned route labels.
- `fig_compression_ladder.pdf` and `fig_rate_distortion.pdf`: main pilot figures.
- `table_predictability_constrained.csv`, `fig_predictability_constrained_tradeoff.pdf`, and `code_cards_predictability_constrained.md`: predictability-constrained RouteCode diagnostics.
- `table_new_model_integration.csv` and `fig_transfer_calibration_curve.pdf`: simulated held-out/new-model calibration diagnostics.
- `table_ablation_summary.csv`, `fig_sensitivity_k_lambda.pdf`, and `fig_seed_stability.pdf`: bounded ablation and robustness diagnostics.
- `table_sensitivity_summary.csv`, `fig_sensitivity_summary.pdf`, and `phase_g_sensitivity_memo.md`: bounded Phase G sensitivity diagnostics.
- `table_external_baselines.csv` and `phase_e_external_baseline_memo.md`: local external-style baseline surrogate diagnostics.
- `table_code_card_interpretability.csv` and `phase_f_code_card_interpretability_memo.md`: label-only versus code-card observability diagnostics.
- `outcomes.csv` and `query_embeddings.csv`: canonical input rows and deterministic local query features used for this run.

## First Results

| method | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- |
| random | 0.2478 | 0.4283 | -0.2428 |
| best_single | 0.3315 | 0.3446 | 0.0000 |
| dataset_oracle | 0.6535 | 0.0226 | 0.9344 |
| kNN | 0.6663 | 0.0097 | 0.9717 |
| svm_embedding_router | 0.6623 | 0.0137 | 0.9601 |
| query_oracle | 0.6761 | 0.0000 | 1.0000 |

Utility-oracle RouteCode rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.3315 | 0.3446 | 0.0000 |
| 2 | 1.0000 | 0.9902 | 0.4824 | 0.1937 | 0.4381 |
| 4 | 2.0000 | 1.9150 | 0.6376 | 0.0385 | 0.8884 |
| 8 | 3.0000 | 2.9765 | 0.6604 | 0.0157 | 0.9544 |
| 16 | 4.0000 | 3.9156 | 0.6673 | 0.0088 | 0.9746 |
| 32 | 5.0000 | 4.8534 | 0.6688 | 0.0073 | 0.9789 |
| 64 | 6.0000 | 5.7950 | 0.6693 | 0.0068 | 0.9802 |
| 128 | 7.0000 | 6.6415 | 0.6709 | 0.0052 | 0.9848 |

Regret-objective RouteCode oracle rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.3315 | 0.3446 | 0.0000 |
| 2 | 1.0000 | 0.9915 | 0.5108 | 0.1653 | 0.5203 |
| 4 | 2.0000 | 1.9255 | 0.6482 | 0.0279 | 0.9189 |
| 8 | 3.0000 | 2.7010 | 0.6761 | 0.0000 | 1.0000 |
| 16 | 4.0000 | 3.4452 | 0.6761 | 0.0000 | 1.0000 |
| 32 | 5.0000 | 4.5485 | 0.6761 | 0.0000 | 1.0000 |
| 64 | 6.0000 | 5.6410 | 0.6761 | 0.0000 | 1.0000 |
| 128 | 7.0000 | 6.5522 | 0.6761 | 0.0000 | 1.0000 |

Predicted RouteCode rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.3315 | 0.3446 | 0.0000 |
| 2 | 1.0000 | 0.9846 | 0.4760 | 0.2000 | 0.4195 |
| 4 | 2.0000 | 1.9180 | 0.6195 | 0.0566 | 0.8357 |
| 8 | 3.0000 | 2.9775 | 0.6421 | 0.0340 | 0.9015 |
| 16 | 4.0000 | 3.8035 | 0.6504 | 0.0257 | 0.9255 |
| 32 | 5.0000 | 4.5617 | 0.6523 | 0.0238 | 0.9309 |
| 64 | 6.0000 | 5.4058 | 0.6554 | 0.0207 | 0.9401 |
| 128 | 7.0000 | 6.1821 | 0.6489 | 0.0272 | 0.9211 |

These values come from synthetic data only and should not be used as novelty or paper-evidence claims.

## External References Checked

The synthetic run did not execute external repositories or download benchmark data. These papers/repos were inspected as novelty boundaries and future baseline sources:

- LLMRouterBench; paper: https://arxiv.org/abs/2601.07206; repo: https://github.com/ynulihao/LLMRouterBench
- RouteLLM; paper: https://arxiv.org/abs/2406.18665; repo: https://github.com/lm-sys/routellm
- LLMRouter; repo: https://github.com/ulab-uiuc/LLMRouter
- RouterBench; paper: https://arxiv.org/abs/2403.12031; repo: https://github.com/withmartian/routerbench
- WebRouter; paper: https://arxiv.org/abs/2510.11221
- FineRouter; paper: https://arxiv.org/abs/2603.19415
- BEST-Route; paper: https://openreview.net/forum?id=tFBIbCVXkG; repo: https://github.com/microsoft/best-route-llm
- GraphRouter; paper: https://openreview.net/forum?id=eU39PDsZtT; repo: https://github.com/ulab-uiuc/LLMRouter
- Universal Model Routing; paper: https://openreview.net/pdf?id=ka82fvJ5f1
- kNN routing; paper: https://arxiv.org/abs/2505.12601; repo: https://github.com/ulab-uiuc/LLMRouter
- Causal LLM Routing; paper: https://openreview.net/forum?id=iZC5xoQQkX

## Leakage Controls

- Train/validation/test splits are assigned by `query_id`; all model rows for a query stay in the same split.
- Best-single, dataset/topic tables, embedding clusters, kNN neighbors, and RouteCode codebooks are fit on train only.
- Query oracle uses test utility only as an upper bound.
- The leaky dataset-label diagnostic is written separately to `table_leakage_gap.csv` and is not a deployable baseline.

## Next Steps

1. Add real loaders only after this synthetic path remains green.
2. Convert LLMRouterBench outcomes to the canonical schema.
3. Add real-benchmark loaders and keep logistic/MLP learned-router rows as first internal baselines.
4. Run split robustness before making benchmark compressibility claims.
5. Add held-out model calibration simulation for the sample-efficiency claim.
