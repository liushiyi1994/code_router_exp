# RouteCode LLMRouterBench Pilot

This run uses `llmrouterbench` records converted into the RouteCode canonical schema. No model generation or external API calls are made by these evaluation scripts.

## Commands

```bash
python experiments/00_data_audit.py --config configs/llmrouterbench_pilot.yaml
python experiments/01_compression_ladder.py --config configs/llmrouterbench_pilot.yaml
python experiments/02_rate_distortion_curve.py --config configs/llmrouterbench_pilot.yaml
python experiments/03_residual_concentration.py --config configs/llmrouterbench_pilot.yaml
python experiments/04_split_sensitivity.py --config configs/llmrouterbench_pilot.yaml
python experiments/05_predictor_diagnostics.py --config configs/llmrouterbench_pilot.yaml
python experiments/06_predictability_constrained.py --config configs/llmrouterbench_pilot.yaml
python experiments/07_new_model_calibration.py --config configs/llmrouterbench_pilot.yaml
python experiments/08_ablation_summary.py --config configs/llmrouterbench_pilot.yaml
python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_pilot.yaml
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_pilot.yaml
python experiments/11_code_card_interpretability.py --config configs/llmrouterbench_pilot.yaml
python experiments/12_official_baseline_artifacts.py --config configs/llmrouterbench_pilot.yaml
python experiments/13_transformer_backbone_readiness.py --config configs/llmrouterbench_pilot.yaml
python experiments/14_routellm_pairwise_alignment.py --config configs/llmrouterbench_pilot.yaml
python experiments/15_routellm_mf_assets.py --config configs/llmrouterbench_pilot.yaml
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_pilot.yaml
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_pilot.yaml
python experiments/18_model_pool_scale.py --config configs/llmrouterbench_scale20.yaml
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_scale20.yaml
python experiments/20_benchmark_coverage.py --config configs/llmrouterbench.yaml
python experiments/00_data_audit.py --config configs/llmrouterbench_broad20.yaml
python experiments/01_compression_ladder.py --config configs/llmrouterbench_broad20.yaml
python experiments/02_rate_distortion_curve.py --config configs/llmrouterbench_broad20.yaml
python experiments/06_predictability_constrained.py --config configs/llmrouterbench_broad20.yaml
python experiments/03_residual_concentration.py --config configs/llmrouterbench_broad20.yaml
python experiments/04_split_sensitivity.py --config configs/llmrouterbench_broad20.yaml
python experiments/08_ablation_summary.py --config configs/llmrouterbench_broad20.yaml
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_broad20.yaml
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_broad20.yaml
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_broad20.yaml
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_broad20.yaml
pytest -q
```

## Outputs

- `table_routability.csv`: best single, cheapest, dataset-label lookup, and query-oracle audit.
- `table_recovered_gap.csv`: compression ladder with bootstrap confidence intervals.
- `table_rate_distortion.csv`: semantic-cluster and RouteCode curves for K = 1, 2, 4, 8, 16, 32, 64, 128.
- `code_cards.md`, `code_cards.json`, and `fig_code_label_heatmap.pdf`: train-set summaries for learned route labels.
- `fig_compression_ladder.pdf` and `fig_rate_distortion.pdf`: main pilot figures.
- `table_residual_concentration.csv` and `fig_residual_concentration.pdf`: residual-regret concentration diagnostics.
- `table_split_sensitivity.csv`, `table_split_rank_correlation.csv`, `table_split_rate_threshold.csv`, and `fig_split_sensitivity.pdf`: split-sensitivity diagnostics.
- `table_predictor_comparison.csv`, `table_utility_weighted_confusion.csv`, `table_calibration_curve.csv`, `fig_utility_weighted_confusion.pdf`, and `fig_calibration_curve.pdf`: RouteCode label-predictor diagnostics.
- `table_predictability_constrained.csv`, `fig_predictability_constrained_tradeoff.pdf`, `code_cards_predictability_constrained.md`, and `phase_d_method_memo.md`: predictability-constrained RouteCode diagnostics.
- `table_new_model_integration.csv`, `fig_transfer_calibration_curve.pdf`, and `phase_e5_new_model_calibration_memo.md`: simulated held-out/new-model calibration diagnostics.
- `table_ablation_summary.csv`, `fig_sensitivity_k_lambda.pdf`, `fig_seed_stability.pdf`, and `phase_f_g_ablation_memo.md`: bounded ablation and robustness diagnostics.
- `table_sensitivity_summary.csv`, `fig_sensitivity_summary.pdf`, and `phase_g_sensitivity_memo.md`: bounded Phase G sensitivity diagnostics.
- `table_external_baselines.csv` and `phase_e_external_baseline_memo.md`: local external-style baseline surrogate diagnostics.
- `table_code_card_interpretability.csv` and `phase_f_code_card_interpretability_memo.md`: label-only versus code-card observability diagnostics.
- `table_official_external_artifacts.csv` and `phase_e_official_baseline_artifacts_memo.md`: official upstream baseline artifact inspection, not split-aligned RouteCode metrics.
- `table_transformer_backbone_readiness.csv` and `phase_f_g_transformer_backbone_readiness_memo.md`: cache-only transformer backbone readiness audit.
- `table_routellm_pairwise_alignment.csv` and `phase_e_routellm_pairwise_alignment_memo.md`: split-aligned RouteLLM pairwise substrate readiness audit.
- `table_routellm_mf_assets.csv` and `phase_e_routellm_mf_assets_memo.md`: split-aligned RouteLLM MF trainer asset readiness audit.
- `table_routellm_mf_split_aligned.csv` and `phase_e_routellm_mf_split_aligned_memo.md`: split-aligned RouteLLM MF training-code evaluation.
- `table_avengerspro_split_aligned.csv` and `phase_e_avengerspro_split_aligned_memo.md`: split-aligned local Avengers-Pro cluster-routing compatibility evaluation.
- `results/llmrouterbench_scale20/table_model_pool_scale.csv` and `results/llmrouterbench_scale20/phase_f_g_model_pool_scale_memo.md`: same-six-dataset, 20-model pool scale/composition robustness diagnostics.
- `results/llmrouterbench_scale20/table_model_pool_transfer.csv` and `results/llmrouterbench_scale20/phase_f_g_model_pool_transfer_memo.md`: same-six-dataset, disjoint 8-source/8-target model-pool transfer diagnostics.
- `table_benchmark_file_coverage.csv`, `table_benchmark_dataset_coverage.csv`, `table_broad_coverage_candidates.csv`, and `phase_g_benchmark_coverage_memo.md`: raw LLMRouterBench coverage and broad complete-rectangle diagnostics.
- `results/llmrouterbench_broad20/table_routability.csv`, `results/llmrouterbench_broad20/table_recovered_gap.csv`, `results/llmrouterbench_broad20/table_rate_distortion.csv`, and `results/llmrouterbench_broad20/phase_b_broad20_memo.md`: broad 18-dataset/20-model Phase B router metrics.
- `results/llmrouterbench_broad20/table_predictability_constrained.csv`, `results/llmrouterbench_broad20/fig_predictability_constrained_tradeoff.pdf`, `results/llmrouterbench_broad20/code_cards_predictability_constrained.md`, and `results/llmrouterbench_broad20/phase_d_method_memo.md`: broad D2 predictability-constrained RouteCode diagnostics.
- `results/llmrouterbench_broad20/table_residual_concentration.csv`, `results/llmrouterbench_broad20/table_residual_risk.csv`, `results/llmrouterbench_broad20/fig_residual_concentration.pdf`, `results/llmrouterbench_broad20/fig_risk_coverage.pdf`, and `results/llmrouterbench_broad20/phase_d5_adaptive_refinement_gate_memo.md`: broad residual/adaptive-refinement gate diagnostics.
- `results/llmrouterbench_broad20/table_split_sensitivity.csv`, `results/llmrouterbench_broad20/table_split_rank_correlation.csv`, `results/llmrouterbench_broad20/table_split_rate_threshold.csv`, and `results/llmrouterbench_broad20/fig_split_sensitivity.pdf`: bounded broad split-sensitivity diagnostics.
- `results/llmrouterbench_broad20/table_ablation_summary.csv`, `results/llmrouterbench_broad20/fig_sensitivity_k_lambda.pdf`, `results/llmrouterbench_broad20/fig_seed_stability.pdf`, and `results/llmrouterbench_broad20/phase_f_g_ablation_memo.md`: bounded broad ablation diagnostics.
- `results/llmrouterbench_broad20/table_model_pool_transfer.csv` and `results/llmrouterbench_broad20/phase_f_g_model_pool_transfer_memo.md`: bounded broad disjoint 8-source/8-target model-pool transfer diagnostics.
- `results/llmrouterbench_broad20/table_external_baselines.csv` and `results/llmrouterbench_broad20/phase_e_external_baseline_memo.md`: broad local external-style baseline surrogate diagnostics, not official upstream-command reproductions.
- `results/llmrouterbench_broad20/table_routellm_mf_split_aligned.csv`, `results/llmrouterbench_broad20/phase_e_routellm_mf_split_aligned_memo.md`, and `results/llmrouterbench_broad20/routellm_mf_split_aligned/`: broad RouteLLM MF official-code evaluation with local RouteCode embeddings, not the upstream published checkpoint.
- `results/llmrouterbench_broad20/table_avengerspro_split_aligned.csv`, `results/llmrouterbench_broad20/phase_e_avengerspro_split_aligned_memo.md`, and `results/llmrouterbench_broad20/avengerspro_split_aligned/`: broad local Avengers-Pro cluster-routing compatibility baseline, not an upstream command-path reproduction.
- `phase_c_observation_memo.md`: Phase C checkpoint memo answering the seven pilot questions.
- `outcomes.csv` and `query_embeddings.csv`: canonical input rows and deterministic local query features used for this run.

## First Results

| method | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- |
| random | 0.5724 | 0.3241 | -0.4135 |
| best_single | 0.6672 | 0.2293 | 0.0000 |
| dataset_oracle | 0.7638 | 0.1328 | 0.4211 |
| kNN | 0.7362 | 0.1603 | 0.3008 |
| svm_embedding_router | 0.6724 | 0.2241 | 0.0226 |
| query_oracle | 0.8966 | 0.0000 | 1.0000 |

Utility-oracle RouteCode rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.6672 | 0.2293 | 0.0000 |
| 2 | 1.0000 | 0.9525 | 0.6707 | 0.2259 | 0.0150 |
| 4 | 2.0000 | 1.9087 | 0.7448 | 0.1517 | 0.3383 |
| 8 | 3.0000 | 2.8035 | 0.8310 | 0.0655 | 0.7143 |
| 16 | 4.0000 | 3.6971 | 0.8897 | 0.0069 | 0.9699 |
| 32 | 5.0000 | 4.4593 | 0.8966 | 0.0000 | 1.0000 |
| 64 | 6.0000 | 4.9065 | 0.8931 | 0.0034 | 0.9850 |
| 128 | 7.0000 | 4.9065 | 0.8931 | 0.0034 | 0.9850 |

Regret-objective RouteCode oracle rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.6672 | 0.2293 | 0.0000 |
| 2 | 1.0000 | 0.8592 | 0.8155 | 0.0810 | 0.6466 |
| 4 | 2.0000 | 1.7460 | 0.8621 | 0.0345 | 0.8496 |
| 8 | 3.0000 | 2.3278 | 0.8966 | 0.0000 | 1.0000 |
| 16 | 4.0000 | 3.1482 | 0.8966 | 0.0000 | 1.0000 |
| 32 | 5.0000 | 3.6211 | 0.8966 | 0.0000 | 1.0000 |
| 64 | 6.0000 | 4.2554 | 0.8966 | 0.0000 | 1.0000 |
| 128 | 7.0000 | 4.2554 | 0.8966 | 0.0000 | 1.0000 |

Predicted RouteCode rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.6672 | 0.2293 | 0.0000 |
| 2 | 1.0000 | 0.9245 | 0.6552 | 0.2414 | -0.0526 |
| 4 | 2.0000 | 1.8645 | 0.6345 | 0.2621 | -0.1429 |
| 8 | 3.0000 | 2.7862 | 0.6121 | 0.2845 | -0.2406 |
| 16 | 4.0000 | 3.8289 | 0.6138 | 0.2828 | -0.2331 |
| 32 | 5.0000 | 4.2924 | 0.6362 | 0.2603 | -0.1353 |
| 64 | 6.0000 | 4.2391 | 0.6707 | 0.2259 | 0.0150 |
| 128 | 7.0000 | 4.2391 | 0.6707 | 0.2259 | 0.0150 |

These values come from a configured LLMRouterBench pilot subset. Treat them as pilot observations, not full-benchmark or paper-level claims.

## External References Checked

This run used released LLMRouterBench outcome JSONs and did not call external model APIs. These papers/repos define the data source, novelty boundaries, and future baseline sources:

- LLMRouterBench; paper: https://arxiv.org/abs/2601.07206; repo: https://github.com/ynulihao/LLMRouterBench
- RouteLLM; paper: https://arxiv.org/abs/2406.18665; repo: https://github.com/lm-sys/routellm
- LLMRouter; repo: https://github.com/ulab-uiuc/LLMRouter
- RouterBench; paper: https://arxiv.org/abs/2403.12031; repo: https://github.com/withmartian/routerbench
- WebRouter; paper: https://arxiv.org/abs/2510.11221
- FineRouter; paper: https://arxiv.org/abs/2603.19415
- BEST-Route; paper: https://openreview.net/forum?id=tFBIbCVXkG; repo: https://github.com/microsoft/best-route-llm
- GraphRouter; paper: https://openreview.net/forum?id=eU39PDsZtT; repo: https://github.com/ulab-uiuc/LLMRouter
- Avengers-Pro; paper: https://arxiv.org/abs/2508.12631; repo: https://github.com/ZhangYiqun018/AvengersPro; local source: `data/raw/external/LLMRouterBench/baselines/AvengersPro`
- Universal Model Routing; paper: https://openreview.net/pdf?id=ka82fvJ5f1
- kNN routing; paper: https://arxiv.org/abs/2505.12601; repo: https://github.com/ulab-uiuc/LLMRouter
- Causal LLM Routing; paper: https://openreview.net/forum?id=iZC5xoQQkX

## Leakage Controls

- Train/validation/test splits are assigned by `query_id`; all model rows for a query stay in the same split.
- Best-single, dataset/topic tables, embedding clusters, kNN neighbors, and RouteCode codebooks are fit on train only.
- Query oracle uses test utility only as an upper bound.
- The leaky dataset-label diagnostic is written separately to `table_leakage_gap.csv` and is not a deployable baseline.

## Next Steps

1. Improve deployable residual predictors before implementing adaptive refinement.
2. Add exact upstream-command external baselines where dependencies are pinned before making method-ranking claims.
3. Expand the broad20 split-sensitivity, ablation, and transfer sweeps beyond the bounded local diagnostics if runtime permits.
4. Run stricter held-out model-pool transfer checks on larger benchmark-scale model pools.
5. Keep final claims scoped to verified evidence until broader robustness checks pass.

## Ablation And Robustness

Command:

```bash
python experiments/08_ablation_summary.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_ablation_summary.csv`: bounded seed, K/lambda, rate-penalty, and training-fraction ablation rows.
- `fig_sensitivity_k_lambda.pdf`: recovered-gap heatmaps over K and lambda.
- `fig_seed_stability.pdf`: seed-stability bars with standard deviations.
- `phase_f_g_ablation_memo.md`: robustness checkpoint memo.

| ablation | method | mean_recovered_gap | min_recovered_gap | max_recovered_gap |
| --- | --- | --- | --- | --- |
| k_lambda | regret_routecode_utility_oracle | 0.9825 | 0.8496 | 1.0000 |
| k_lambda | flat_routecode_utility_oracle | 0.8603 | 0.3383 | 1.0000 |
| k_lambda | semantic_embedding_kmeans | 0.2990 | 0.2331 | 0.3609 |
| k_lambda | d2_embedding_centroid | 0.2921 | 0.2181 | 0.3460 |
| k_lambda | best_single | 0.0000 | 0.0000 | 0.0000 |
| rate_penalty | d2_embedding_centroid | 0.3459 | 0.3459 | 0.3459 |
| seed_stability | d2_embedding_centroid | 0.3320 | 0.3121 | 0.3459 |
| seed_stability | kNN | 0.2723 | 0.2324 | 0.3008 |
| seed_stability | svm_embedding_router | 0.0052 | -0.0211 | 0.0226 |
| seed_stability | best_single | 0.0000 | 0.0000 | 0.0000 |
| seed_stability | logistic_embedding_router | -0.0064 | -0.0493 | 0.0301 |
| train_fraction | logistic_embedding_router | 0.0050 | -0.0301 | 0.0301 |
| train_fraction | svm_embedding_router | -0.0075 | -0.0752 | 0.0301 |

## Sensitivity Suite

Command:

```bash
python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_sensitivity_summary.csv`: bounded embedding, clustering, label-noise, cost, price-ratio, model-pool subset/composition, automatic dominated/complementary pool, domain-granularity, query-length, and bootstrap sensitivity rows.
- `fig_sensitivity_summary.pdf`: method-by-sensitivity recovered-gap heatmap.
- `phase_g_sensitivity_memo.md`: Phase G checkpoint memo.

| sensitivity | method | mean_recovered_gap |
| --- | --- | --- |
| bootstrap_sampling | d2_embedding_centroid | 0.3459 |
| clustering_algorithm | d2_embedding_centroid | 0.3459 |
| clustering_algorithm | semantic_embedding_cluster | 0.3308 |
| clustering_algorithm | kNN | 0.3008 |
| clustering_algorithm | best_single | 0.0000 |
| cost_misestimation | d2_embedding_centroid | 0.4305 |
| cost_misestimation | kNN | 0.4236 |
| cost_misestimation | best_single | 0.0161 |
| domain_granularity | d2_embedding_centroid | 0.1668 |
| domain_granularity | kNN | 0.0277 |
| domain_granularity | best_single | 0.0000 |
| embedding_backbone | d2_embedding_centroid | 0.3233 |
| embedding_backbone | kNN | 0.2757 |
| embedding_backbone | best_single | 0.0000 |
| label_noise | logistic_embedding_router | -0.0627 |
| model_pool | d2_embedding_centroid | 0.3682 |
| model_pool | kNN | 0.3678 |
| model_pool | best_single | 0.0000 |
| model_pool_auto | d2_embedding_centroid | 0.2839 |
| model_pool_auto | kNN | 0.2167 |
| model_pool_auto | best_single | 0.0000 |
| model_pool_composition | d2_embedding_centroid | 0.2246 |
| model_pool_composition | kNN | 0.1730 |
| model_pool_composition | best_single | 0.0000 |
| price_ratio | kNN | 0.3959 |
| price_ratio | d2_embedding_centroid | 0.3699 |
| price_ratio | best_single | 0.0000 |
| query_length_bucket | d2_embedding_centroid | 0.3163 |

## RouteCode Predictor Diagnostics

Command:

```bash
python experiments/05_predictor_diagnostics.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_predictor_comparison.csv`: oracle-code label accuracy, calibration, and routing utility by predictor.
- `table_utility_weighted_confusion.csv`: label-confusion cells weighted by utility regret.
- `table_calibration_curve.csv`: confidence-bin calibration data.
- `fig_utility_weighted_confusion.pdf`: regret-weighted confusion heatmap for the best deployable predictor.
- `fig_calibration_curve.pdf`: route-label predictor calibration curves.

| predictor | label_accuracy | ece | mean_utility | oracle_code_regret | recovered_gap_vs_query_oracle |
| --- | --- | --- | --- | --- | --- |
| utility_oracle_labels | 1.0000 | 0.0000 | 0.8897 | 0.0000 | 0.9699 |
| embedding_centroid_assignment | 0.1362 | 0.0675 | 0.6586 | 0.2310 | -0.0376 |
| mlp_label_predictor | 0.1466 | 0.5778 | 0.6345 | 0.2552 | -0.1429 |
| knn_label_predictor | 0.2052 | 0.0839 | 0.6207 | 0.2690 | -0.2030 |
| logistic_label_predictor | 0.1517 | 0.5454 | 0.6138 | 0.2759 | -0.2331 |

## Predictability-Constrained RouteCode

Command:

```bash
python experiments/06_predictability_constrained.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_predictability_constrained.csv`: alpha sweep with D2 joint-label, embedding-centroid, and logistic-label rows plus comparison baselines.
- `fig_predictability_constrained_tradeoff.pdf`: D2 utility and label-predictability tradeoff by alpha.
- `code_cards_predictability_constrained.md` and `code_cards_predictability_constrained.json`: code cards for the selected D2 alpha.
- `fig_code_label_heatmap_predictability_constrained.pdf`: selected D2 label utility profiles.
- `phase_d_method_memo.md`: D2 checkpoint memo and recommended interpretation.

| method | alpha | mean_utility | recovered_gap_vs_oracle | label_accuracy |
| --- | --- | --- | --- | --- |
| d2_embedding_centroid | 0.0000 | 0.6948 | 0.1203 | 0.1500 |
| d2_embedding_centroid | 0.0500 | 0.6086 | -0.2556 | 0.1707 |
| d2_embedding_centroid | 0.1000 | 0.6500 | -0.0752 | 0.2862 |
| d2_embedding_centroid | 0.3000 | 0.7379 | 0.3083 | 0.5948 |
| d2_embedding_centroid | 1.0000 | 0.7431 | 0.3308 | 0.9086 |
| d2_embedding_centroid | 3.0000 | 0.7466 | 0.3459 | 0.9810 |
| d2_embedding_centroid | 10.0000 | 0.7414 | 0.3233 | 0.9897 |
| d2_logistic_label_predictor | 0.0000 | 0.6914 | 0.1053 | 0.1466 |
| d2_logistic_label_predictor | 0.0500 | 0.6328 | -0.1504 | 0.1638 |
| d2_logistic_label_predictor | 0.1000 | 0.6345 | -0.1429 | 0.2207 |
| d2_logistic_label_predictor | 0.3000 | 0.7241 | 0.2481 | 0.4966 |
| d2_logistic_label_predictor | 1.0000 | 0.7328 | 0.2857 | 0.8259 |
| d2_logistic_label_predictor | 3.0000 | 0.7448 | 0.3383 | 0.8724 |
| d2_logistic_label_predictor | 10.0000 | 0.7448 | 0.3383 | 0.8672 |
| dataset_label_lookup |  | 0.7534 | 0.3759 |  |
| flat_routecode_logistic_label_predictor |  | 0.6138 | -0.2331 |  |
| kNN |  | 0.7362 | 0.3008 |  |
| semantic_embedding_kmeans |  | 0.7362 | 0.3008 |  |

## New-Model Calibration

Command:

```bash
python experiments/07_new_model_calibration.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_new_model_integration.csv`: held-out/new-model calibration sweep.
- `fig_transfer_calibration_curve.pdf`: utility vs new-model calibration evaluations.
- `phase_e5_new_model_calibration_memo.md`: D4/E5 checkpoint memo.

| method | new_model_id | examples_per_label | calibration_query_count | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| direct_retraining_budgeted_gradient_boosting | DeepSeek-R1-Distill-Qwen-7B | 1 | 16 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | DeepSeek-R1-Distill-Qwen-7B | 2 | 32 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | DeepSeek-R1-Distill-Qwen-7B | 4 | 64 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | DeepSeek-R1-Distill-Qwen-7B | 8 | 127 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | DeepSeek-R1-Distill-Qwen-7B | 16 | 238 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | DeepSeek-R1-Distill-Qwen-7B | 32 | 439 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_gradient_boosting | DeepSeek-R1-Distill-Qwen-7B | 64 | 717 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_knn | DeepSeek-R1-Distill-Qwen-7B | 1 | 16 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | DeepSeek-R1-Distill-Qwen-7B | 2 | 32 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | DeepSeek-R1-Distill-Qwen-7B | 4 | 64 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | DeepSeek-R1-Distill-Qwen-7B | 8 | 127 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | DeepSeek-R1-Distill-Qwen-7B | 16 | 238 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | DeepSeek-R1-Distill-Qwen-7B | 32 | 439 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | DeepSeek-R1-Distill-Qwen-7B | 64 | 717 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_logistic | DeepSeek-R1-Distill-Qwen-7B | 1 | 16 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | DeepSeek-R1-Distill-Qwen-7B | 2 | 32 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | DeepSeek-R1-Distill-Qwen-7B | 4 | 64 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | DeepSeek-R1-Distill-Qwen-7B | 8 | 127 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | DeepSeek-R1-Distill-Qwen-7B | 16 | 238 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | DeepSeek-R1-Distill-Qwen-7B | 32 | 439 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | DeepSeek-R1-Distill-Qwen-7B | 64 | 717 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_mlp | DeepSeek-R1-Distill-Qwen-7B | 1 | 16 | 0.6828 | 0.0677 |
| direct_retraining_budgeted_mlp | DeepSeek-R1-Distill-Qwen-7B | 2 | 32 | 0.6828 | 0.0677 |
| direct_retraining_budgeted_mlp | DeepSeek-R1-Distill-Qwen-7B | 4 | 64 | 0.6828 | 0.0677 |
| direct_retraining_budgeted_mlp | DeepSeek-R1-Distill-Qwen-7B | 8 | 127 | 0.6828 | 0.0677 |
| direct_retraining_budgeted_mlp | DeepSeek-R1-Distill-Qwen-7B | 16 | 238 | 0.6828 | 0.0677 |
| direct_retraining_budgeted_mlp | DeepSeek-R1-Distill-Qwen-7B | 32 | 439 | 0.6638 | -0.0150 |
| direct_retraining_budgeted_mlp | DeepSeek-R1-Distill-Qwen-7B | 64 | 717 | 0.6569 | -0.0451 |
| direct_retraining_budgeted_svm | DeepSeek-R1-Distill-Qwen-7B | 1 | 16 | 0.6741 | 0.0301 |
| direct_retraining_budgeted_svm | DeepSeek-R1-Distill-Qwen-7B | 2 | 32 | 0.6741 | 0.0301 |
| direct_retraining_budgeted_svm | DeepSeek-R1-Distill-Qwen-7B | 4 | 64 | 0.6741 | 0.0301 |
| direct_retraining_budgeted_svm | DeepSeek-R1-Distill-Qwen-7B | 8 | 127 | 0.6741 | 0.0301 |
| direct_retraining_budgeted_svm | DeepSeek-R1-Distill-Qwen-7B | 16 | 238 | 0.6741 | 0.0301 |
| direct_retraining_budgeted_svm | DeepSeek-R1-Distill-Qwen-7B | 32 | 439 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_svm | DeepSeek-R1-Distill-Qwen-7B | 64 | 717 | 0.6724 | 0.0226 |
| routecode_label_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 16 | 0.5931 | -0.3233 |
| routecode_label_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 32 | 0.7017 | 0.1504 |
| routecode_label_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 64 | 0.7345 | 0.2932 |
| routecode_label_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 127 | 0.7069 | 0.1729 |
| routecode_label_calibration | DeepSeek-R1-Distill-Qwen-7B | 16 | 238 | 0.7328 | 0.2857 |
| routecode_label_calibration | DeepSeek-R1-Distill-Qwen-7B | 32 | 439 | 0.7414 | 0.3233 |
| routecode_label_calibration | DeepSeek-R1-Distill-Qwen-7B | 64 | 717 | 0.7414 | 0.3233 |
| direct_retraining_budgeted_gradient_boosting | Intern-S1-mini | 1 | 16 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_gradient_boosting | Intern-S1-mini | 2 | 32 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_gradient_boosting | Intern-S1-mini | 4 | 64 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_gradient_boosting | Intern-S1-mini | 8 | 127 | 0.6741 | 0.0301 |
| direct_retraining_budgeted_gradient_boosting | Intern-S1-mini | 16 | 238 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_gradient_boosting | Intern-S1-mini | 32 | 438 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_gradient_boosting | Intern-S1-mini | 64 | 715 | 0.6741 | 0.0301 |
| direct_retraining_budgeted_knn | Intern-S1-mini | 1 | 16 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_knn | Intern-S1-mini | 2 | 32 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_knn | Intern-S1-mini | 4 | 64 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_knn | Intern-S1-mini | 8 | 127 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_knn | Intern-S1-mini | 16 | 238 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_knn | Intern-S1-mini | 32 | 438 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_knn | Intern-S1-mini | 64 | 715 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_logistic | Intern-S1-mini | 1 | 16 | 0.6810 | 0.0602 |
| direct_retraining_budgeted_logistic | Intern-S1-mini | 2 | 32 | 0.6810 | 0.0602 |
| direct_retraining_budgeted_logistic | Intern-S1-mini | 4 | 64 | 0.6793 | 0.0526 |
| direct_retraining_budgeted_logistic | Intern-S1-mini | 8 | 127 | 0.6810 | 0.0602 |
| direct_retraining_budgeted_logistic | Intern-S1-mini | 16 | 238 | 0.6810 | 0.0602 |
| direct_retraining_budgeted_logistic | Intern-S1-mini | 32 | 438 | 0.6793 | 0.0526 |
| direct_retraining_budgeted_logistic | Intern-S1-mini | 64 | 715 | 0.6810 | 0.0602 |
| direct_retraining_budgeted_mlp | Intern-S1-mini | 1 | 16 | 0.6810 | 0.0602 |
| direct_retraining_budgeted_mlp | Intern-S1-mini | 2 | 32 | 0.6810 | 0.0602 |
| direct_retraining_budgeted_mlp | Intern-S1-mini | 4 | 64 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_mlp | Intern-S1-mini | 8 | 127 | 0.6776 | 0.0451 |
| direct_retraining_budgeted_mlp | Intern-S1-mini | 16 | 238 | 0.6810 | 0.0602 |
| direct_retraining_budgeted_mlp | Intern-S1-mini | 32 | 438 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_mlp | Intern-S1-mini | 64 | 715 | 0.6741 | 0.0301 |
| direct_retraining_budgeted_svm | Intern-S1-mini | 1 | 16 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Intern-S1-mini | 2 | 32 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Intern-S1-mini | 4 | 64 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_svm | Intern-S1-mini | 8 | 127 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Intern-S1-mini | 16 | 238 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Intern-S1-mini | 32 | 438 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_svm | Intern-S1-mini | 64 | 715 | 0.6690 | 0.0075 |
| routecode_label_calibration | Intern-S1-mini | 1 | 16 | 0.7224 | 0.2406 |
| routecode_label_calibration | Intern-S1-mini | 2 | 32 | 0.7155 | 0.2105 |
| routecode_label_calibration | Intern-S1-mini | 4 | 64 | 0.7328 | 0.2857 |
| routecode_label_calibration | Intern-S1-mini | 8 | 127 | 0.7241 | 0.2481 |
| routecode_label_calibration | Intern-S1-mini | 16 | 238 | 0.7362 | 0.3008 |
| routecode_label_calibration | Intern-S1-mini | 32 | 438 | 0.7362 | 0.3008 |
| routecode_label_calibration | Intern-S1-mini | 64 | 715 | 0.7362 | 0.3008 |
| direct_retraining_budgeted_gradient_boosting | Llama-3.1-8B-Instruct | 1 | 16 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | Llama-3.1-8B-Instruct | 2 | 32 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_gradient_boosting | Llama-3.1-8B-Instruct | 4 | 63 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_gradient_boosting | Llama-3.1-8B-Instruct | 8 | 123 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_gradient_boosting | Llama-3.1-8B-Instruct | 16 | 239 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | Llama-3.1-8B-Instruct | 32 | 433 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_gradient_boosting | Llama-3.1-8B-Instruct | 64 | 698 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Llama-3.1-8B-Instruct | 1 | 16 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Llama-3.1-8B-Instruct | 2 | 32 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Llama-3.1-8B-Instruct | 4 | 63 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Llama-3.1-8B-Instruct | 8 | 123 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Llama-3.1-8B-Instruct | 16 | 239 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Llama-3.1-8B-Instruct | 32 | 433 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Llama-3.1-8B-Instruct | 64 | 698 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_logistic | Llama-3.1-8B-Instruct | 1 | 16 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | Llama-3.1-8B-Instruct | 2 | 32 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | Llama-3.1-8B-Instruct | 4 | 63 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | Llama-3.1-8B-Instruct | 8 | 123 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | Llama-3.1-8B-Instruct | 16 | 239 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_logistic | Llama-3.1-8B-Instruct | 32 | 433 | 0.6793 | 0.0526 |
| direct_retraining_budgeted_logistic | Llama-3.1-8B-Instruct | 64 | 698 | 0.6793 | 0.0526 |
| direct_retraining_budgeted_mlp | Llama-3.1-8B-Instruct | 1 | 16 | 0.6517 | -0.0677 |
| direct_retraining_budgeted_mlp | Llama-3.1-8B-Instruct | 2 | 32 | 0.6845 | 0.0752 |
| direct_retraining_budgeted_mlp | Llama-3.1-8B-Instruct | 4 | 63 | 0.6845 | 0.0752 |
| direct_retraining_budgeted_mlp | Llama-3.1-8B-Instruct | 8 | 123 | 0.6552 | -0.0526 |
| direct_retraining_budgeted_mlp | Llama-3.1-8B-Instruct | 16 | 239 | 0.6534 | -0.0602 |
| direct_retraining_budgeted_mlp | Llama-3.1-8B-Instruct | 32 | 433 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_mlp | Llama-3.1-8B-Instruct | 64 | 698 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_svm | Llama-3.1-8B-Instruct | 1 | 16 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Llama-3.1-8B-Instruct | 2 | 32 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Llama-3.1-8B-Instruct | 4 | 63 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Llama-3.1-8B-Instruct | 8 | 123 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Llama-3.1-8B-Instruct | 16 | 239 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_svm | Llama-3.1-8B-Instruct | 32 | 433 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_svm | Llama-3.1-8B-Instruct | 64 | 698 | 0.6690 | 0.0075 |
| routecode_label_calibration | Llama-3.1-8B-Instruct | 1 | 16 | 0.6707 | 0.0150 |
| routecode_label_calibration | Llama-3.1-8B-Instruct | 2 | 32 | 0.6190 | -0.2105 |
| routecode_label_calibration | Llama-3.1-8B-Instruct | 4 | 63 | 0.7069 | 0.1729 |
| routecode_label_calibration | Llama-3.1-8B-Instruct | 8 | 123 | 0.6810 | 0.0602 |
| routecode_label_calibration | Llama-3.1-8B-Instruct | 16 | 239 | 0.7241 | 0.2481 |
| routecode_label_calibration | Llama-3.1-8B-Instruct | 32 | 433 | 0.7241 | 0.2481 |
| routecode_label_calibration | Llama-3.1-8B-Instruct | 64 | 698 | 0.7241 | 0.2481 |
| direct_retraining_budgeted_gradient_boosting | MiniCPM4.1-8B | 1 | 16 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | MiniCPM4.1-8B | 2 | 32 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | MiniCPM4.1-8B | 4 | 64 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_gradient_boosting | MiniCPM4.1-8B | 8 | 127 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | MiniCPM4.1-8B | 16 | 237 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | MiniCPM4.1-8B | 32 | 407 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_gradient_boosting | MiniCPM4.1-8B | 64 | 638 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_knn | MiniCPM4.1-8B | 1 | 16 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | MiniCPM4.1-8B | 2 | 32 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | MiniCPM4.1-8B | 4 | 64 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | MiniCPM4.1-8B | 8 | 127 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | MiniCPM4.1-8B | 16 | 237 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | MiniCPM4.1-8B | 32 | 407 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | MiniCPM4.1-8B | 64 | 638 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_logistic | MiniCPM4.1-8B | 1 | 16 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_logistic | MiniCPM4.1-8B | 2 | 32 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_logistic | MiniCPM4.1-8B | 4 | 64 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_logistic | MiniCPM4.1-8B | 8 | 127 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_logistic | MiniCPM4.1-8B | 16 | 237 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_logistic | MiniCPM4.1-8B | 32 | 407 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_logistic | MiniCPM4.1-8B | 64 | 638 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_mlp | MiniCPM4.1-8B | 1 | 16 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_mlp | MiniCPM4.1-8B | 2 | 32 | 0.6759 | 0.0376 |
| direct_retraining_budgeted_mlp | MiniCPM4.1-8B | 4 | 64 | 0.6621 | -0.0226 |
| direct_retraining_budgeted_mlp | MiniCPM4.1-8B | 8 | 127 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_mlp | MiniCPM4.1-8B | 16 | 237 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_mlp | MiniCPM4.1-8B | 32 | 407 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_mlp | MiniCPM4.1-8B | 64 | 638 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_svm | MiniCPM4.1-8B | 1 | 16 | 0.6586 | -0.0376 |
| direct_retraining_budgeted_svm | MiniCPM4.1-8B | 2 | 32 | 0.6586 | -0.0376 |
| direct_retraining_budgeted_svm | MiniCPM4.1-8B | 4 | 64 | 0.6603 | -0.0301 |
| direct_retraining_budgeted_svm | MiniCPM4.1-8B | 8 | 127 | 0.6586 | -0.0376 |
| direct_retraining_budgeted_svm | MiniCPM4.1-8B | 16 | 237 | 0.6586 | -0.0376 |
| direct_retraining_budgeted_svm | MiniCPM4.1-8B | 32 | 407 | 0.6586 | -0.0376 |
| direct_retraining_budgeted_svm | MiniCPM4.1-8B | 64 | 638 | 0.6603 | -0.0301 |
| routecode_label_calibration | MiniCPM4.1-8B | 1 | 16 | 0.7052 | 0.1654 |
| routecode_label_calibration | MiniCPM4.1-8B | 2 | 32 | 0.7172 | 0.2180 |
| routecode_label_calibration | MiniCPM4.1-8B | 4 | 64 | 0.7379 | 0.3083 |
| routecode_label_calibration | MiniCPM4.1-8B | 8 | 127 | 0.7379 | 0.3083 |
| routecode_label_calibration | MiniCPM4.1-8B | 16 | 237 | 0.7414 | 0.3233 |
| routecode_label_calibration | MiniCPM4.1-8B | 32 | 407 | 0.7414 | 0.3233 |
| routecode_label_calibration | MiniCPM4.1-8B | 64 | 638 | 0.7276 | 0.2632 |
| direct_retraining_budgeted_gradient_boosting | Qwen2.5-Coder-7B-Instruct | 1 | 16 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_gradient_boosting | Qwen2.5-Coder-7B-Instruct | 2 | 32 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_gradient_boosting | Qwen2.5-Coder-7B-Instruct | 4 | 64 | 0.6707 | 0.0150 |
| direct_retraining_budgeted_gradient_boosting | Qwen2.5-Coder-7B-Instruct | 8 | 124 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_gradient_boosting | Qwen2.5-Coder-7B-Instruct | 16 | 214 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_gradient_boosting | Qwen2.5-Coder-7B-Instruct | 32 | 376 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_gradient_boosting | Qwen2.5-Coder-7B-Instruct | 64 | 607 | 0.6724 | 0.0226 |
| direct_retraining_budgeted_knn | Qwen2.5-Coder-7B-Instruct | 1 | 16 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Qwen2.5-Coder-7B-Instruct | 2 | 32 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Qwen2.5-Coder-7B-Instruct | 4 | 64 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Qwen2.5-Coder-7B-Instruct | 8 | 124 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Qwen2.5-Coder-7B-Instruct | 16 | 214 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Qwen2.5-Coder-7B-Instruct | 32 | 376 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_knn | Qwen2.5-Coder-7B-Instruct | 64 | 607 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_logistic | Qwen2.5-Coder-7B-Instruct | 1 | 16 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_logistic | Qwen2.5-Coder-7B-Instruct | 2 | 32 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_logistic | Qwen2.5-Coder-7B-Instruct | 4 | 64 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_logistic | Qwen2.5-Coder-7B-Instruct | 8 | 124 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_logistic | Qwen2.5-Coder-7B-Instruct | 16 | 214 | 0.6638 | -0.0150 |
| direct_retraining_budgeted_logistic | Qwen2.5-Coder-7B-Instruct | 32 | 376 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_logistic | Qwen2.5-Coder-7B-Instruct | 64 | 607 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_mlp | Qwen2.5-Coder-7B-Instruct | 1 | 16 | 0.6776 | 0.0451 |
| direct_retraining_budgeted_mlp | Qwen2.5-Coder-7B-Instruct | 2 | 32 | 0.6621 | -0.0226 |
| direct_retraining_budgeted_mlp | Qwen2.5-Coder-7B-Instruct | 4 | 64 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_mlp | Qwen2.5-Coder-7B-Instruct | 8 | 124 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_mlp | Qwen2.5-Coder-7B-Instruct | 16 | 214 | 0.6552 | -0.0526 |
| direct_retraining_budgeted_mlp | Qwen2.5-Coder-7B-Instruct | 32 | 376 | 0.6603 | -0.0301 |
| direct_retraining_budgeted_mlp | Qwen2.5-Coder-7B-Instruct | 64 | 607 | 0.6603 | -0.0301 |
| direct_retraining_budgeted_svm | Qwen2.5-Coder-7B-Instruct | 1 | 16 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_svm | Qwen2.5-Coder-7B-Instruct | 2 | 32 | 0.6690 | 0.0075 |
| direct_retraining_budgeted_svm | Qwen2.5-Coder-7B-Instruct | 4 | 64 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_svm | Qwen2.5-Coder-7B-Instruct | 8 | 124 | 0.6672 | 0.0000 |
| direct_retraining_budgeted_svm | Qwen2.5-Coder-7B-Instruct | 16 | 214 | 0.6655 | -0.0075 |
| direct_retraining_budgeted_svm | Qwen2.5-Coder-7B-Instruct | 32 | 376 | 0.6621 | -0.0226 |
| direct_retraining_budgeted_svm | Qwen2.5-Coder-7B-Instruct | 64 | 607 | 0.6621 | -0.0226 |
| routecode_label_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 16 | 0.6983 | 0.1353 |
| routecode_label_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 32 | 0.7259 | 0.2556 |
| routecode_label_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 64 | 0.7328 | 0.2857 |
| routecode_label_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 124 | 0.7017 | 0.1504 |
| routecode_label_calibration | Qwen2.5-Coder-7B-Instruct | 16 | 214 | 0.7103 | 0.1880 |
| routecode_label_calibration | Qwen2.5-Coder-7B-Instruct | 32 | 376 | 0.7431 | 0.3308 |
| routecode_label_calibration | Qwen2.5-Coder-7B-Instruct | 64 | 607 | 0.7431 | 0.3308 |
| direct_retraining_budgeted_gradient_boosting | Qwen3-8B | 1 | 16 | 0.5948 | -0.3158 |
| direct_retraining_budgeted_gradient_boosting | Qwen3-8B | 2 | 31 | 0.5948 | -0.3158 |
| direct_retraining_budgeted_gradient_boosting | Qwen3-8B | 4 | 61 | 0.5948 | -0.3158 |
| direct_retraining_budgeted_gradient_boosting | Qwen3-8B | 8 | 121 | 0.5914 | -0.3308 |
| direct_retraining_budgeted_gradient_boosting | Qwen3-8B | 16 | 225 | 0.5966 | -0.3083 |
| direct_retraining_budgeted_gradient_boosting | Qwen3-8B | 32 | 378 | 0.5966 | -0.3083 |
| direct_retraining_budgeted_gradient_boosting | Qwen3-8B | 64 | 573 | 0.5931 | -0.3233 |
| direct_retraining_budgeted_knn | Qwen3-8B | 1 | 16 | 0.5897 | -0.3383 |
| direct_retraining_budgeted_knn | Qwen3-8B | 2 | 31 | 0.5897 | -0.3383 |
| direct_retraining_budgeted_knn | Qwen3-8B | 4 | 61 | 0.5897 | -0.3383 |
| direct_retraining_budgeted_knn | Qwen3-8B | 8 | 121 | 0.5897 | -0.3383 |
| direct_retraining_budgeted_knn | Qwen3-8B | 16 | 225 | 0.5897 | -0.3383 |
| direct_retraining_budgeted_knn | Qwen3-8B | 32 | 378 | 0.5897 | -0.3383 |
| direct_retraining_budgeted_knn | Qwen3-8B | 64 | 573 | 0.5897 | -0.3383 |
| direct_retraining_budgeted_logistic | Qwen3-8B | 1 | 16 | 0.6069 | -0.2632 |
| direct_retraining_budgeted_logistic | Qwen3-8B | 2 | 31 | 0.6069 | -0.2632 |
| direct_retraining_budgeted_logistic | Qwen3-8B | 4 | 61 | 0.6069 | -0.2632 |
| direct_retraining_budgeted_logistic | Qwen3-8B | 8 | 121 | 0.6069 | -0.2632 |
| direct_retraining_budgeted_logistic | Qwen3-8B | 16 | 225 | 0.6069 | -0.2632 |
| direct_retraining_budgeted_logistic | Qwen3-8B | 32 | 378 | 0.6052 | -0.2707 |
| direct_retraining_budgeted_logistic | Qwen3-8B | 64 | 573 | 0.6069 | -0.2632 |
| direct_retraining_budgeted_mlp | Qwen3-8B | 1 | 16 | 0.6172 | -0.2180 |
| direct_retraining_budgeted_mlp | Qwen3-8B | 2 | 31 | 0.6172 | -0.2180 |
| direct_retraining_budgeted_mlp | Qwen3-8B | 4 | 61 | 0.6172 | -0.2180 |
| direct_retraining_budgeted_mlp | Qwen3-8B | 8 | 121 | 0.6172 | -0.2180 |
| direct_retraining_budgeted_mlp | Qwen3-8B | 16 | 225 | 0.6190 | -0.2105 |
| direct_retraining_budgeted_mlp | Qwen3-8B | 32 | 378 | 0.6138 | -0.2331 |
| direct_retraining_budgeted_mlp | Qwen3-8B | 64 | 573 | 0.6103 | -0.2481 |
| direct_retraining_budgeted_svm | Qwen3-8B | 1 | 16 | 0.5966 | -0.3083 |
| direct_retraining_budgeted_svm | Qwen3-8B | 2 | 31 | 0.5966 | -0.3083 |
| direct_retraining_budgeted_svm | Qwen3-8B | 4 | 61 | 0.5966 | -0.3083 |
| direct_retraining_budgeted_svm | Qwen3-8B | 8 | 121 | 0.5966 | -0.3083 |
| direct_retraining_budgeted_svm | Qwen3-8B | 16 | 225 | 0.5966 | -0.3083 |
| direct_retraining_budgeted_svm | Qwen3-8B | 32 | 378 | 0.5966 | -0.3083 |
| direct_retraining_budgeted_svm | Qwen3-8B | 64 | 573 | 0.5966 | -0.3083 |
| routecode_label_calibration | Qwen3-8B | 1 | 16 | 0.7241 | 0.2481 |
| routecode_label_calibration | Qwen3-8B | 2 | 31 | 0.7086 | 0.1805 |
| routecode_label_calibration | Qwen3-8B | 4 | 61 | 0.7379 | 0.3083 |
| routecode_label_calibration | Qwen3-8B | 8 | 121 | 0.7224 | 0.2406 |
| routecode_label_calibration | Qwen3-8B | 16 | 225 | 0.7379 | 0.3083 |
| routecode_label_calibration | Qwen3-8B | 32 | 378 | 0.7379 | 0.3083 |
| routecode_label_calibration | Qwen3-8B | 64 | 573 | 0.7190 | 0.2256 |

## External Baseline Surrogates

Command:

```bash
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_external_baselines.csv`: local external-style baseline surrogate rows with explicit implementation notes.
- `phase_e_external_baseline_memo.md`: Phase E checkpoint memo for these surrogate baselines.

| method | baseline_family | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- |
| query_oracle | reference | 0.8966 | 1.0000 |
| kNN | reference | 0.7362 | 0.3008 |
| routellm_style_mf_utility_router | external_style_surrogate | 0.7052 | 0.1654 |
| routellm_binary_logistic_surrogate_t0.25 | external_style_surrogate | 0.6931 | 0.1128 |
| best_single | reference | 0.6672 | 0.0000 |
| routellm_pair_strong_only | binary_pair_reference | 0.6672 | 0.0000 |
| routellm_binary_logistic_surrogate_t0.5 | external_style_surrogate | 0.6552 | -0.0526 |
| routellm_binary_logistic_surrogate_t0.75 | external_style_surrogate | 0.6241 | -0.1880 |
| routellm_pair_weak_only | binary_pair_reference | 0.6086 | -0.2556 |

## Code-Card Interpretability

Command:

```bash
python experiments/11_code_card_interpretability.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_code_card_interpretability.csv`: label-only versus code-card observability coverage for flat and D2 RouteCode.
- `phase_f_code_card_interpretability_memo.md`: Phase F memo for the code-card interpretability ablation.

| codebook | condition | available_explainability_fields | representative_query_coverage | failure_case_coverage | human_explanation_coverage |
| --- | --- | --- | --- | --- | --- |
| flat_routecode | label_only | 1 | 0.0000 | 0.0000 | 0.0000 |
| flat_routecode | with_code_cards | 9 | 1.0000 | 1.0000 | 1.0000 |
| predictability_constrained_routecode | label_only | 1 | 0.0000 | 0.0000 | 0.0000 |
| predictability_constrained_routecode | with_code_cards | 9 | 1.0000 | 1.0000 | 1.0000 |

## Residual Concentration

Command:

```bash
python experiments/03_residual_concentration.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_residual_concentration.csv`: fraction of residual regret captured by top-regret queries.
- `table_residual_risk.csv`: regret capture and AUC for deployable residual-risk signals.
- `table_residual_queries.csv`: per-query regret, margin, label, confidence, centroid distance, and kNN disagreement.
- `table_residual_by_label.csv`: per-label residual summary.
- `fig_residual_concentration.pdf`: residual concentration curve.
- `fig_risk_coverage.pdf`: regret mass captured by top-risk query fractions.
- `phase_d5_adaptive_refinement_gate_memo.md`: gate memo for whether adaptive refinement is justified.

Residual concentration:

| top_fraction | n_queries | top_regret | total_regret | regret_mass_fraction |
| --- | --- | --- | --- | --- |
| 0.0500 | 29.0000 | 29.0000 | 164.0000 | 0.1768 |
| 0.1000 | 58.0000 | 58.0000 | 164.0000 | 0.3537 |
| 0.2000 | 116.0000 | 116.0000 | 164.0000 | 0.7073 |

Deployable risk coverage:

| score | top_fraction | regret_mass_fraction | positive_regret_recall | auc_regret_positive |
| --- | --- | --- | --- | --- |
| low_route_label_confidence | 0.0500 | 0.0549 | 0.0549 | 0.5547 |
| low_route_label_confidence | 0.1000 | 0.1220 | 0.1220 | 0.5547 |
| centroid_distance_risk | 0.0500 | 0.0122 | 0.0122 | 0.4560 |
| centroid_distance_risk | 0.1000 | 0.0610 | 0.0610 | 0.4560 |
| knn_disagreement | 0.0500 | 0.0305 | 0.0305 | 0.5614 |
| knn_disagreement | 0.1000 | 0.0915 | 0.0915 | 0.5614 |
| knn_disagreement_plus_distance | 0.0500 | 0.0244 | 0.0244 | 0.5385 |
| knn_disagreement_plus_distance | 0.1000 | 0.0976 | 0.0976 | 0.5385 |

## Official External Baseline Artifacts

Command:

```bash
python experiments/12_official_baseline_artifacts.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_official_external_artifacts.csv`: parsed official upstream RouteLLM MF artifact rows from the local LLMRouterBench checkout.
- `phase_e_official_baseline_artifacts_memo.md`: compatibility memo explaining why these artifacts are not RouteCode split-aligned metrics.

Source directory: `data/raw/external/LLMRouterBench/baselines/RouteLLM/results`.

These rows are official upstream artifacts, but they are not RouteCode split-aligned and should not be ranked directly against `table_rate_distortion.csv`.

| method | seed | total | selection_accuracy | routing_accuracy | total_cost | csv_selection_accuracy | csv_total_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RouteLLM-MF | 42 | 3858 | 0.6029 | 0.6881 | 124.9606 | 0.6029 | 124.9606 |
| RouteLLM-MF | 999 | 3858 | 0.6262 | 0.6952 | 126.6179 | 0.6262 | 126.6179 |
| RouteLLM-MF | 2024 | 3858 | 0.6260 | 0.6929 | 126.1494 | 0.6260 | 126.1494 |
| RouteLLM-MF | 2025 | 3858 | 0.6179 | 0.6980 | 124.6618 | 0.6179 | 124.6618 |
| RouteLLM-MF | 3407 | 3858 | 0.6094 | 0.6827 | 126.0154 | 0.6094 | 126.0154 |

## Transformer Backbone Readiness

Command:

```bash
python experiments/13_transformer_backbone_readiness.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_transformer_backbone_readiness.csv`: cache-only transformer text-backbone readiness scan.
- `phase_f_g_transformer_backbone_readiness_memo.md`: memo explaining why transformer embedding/direct-router baselines are or are not runnable locally.

Cache directory: `/home/liush/.cache/huggingface/hub`.

Runnable encoder candidates found: `0`. This scan performs no downloads and does not load model weights.

| model_id | cache_status | runnable_as_encoder_baseline | reason | architecture | size_gb |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-4B | cached | False | causal_lm_not_lightweight_encoder | Qwen3ForCausalLM | 7.5073 |
| Tongyi-MAI/Z-Image | cached | False | missing_transformer_config |  | 19.1363 |
| ai-toolkit/flux2_vae | cached | False | missing_transformer_config |  | 0.3131 |
| black-forest-labs/FLUX.1-Kontext-dev | cached | False | missing_transformer_config |  | 8.1181 |
| black-forest-labs/FLUX.2-klein-4B | cached | False | missing_transformer_config |  | 0.0000 |
| black-forest-labs/FLUX.2-klein-base-4B | cached | False | missing_transformer_config |  | 22.1014 |
| answerdotai/ModernBERT-base | missing_local_cache | False | missing_local_cache |  | 0.0000 |
| microsoft/deberta-v3-base | missing_local_cache | False | missing_local_cache |  | 0.0000 |

## RouteLLM Pairwise Alignment Substrate

Command:

```bash
python experiments/14_routellm_pairwise_alignment.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `routellm_pairwise/pairwise_train.json`: RouteCode train-split strong/weak pairwise records.
- `routellm_pairwise/pairwise_test.json`: RouteCode test-split strong/weak pairwise records.
- `routellm_pairwise/metadata.json`: split-alignment and compatibility metadata.
- `table_routellm_pairwise_alignment.csv`: winner distribution and split-alignment summary.
- `phase_e_routellm_pairwise_alignment_memo.md`: Phase E memo explaining remaining official RouteLLM work.

Artifact directory: `results/llmrouterbench_pilot/routellm_pairwise`.

This is not an official RouteLLM MF/BERT result; it is a split-aligned substrate for a future official run.

| split | record_count | decisive_count | tie_count | model_a_win_count | model_b_win_count | model_a_win_rate | model_b_win_rate | tie_rate | mean_utility_margin_model_a_minus_b | strong_model | weak_model | model_a | model_b | split_aligned_with_routecode | official_routellm_result | routecode_metric_compatible | implementation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1738 | 627 | 1111 | 393 | 234 | 0.2261 | 0.1346 | 0.6392 | 0.0915 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | False | False | Pairwise data substrate for later official RouteLLM evaluation; not an official RouteLLM MF/BERT result. |
| test | 580 | 206 | 374 | 120 | 86 | 0.2069 | 0.1483 | 0.6448 | 0.0586 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | False | False | Pairwise data substrate for later official RouteLLM evaluation; not an official RouteLLM MF/BERT result. |
| overall | 2318 | 833 | 1485 | 513 | 320 | 0.2213 | 0.1381 | 0.6406 | 0.0833 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | False | False | Pairwise data substrate for later official RouteLLM evaluation; not an official RouteLLM MF/BERT result. |

## RouteLLM MF Trainer Assets

Command:

```bash
python experiments/15_routellm_mf_assets.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `routellm_mf_assets/pairwise_train.json`: quality-winner RouteLLM-MF train records, with utility fields retained.
- `routellm_mf_assets/pairwise_test.json`: quality-winner RouteLLM-MF test records, with ties retained.
- `routellm_mf_assets/prompt_embeddings.npy`: RouteCode deterministic query embeddings aligned to `idx`.
- `routellm_mf_assets/prompt_index.json`: query-id to MF prompt-index mapping.
- `routellm_mf_assets/mf_train_config.local.json`: local CPU config for the LLMRouterBench RouteLLM MF trainer.
- `table_routellm_mf_assets.csv`: asset compatibility and winner-distribution summary.
- `phase_e_routellm_mf_assets_memo.md`: Phase E memo explaining the remaining train/eval step.

Artifact directory: `results/llmrouterbench_pilot/routellm_mf_assets`.

These files are trainer inputs, not a trained RouteLLM MF result.

| split | record_count | decisive_count | tie_count | model_a_quality_win_count | model_b_quality_win_count | model_a_quality_win_rate | model_b_quality_win_rate | quality_tie_rate | model_a_utility_win_count | model_b_utility_win_count | utility_tie_count | mean_utility_margin_model_a_minus_b | strong_model | weak_model | split_aligned_with_routecode | official_trainer_compatible | official_routellm_result | routecode_metric_compatible | implementation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 627 | 627 | 0 | 393 | 234 | 0.6268 | 0.3732 | 0.0000 | 393 | 234 | 0 | 0.2536 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | True | False | False | Official-trainer-compatible RouteLLM MF assets with local RouteCode embeddings; not a trained RouteLLM MF result. |
| test | 580 | 206 | 374 | 120 | 86 | 0.2069 | 0.1483 | 0.6448 | 120 | 86 | 374 | 0.0586 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | True | False | False | Official-trainer-compatible RouteLLM MF assets with local RouteCode embeddings; not a trained RouteLLM MF result. |
| overall | 1207 | 833 | 374 | 513 | 320 | 0.4250 | 0.2651 | 0.3099 | 513 | 320 | 374 | 0.1599 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | True | True | False | False | Official-trainer-compatible RouteLLM MF assets with local RouteCode embeddings; not a trained RouteLLM MF result. |

## RouteLLM MF Split-Aligned Evaluation

Command:

```bash
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `routellm_mf_split_aligned/mf_model.pt`: checkpoint trained with local LLMRouterBench RouteLLM MF source.
- `routellm_mf_split_aligned/raw_metrics.json`: threshold-level quality metrics and win rates.
- `table_routellm_mf_split_aligned.csv`: RouteCode utility metrics plus RouteLLM-style quality metrics.
- `phase_e_routellm_mf_split_aligned_memo.md`: Phase E memo and caveats.

Artifact directory: `results/llmrouterbench_pilot/routellm_mf_split_aligned`.

This uses official MF training code with local RouteCode embeddings; it is not the upstream published RouteLLM checkpoint.

| mean_utility | oracle_regret | mean_quality | normalized_cost | method | K | utility_ci_low | utility_ci_high | recovered_gap_vs_learned | recovered_gap_vs_oracle | selected_model_entropy | rate_log2K | empirical_H_Z | threshold | strong_model | weak_model | selection_accuracy | routing_accuracy_decisive | decisive_count | tie_count | strong_selection_rate | weak_selection_rate | mean_strong_win_rate | train_loss | validation_accuracy | official_training_code_used | official_upstream_checkpoint | split_aligned_with_routecode | routecode_metric_compatible | baseline_family | implementation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6672 | 0.2293 | 0.6672 | 0.1614 | routellm_mf_split_aligned_t0.25 | 2 | 0.6259 | 0.7044 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2500 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | 0.6672 | 0.5825 | 206 | 374 | 1.0000 | 0.0000 | 0.6062 | None | 0.7476 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |
| 0.7259 | 0.1707 | 0.7259 | 0.1164 | routellm_mf_split_aligned_t0.5 | 2 | 0.6897 | 0.7586 | 0.8500 | 0.2556 | 0.7857 | 1.0000 | 0.7857 | 0.5000 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | 0.7259 | 0.7476 | 206 | 374 | 0.7655 | 0.2345 | 0.6062 | None | 0.7476 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |
| 0.6121 | 0.2845 | 0.6121 | 0.0070 | routellm_mf_split_aligned_t0.75 | 2 | 0.5715 | 0.6448 | -0.8000 | -0.2406 | 0.1732 | 1.0000 | 0.1732 | 0.7500 | Qwen3-8B | Qwen2.5-Coder-7B-Instruct | 0.6121 | 0.4272 | 206 | 374 | 0.0259 | 0.9741 | 0.6062 | None | 0.7476 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |

## Avengers-Pro Split-Aligned Evaluation

Command:

```bash
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_pilot.yaml
```

Outputs:

- `table_avengerspro_split_aligned.csv`: RouteCode utility rows from a local implementation of the Avengers-Pro cluster-routing contract.
- `avengerspro_split_aligned/train.jsonl`, `avengerspro_split_aligned/test.jsonl`, and `avengerspro_split_aligned/baseline_scores.json`: split-aligned Avengers-Pro-format assets.
- `phase_e_avengerspro_split_aligned_memo.md`: caveats and adapter notes.

| method | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- |
| avengerspro_simple_cluster_k16 | 0.7397 | 0.3158 |
| avengerspro_balance_cluster_k16_w0.7_c0.3 | 0.7241 | 0.2481 |

## Benchmark Coverage Audit

Command:

```bash
python experiments/20_benchmark_coverage.py --config configs/llmrouterbench.yaml
```

Outputs:

- `table_benchmark_file_coverage.csv`: latest raw result file coverage by dataset, split, and model.
- `table_benchmark_dataset_coverage.csv`: dataset-level coverage and configured taxonomy status.
- `table_broad_coverage_candidates.csv`: complete dataset/model rectangle candidates.
- `phase_g_benchmark_coverage_memo.md`: benchmark coverage checkpoint memo.

Candidate rectangles:

| model_count | dataset_count | complete_query_count | complete_row_count | models | datasets | dataset_splits |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 18 | 14041 | 84246 | DeepHermes-3-Llama-3-8B-Preview;DeepSeek-R1-0528-Qwen3-8B;DeepSeek-R1-Distill-Qwen-7B;Fin-R1;GLM-Z1-9B-0414;Intern-S1-mini | aime;arcc;bbh;emorynlp;finqa;gpqa;humaneval;kandk;korbench;livecodebench;livemathbench;math500;mathbench;mbpp;medqa;meld;mmlupro;winogrande | aime:hybrid:60;arcc:test:1172;bbh:test:1080;emorynlp:test:697;finqa:test:1147;gpqa:test:198;humaneval:test:164;kandk:test:700;korbench:test:1250;livecodebench:test:1055;livemathbench:test:121;math500:test:500;mathbench:test:150;mbpp:test:974;medqa:test:1273;meld:test:1232;mmlupro:test_1000:1001;winogrande:valid:1267 |
| 10 | 18 | 14041 | 140410 | DeepHermes-3-Llama-3-8B-Preview;DeepSeek-R1-0528-Qwen3-8B;DeepSeek-R1-Distill-Qwen-7B;Fin-R1;GLM-Z1-9B-0414;Intern-S1-mini;Llama-3.1-8B-Instruct;Llama-3.1-8B-UltraMedical;Llama-3.1-Nemotron-Nano-8B-v1;MiMo-7B-RL-0530 | aime;arcc;bbh;emorynlp;finqa;gpqa;humaneval;kandk;korbench;livecodebench;livemathbench;math500;mathbench;mbpp;medqa;meld;mmlupro;winogrande | aime:hybrid:60;arcc:test:1172;bbh:test:1080;emorynlp:test:697;finqa:test:1147;gpqa:test:198;humaneval:test:164;kandk:test:700;korbench:test:1250;livecodebench:test:1055;livemathbench:test:121;math500:test:500;mathbench:test:150;mbpp:test:974;medqa:test:1273;meld:test:1232;mmlupro:test_1000:1001;winogrande:valid:1267 |
| 20 | 18 | 14041 | 280820 | DeepHermes-3-Llama-3-8B-Preview;DeepSeek-R1-0528-Qwen3-8B;DeepSeek-R1-Distill-Qwen-7B;Fin-R1;GLM-Z1-9B-0414;Intern-S1-mini;Llama-3.1-8B-Instruct;Llama-3.1-8B-UltraMedical;Llama-3.1-Nemotron-Nano-8B-v1;MiMo-7B-RL-0530;MiniCPM4.1-8B;NVIDIA-Nemotron-Nano-9B-v2;OpenThinker3-7B;Qwen2.5-Coder-7B-Instruct;Qwen3-8B;cogito-v1-preview-llama-8B;gemma-2-9b-it;glm-4-9b-chat;granite-3.3-8b-instruct;internlm3-8b-instruct | aime;arcc;bbh;emorynlp;finqa;gpqa;humaneval;kandk;korbench;livecodebench;livemathbench;math500;mathbench;mbpp;medqa;meld;mmlupro;winogrande | aime:hybrid:60;arcc:test:1172;bbh:test:1080;emorynlp:test:697;finqa:test:1147;gpqa:test:198;humaneval:test:164;kandk:test:700;korbench:test:1250;livecodebench:test:1055;livemathbench:test:121;math500:test:500;mathbench:test:150;mbpp:test:974;medqa:test:1273;meld:test:1232;mmlupro:test_1000:1001;winogrande:valid:1267 |
| 32 | 5 | 2435 | 77920 | DeepHermes-3-Llama-3-8B-Preview;DeepSeek-R1-0528-Qwen3-8B;DeepSeek-R1-Distill-Qwen-7B;Fin-R1;GLM-Z1-9B-0414;Intern-S1-mini;Llama-3.1-8B-Instruct;Llama-3.1-8B-UltraMedical;Llama-3.1-Nemotron-Nano-8B-v1;MiMo-7B-RL-0530;MiniCPM4.1-8B;NVIDIA-Nemotron-Nano-9B-v2;OpenThinker3-7B;Qwen2.5-Coder-7B-Instruct;Qwen3-8B;cogito-v1-preview-llama-8B;gemma-2-9b-it;glm-4-9b-chat;granite-3.3-8b-instruct;internlm3-8b-instruct;claude-sonnet-4;deepseek-r1-0528;deepseek-v3-0324;gemini-2.5-flash;gemini-2.5-pro;glm-4.6;gpt-5;intern-s1;kimi-k2-0905;qwen3-235b-a22b-2507;qwen3-235b-a22b-thinking-2507;gpt-5-chat | aime;gpqa;livecodebench;livemathbench;mmlupro | aime:hybrid:60;gpqa:test:198;livecodebench:test:1055;livemathbench:test:121;mmlupro:test_1000:1001 |

Dataset coverage sample:

| dataset | domain | task_family | task_subtype | model_count | total_model_records |
| --- | --- | --- | --- | --- | --- |
| aime | math | mathematical_reasoning | competition_math | 38 | 2280 |
| arc-agi | reasoning | abstract_reasoning | grid_transformation | 17 | 6800 |
| arcc | science | scientific_reasoning | arc_challenge | 20 | 23440 |
| arenahard | instruction_following | instruction_following | arena_hard_general | 1 | 750 |
| bbh | reasoning | logical_reasoning | big_bench_hard | 20 | 21600 |
| emorynlp | dialogue | dialogue_understanding | conversation_analysis | 20 | 13940 |
| finqa | finance | financial_reasoning | financial_qa | 20 | 22940 |
| gpqa | science | scientific_reasoning | graduate_science_qa | 38 | 7524 |
| hle | broad_knowledge | broad_knowledge | expert_knowledge_exam | 17 | 39870 |
| humaneval | code | code_generation | function_synthesis | 22 | 3608 |
| kandk | logical_reasoning | logical_reasoning | knights_and_knaves | 20 | 14000 |
| korbench | multilingual | multilingual_reasoning | korean_benchmark | 20 | 25000 |
| livecodebench | code | code_generation | live_code_generation | 38 | 40090 |
| livemathbench | math | mathematical_reasoning | live_math_problem_solving | 35 | 4235 |
| math500 | math | mathematical_reasoning | math_word_problems | 20 | 10000 |
| mathbench | math | mathematical_reasoning | math_benchmark | 20 | 3000 |
| mbpp | code | code_generation | program_synthesis | 20 | 19480 |
| medqa | medicine | medical_reasoning | medical_qa | 20 | 25460 |
| meld | dialogue | dialogue_understanding | dialogue_emotion | 20 | 24640 |
| mmlupro | broad_knowledge | broad_knowledge | multi_domain_exam | 38 | 81036 |
