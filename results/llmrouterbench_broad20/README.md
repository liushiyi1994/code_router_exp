# RouteCode LLMRouterBench Broad20 Run

This run uses the audited 18-dataset/20-model LLMRouterBench complete rectangle converted into the RouteCode canonical schema. No model generation or external API calls are made by these evaluation scripts. Completed broad20 evidence currently covers B0/B1/B2, D2 predictability-constrained RouteCode, the residual/adaptive-refinement gate, bounded split sensitivity, bounded ablations, bounded held-out model-pool transfer, local external-style baseline surrogates, split-aligned RouteLLM MF official-code evaluation, and a split-aligned local Avengers-Pro compatibility baseline.

## Commands

```bash
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
- `phase_b_broad20_memo.md`: broad Phase B checkpoint memo.
- `table_predictability_constrained.csv`, `fig_predictability_constrained_tradeoff.pdf`, `code_cards_predictability_constrained.md`, and `phase_d_method_memo.md`: broad D2 diagnostics.
- `table_residual_concentration.csv`, `table_residual_risk.csv`, `table_residual_queries.csv`, `table_residual_by_label.csv`, `fig_residual_concentration.pdf`, `fig_risk_coverage.pdf`, and `phase_d5_adaptive_refinement_gate_memo.md`: broad residual/adaptive-refinement gate diagnostics.
- `table_split_sensitivity.csv`, `table_split_rank_correlation.csv`, `table_split_rate_threshold.csv`, and `fig_split_sensitivity.pdf`: bounded broad split-sensitivity diagnostics.
- `table_ablation_summary.csv`, `fig_sensitivity_k_lambda.pdf`, `fig_seed_stability.pdf`, and `phase_f_g_ablation_memo.md`: bounded broad ablation diagnostics.
- `table_model_pool_transfer.csv` and `phase_f_g_model_pool_transfer_memo.md`: bounded broad disjoint 8-source/8-target model-pool transfer diagnostics.
- `table_external_baselines.csv` and `phase_e_external_baseline_memo.md`: broad local external-style baseline surrogate diagnostics, not official upstream-command reproductions.
- `table_routellm_mf_split_aligned.csv`, `phase_e_routellm_mf_split_aligned_memo.md`, and `routellm_mf_split_aligned/`: broad RouteLLM MF official-code evaluation with local RouteCode embeddings, not the upstream published checkpoint.
- `table_avengerspro_split_aligned.csv`, `phase_e_avengerspro_split_aligned_memo.md`, and `avengerspro_split_aligned/`: broad local Avengers-Pro cluster-routing compatibility baseline, not an upstream command-path reproduction.
- `outcomes.csv` and `query_embeddings.csv`: canonical input rows and deterministic local query features used for this run.

## First Results

| method | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- |
| random | 0.5438 | 0.3722 | -0.7534 |
| best_single | 0.7037 | 0.2123 | 0.0000 |
| dataset_oracle | 0.7400 | 0.1759 | 0.1711 |
| kNN | 0.7023 | 0.2137 | -0.0067 |
| svm_embedding_router | 0.5018 | 0.4142 | -0.9513 |
| query_oracle | 0.9160 | 0.0000 | 1.0000 |

Utility-oracle RouteCode rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.7037 | 0.2123 | 0.0000 |
| 2 | 1.0000 | 0.9932 | 0.7037 | 0.2123 | 0.0000 |
| 4 | 2.0000 | 1.9672 | 0.7073 | 0.2087 | 0.0168 |
| 8 | 3.0000 | 2.9212 | 0.7290 | 0.1870 | 0.1191 |
| 16 | 4.0000 | 3.8177 | 0.7813 | 0.1346 | 0.3658 |
| 32 | 5.0000 | 4.7183 | 0.8130 | 0.1029 | 0.5151 |
| 64 | 6.0000 | 5.5982 | 0.8536 | 0.0623 | 0.7064 |
| 128 | 7.0000 | 6.4661 | 0.8714 | 0.0445 | 0.7903 |

Regret-objective RouteCode oracle rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.7037 | 0.2123 | 0.0000 |
| 2 | 1.0000 | 0.4636 | 0.7439 | 0.1720 | 0.1896 |
| 4 | 2.0000 | 1.4469 | 0.8419 | 0.0741 | 0.6510 |
| 8 | 3.0000 | 2.2734 | 0.8850 | 0.0310 | 0.8540 |
| 16 | 4.0000 | 2.7960 | 0.9092 | 0.0068 | 0.9681 |
| 32 | 5.0000 | 3.6803 | 0.9160 | 0.0000 | 1.0000 |
| 64 | 6.0000 | 4.4632 | 0.9160 | 0.0000 | 1.0000 |
| 128 | 7.0000 | 4.6811 | 0.9160 | 0.0000 | 1.0000 |

Predicted RouteCode rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.7037 | 0.2123 | 0.0000 |
| 2 | 1.0000 | 0.9860 | 0.7037 | 0.2123 | 0.0000 |
| 4 | 2.0000 | 1.8807 | 0.7026 | 0.2133 | -0.0050 |
| 8 | 3.0000 | 2.6964 | 0.6403 | 0.2756 | -0.2987 |
| 16 | 4.0000 | 3.4755 | 0.6022 | 0.3137 | -0.4782 |
| 32 | 5.0000 | 4.3071 | 0.6261 | 0.2899 | -0.3658 |
| 64 | 6.0000 | 5.3968 | 0.5634 | 0.3526 | -0.6611 |
| 128 | 7.0000 | 6.5587 | 0.5783 | 0.3376 | -0.5906 |

These values come from the audited LLMRouterBench broad20 rectangle. Treat them as broad Phase B observations, not paper-level claims.

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
- Universal Model Routing; paper: https://openreview.net/pdf?id=ka82fvJ5f1
- kNN routing; paper: https://arxiv.org/abs/2505.12601; repo: https://github.com/ulab-uiuc/LLMRouter
- Causal LLM Routing; paper: https://openreview.net/forum?id=iZC5xoQQkX

## Leakage Controls

- Train/validation/test splits are assigned by `query_id`; all model rows for a query stay in the same split.
- Best-single, dataset/topic tables, embedding clusters, kNN neighbors, and RouteCode codebooks are fit on train only.
- Query oracle uses test utility only as an upper bound.
- The leaky dataset-label diagnostic is written separately to `table_leakage_gap.csv` and is not a deployable baseline.

## Next Steps

1. Run exact upstream external baselines where dependencies can be pinned.
2. Add cached pretrained encoder-backbone rows when suitable checkpoints are available.
3. Expand broad20 split sensitivity, ablations, and transfer beyond the bounded local sweeps if runtime permits.
4. Run stricter held-out model-pool transfer checks on larger benchmark-scale model pools.
5. Keep final claims scoped to verified broad evidence until broader robustness checks pass.

## Predictability-Constrained RouteCode

Command:

```bash
python experiments/06_predictability_constrained.py --config configs/llmrouterbench_broad20.yaml
```

Outputs:

- `table_predictability_constrained.csv`: alpha sweep with D2 joint-label, embedding-centroid, and logistic-label rows plus comparison baselines.
- `fig_predictability_constrained_tradeoff.pdf`: D2 utility and label-predictability tradeoff by alpha.
- `code_cards_predictability_constrained.md` and `code_cards_predictability_constrained.json`: code cards for the selected D2 alpha.
- `fig_code_label_heatmap_predictability_constrained.pdf`: selected D2 label utility profiles.
- `phase_d_method_memo.md`: D2 checkpoint memo and recommended interpretation.

| method | alpha | mean_utility | recovered_gap_vs_oracle | label_accuracy |
| --- | --- | --- | --- | --- |
| d2_embedding_centroid | 0.0000 | 0.6585 | -0.2131 | 0.2204 |
| d2_embedding_centroid | 0.0500 | 0.6592 | -0.2097 | 0.3048 |
| d2_embedding_centroid | 0.1000 | 0.6802 | -0.1107 | 0.3896 |
| d2_embedding_centroid | 0.3000 | 0.7197 | 0.0755 | 0.5648 |
| d2_embedding_centroid | 1.0000 | 0.7229 | 0.0906 | 0.7646 |
| d2_embedding_centroid | 3.0000 | 0.7172 | 0.0638 | 0.9776 |
| d2_embedding_centroid | 10.0000 | 0.7190 | 0.0721 | 0.9950 |
| d2_logistic_label_predictor | 0.0000 | 0.6115 | -0.4346 | 0.2365 |
| d2_logistic_label_predictor | 0.0500 | 0.6150 | -0.4178 | 0.3063 |
| d2_logistic_label_predictor | 0.1000 | 0.6278 | -0.3574 | 0.3914 |
| d2_logistic_label_predictor | 0.3000 | 0.7176 | 0.0654 | 0.5552 |
| d2_logistic_label_predictor | 1.0000 | 0.7187 | 0.0705 | 0.7472 |
| d2_logistic_label_predictor | 3.0000 | 0.7176 | 0.0654 | 0.9523 |
| d2_logistic_label_predictor | 10.0000 | 0.7194 | 0.0738 | 0.9537 |
| dataset_label_lookup |  | 0.7172 | 0.0638 |  |
| flat_routecode_logistic_label_predictor |  | 0.6022 | -0.4782 |  |
| kNN |  | 0.7023 | -0.0067 |  |
| semantic_embedding_kmeans |  | 0.7222 | 0.0872 |  |

## Residual Concentration

Command:

```bash
python experiments/03_residual_concentration.py --config configs/llmrouterbench_broad20.yaml
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
| 0.0500 | 141.0000 | 141.0000 | 881.0000 | 0.1600 |
| 0.1000 | 281.0000 | 281.0000 | 881.0000 | 0.3190 |
| 0.2000 | 562.0000 | 562.0000 | 881.0000 | 0.6379 |

Deployable risk coverage:

| score | top_fraction | regret_mass_fraction | positive_regret_recall | auc_regret_positive |
| --- | --- | --- | --- | --- |
| low_route_label_confidence | 0.0500 | 0.0556 | 0.0556 | 0.5370 |
| low_route_label_confidence | 0.1000 | 0.1056 | 0.1056 | 0.5370 |
| centroid_distance_risk | 0.0500 | 0.0397 | 0.0397 | 0.4681 |
| centroid_distance_risk | 0.1000 | 0.0851 | 0.0851 | 0.4681 |
| knn_disagreement | 0.0500 | 0.0386 | 0.0386 | 0.5272 |
| knn_disagreement | 0.1000 | 0.0817 | 0.0817 | 0.5272 |
| knn_disagreement_plus_distance | 0.0500 | 0.0420 | 0.0420 | 0.4979 |
| knn_disagreement_plus_distance | 0.1000 | 0.0874 | 0.0874 | 0.4979 |

## Split Sensitivity

Command:

```bash
python experiments/04_split_sensitivity.py --config configs/llmrouterbench_broad20.yaml
```

Outputs:

- `table_split_sensitivity.csv`: method metrics for each split scenario.
- `table_split_rank_correlation.csv`: ranking correlation and degradation against the random split.
- `table_split_rate_threshold.csv`: RouteCode predicted-label rate needed to recover 80% learned-router gain when reached.
- `fig_split_sensitivity.pdf`: heatmap of recovered gap vs oracle across scenarios.

Random split ranking snapshot:

| method | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- |
| query_oracle | 0.9160 | 1.0000 |
| embedding_cluster_lookup | 0.7222 | 0.0872 |
| dataset_label_lookup | 0.7172 | 0.0638 |
| predicted_topic_lookup | 0.7169 | 0.0621 |
| best_single | 0.7037 | 0.0000 |
| kNN | 0.7023 | -0.0067 |
| routecode_predicted_labels | 0.6022 | -0.4782 |
| logistic_embedding_router | 0.5135 | -0.8960 |

Lowest rank correlations vs random:

| scenario | scenario_type | rank_correlation_vs_random | mean_absolute_utility_delta_vs_random | mean_absolute_recovered_gap_delta_vs_random |
| --- | --- | --- | --- | --- |
| leave_dataset_out:aime | leave_one_dataset_out | 0.7832 | 0.0768 | 0.8362 |
| cluster_held_out:0 | cluster_held_out | 0.7904 | 0.1139 | 0.5672 |
| leave_domain_out:broad_knowledge | leave_one_domain_out | 0.9581 | 0.0603 | 0.1618 |
| domain_homogeneous:broad_knowledge | domain_homogeneous | 0.9759 | 0.0188 | 0.0780 |
| model_pool_holdout:DeepHermes-3-Llama-3-8B-Preview | model_pool_holdout | 0.9762 | 0.0022 | 0.0103 |
| random | random | 1.0000 | 0.0000 | 0.0000 |

## Ablation And Robustness

Command:

```bash
python experiments/08_ablation_summary.py --config configs/llmrouterbench_broad20.yaml
```

Outputs:

- `table_ablation_summary.csv`: bounded seed, K/lambda, rate-penalty, and training-fraction ablation rows.
- `fig_sensitivity_k_lambda.pdf`: recovered-gap heatmaps over K and lambda.
- `fig_seed_stability.pdf`: seed-stability bars with standard deviations.
- `phase_f_g_ablation_memo.md`: robustness checkpoint memo.

| ablation | method | mean_recovered_gap | min_recovered_gap | max_recovered_gap |
| --- | --- | --- | --- | --- |
| k_lambda | regret_routecode_utility_oracle | 0.9360 | 0.8272 | 1.0000 |
| k_lambda | flat_routecode_utility_oracle | 0.3155 | 0.1191 | 0.5151 |
| k_lambda | semantic_embedding_kmeans | 0.0792 | 0.0587 | 0.0940 |
| k_lambda | d2_embedding_centroid | 0.0696 | 0.0537 | 0.0906 |
| k_lambda | best_single | 0.0000 | 0.0000 | 0.0000 |
| rate_penalty | d2_embedding_centroid | 0.0906 | 0.0906 | 0.0906 |
| seed_stability | d2_embedding_centroid | 0.0990 | 0.0906 | 0.1073 |
| seed_stability | best_single | 0.0000 | 0.0000 | 0.0000 |
| seed_stability | kNN | -0.0102 | -0.0136 | -0.0067 |
| seed_stability | logistic_embedding_router | -0.9608 | -1.0256 | -0.8960 |
| seed_stability | svm_embedding_router | -1.0157 | -1.0801 | -0.9513 |
| train_fraction | logistic_embedding_router | -0.8691 | -0.9010 | -0.8372 |
| train_fraction | svm_embedding_router | -0.9111 | -0.9513 | -0.8708 |

## Held-Out Model-Pool Transfer

Command:

```bash
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_broad20.yaml
```

Outputs:

- `table_model_pool_transfer.csv`: disjoint source/target pool transfer rows for target baselines, native D2, and transferred source-D2 labels.
- `phase_f_g_model_pool_transfer_memo.md`: held-out model-pool transfer checkpoint memo.

| transfer_scenario | source_model_count | target_model_count | method | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| complementary_to_remaining_top | 8 | 8 | source_d2_label_transfer | 0.7140 | 0.1739 |
| complementary_to_remaining_top | 8 | 8 | target_best_single | 0.6813 | 0.0000 |
| complementary_to_remaining_top | 8 | 8 | target_d2_native | 0.7130 | 0.1682 |
| complementary_to_remaining_top | 8 | 8 | target_direct_knn | 0.6830 | 0.0095 |
| complementary_to_remaining_top | 8 | 8 | target_direct_logistic | 0.6841 | 0.0151 |
| complementary_to_remaining_top | 8 | 8 | target_direct_svm | 0.6820 | 0.0038 |
| complementary_to_remaining_top | 8 | 8 | target_kNN | 0.6927 | 0.0605 |
| dominated_to_remaining_top | 8 | 8 | source_d2_label_transfer | 0.7123 | 0.2162 |
| dominated_to_remaining_top | 8 | 8 | target_best_single | 0.6656 | 0.0000 |
| dominated_to_remaining_top | 8 | 8 | target_d2_native | 0.7098 | 0.2046 |
| dominated_to_remaining_top | 8 | 8 | target_direct_knn | 0.6660 | 0.0017 |
| dominated_to_remaining_top | 8 | 8 | target_direct_logistic | 0.6699 | 0.0198 |
| dominated_to_remaining_top | 8 | 8 | target_direct_svm | 0.6674 | 0.0083 |
| dominated_to_remaining_top | 8 | 8 | target_kNN | 0.7037 | 0.1766 |
| top_to_next | 8 | 8 | source_d2_label_transfer | 0.6047 | 0.2274 |
| top_to_next | 8 | 8 | target_best_single | 0.5410 | 0.0000 |
| top_to_next | 8 | 8 | target_d2_native | 0.6136 | 0.2592 |
| top_to_next | 8 | 8 | target_direct_knn | 0.5520 | 0.0394 |
| top_to_next | 8 | 8 | target_direct_logistic | 0.5495 | 0.0305 |
| top_to_next | 8 | 8 | target_direct_svm | 0.5413 | 0.0013 |
| top_to_next | 8 | 8 | target_kNN | 0.5990 | 0.2071 |

## External Baseline Surrogates

Command:

```bash
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_broad20.yaml
```

Outputs:

- `table_external_baselines.csv`: local external-style baseline surrogate rows with explicit implementation notes.
- `phase_e_external_baseline_memo.md`: Phase E checkpoint memo for these surrogate baselines.

| method | baseline_family | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- |
| query_oracle | reference | 0.9160 | 1.0000 |
| best_single | reference | 0.7037 | 0.0000 |
| routellm_pair_strong_only | binary_pair_reference | 0.7037 | 0.0000 |
| kNN | reference | 0.7023 | -0.0067 |
| routellm_style_mf_utility_router | external_style_surrogate | 0.6934 | -0.0487 |
| routellm_binary_logistic_surrogate_t0.25 | external_style_surrogate | 0.6706 | -0.1560 |
| routellm_binary_logistic_surrogate_t0.5 | external_style_surrogate | 0.5545 | -0.7030 |
| routellm_binary_logistic_surrogate_t0.75 | external_style_surrogate | 0.3846 | -1.5034 |
| routellm_pair_weak_only | binary_pair_reference | 0.3462 | -1.6846 |

## RouteLLM MF Split-Aligned Evaluation

Command:

```bash
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_broad20.yaml
```

Outputs:

- `routellm_mf_split_aligned/mf_model.pt`: checkpoint trained with local LLMRouterBench RouteLLM MF source.
- `routellm_mf_split_aligned/raw_metrics.json`: threshold-level quality metrics and win rates.
- `table_routellm_mf_split_aligned.csv`: RouteCode utility metrics plus RouteLLM-style quality metrics.
- `phase_e_routellm_mf_split_aligned_memo.md`: Phase E memo and caveats.

Artifact directory: `results/llmrouterbench_broad20/routellm_mf_split_aligned`.

This uses official MF training code with local RouteCode embeddings; it is not the upstream published RouteLLM checkpoint.

| mean_utility | oracle_regret | mean_quality | normalized_cost | method | K | utility_ci_low | utility_ci_high | recovered_gap_vs_learned | recovered_gap_vs_oracle | selected_model_entropy | rate_log2K | empirical_H_Z | threshold | strong_model | weak_model | selection_accuracy | routing_accuracy_decisive | decisive_count | tie_count | strong_selection_rate | weak_selection_rate | mean_strong_win_rate | train_loss | validation_accuracy | official_training_code_used | official_upstream_checkpoint | split_aligned_with_routecode | routecode_metric_compatible | baseline_family | implementation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7041 | 0.2119 | 0.7041 | 0.0973 | routellm_mf_split_aligned_t0.25 | 2 | 0.6880 | 0.7206 | 0.0000 | 0.0017 | 0.0046 | 1.0000 | 0.0046 | 0.2500 | Qwen3-8B | MiMo-7B-RL-0530 | 0.7041 | 0.9089 | 1230 | 1578 | 0.9996 | 0.0004 | 0.9067 | None | 0.9106 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |
| 0.7073 | 0.2087 | 0.7073 | 0.0959 | routellm_mf_split_aligned_t0.5 | 2 | 0.6910 | 0.7222 | 0.0000 | 0.0168 | 0.0945 | 1.0000 | 0.0945 | 0.5000 | Qwen3-8B | MiMo-7B-RL-0530 | 0.7073 | 0.9163 | 1230 | 1578 | 0.9879 | 0.0121 | 0.9067 | None | 0.9106 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |
| 0.7001 | 0.2158 | 0.7001 | 0.0888 | routellm_mf_split_aligned_t0.75 | 2 | 0.6845 | 0.7140 | 0.0000 | -0.0168 | 0.4138 | 1.0000 | 0.4138 | 0.7500 | Qwen3-8B | MiMo-7B-RL-0530 | 0.7001 | 0.9000 | 1230 | 1578 | 0.9167 | 0.0833 | 0.9067 | None | 0.9106 | True | False | True | True | official_code_local_embedding | LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; not the upstream published RouteLLM checkpoint. |

## Avengers-Pro Split-Aligned Evaluation

Command:

```bash
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_broad20.yaml
```

Outputs:

- `table_avengerspro_split_aligned.csv`: RouteCode utility rows from a local implementation of the Avengers-Pro cluster-routing contract.
- `avengerspro_split_aligned/train.jsonl`, `avengerspro_split_aligned/test.jsonl`, and `avengerspro_split_aligned/baseline_scores.json`: split-aligned Avengers-Pro-format assets.
- `phase_e_avengerspro_split_aligned_memo.md`: caveats and adapter notes.

| method | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- |
| avengerspro_simple_cluster_k16 | 0.6574 | -0.2181 |
| avengerspro_balance_cluster_k16_w0.7_c0.3 | 0.6449 | -0.2768 |
