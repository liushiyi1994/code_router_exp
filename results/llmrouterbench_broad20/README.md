# RouteCode LLMRouterBench Pilot

This run uses `llmrouterbench` records converted into the RouteCode canonical schema. No model generation or external API calls are made by these evaluation scripts.

## Commands

```bash
python experiments/00_data_audit.py --config configs/llmrouterbench_broad20.yaml
python experiments/01_compression_ladder.py --config configs/llmrouterbench_broad20.yaml
python experiments/02_rate_distortion_curve.py --config configs/llmrouterbench_broad20.yaml
python experiments/03_residual_concentration.py --config configs/llmrouterbench_broad20.yaml
python experiments/04_split_sensitivity.py --config configs/llmrouterbench_broad20.yaml
python experiments/05_predictor_diagnostics.py --config configs/llmrouterbench_broad20.yaml
python experiments/06_predictability_constrained.py --config configs/llmrouterbench_broad20.yaml
python experiments/07_new_model_calibration.py --config configs/llmrouterbench_broad20.yaml
python experiments/08_ablation_summary.py --config configs/llmrouterbench_broad20.yaml
python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_broad20.yaml
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_broad20.yaml
python experiments/11_code_card_interpretability.py --config configs/llmrouterbench_broad20.yaml
python experiments/12_official_baseline_artifacts.py --config configs/llmrouterbench_broad20.yaml
python experiments/13_transformer_backbone_readiness.py --config configs/llmrouterbench_broad20.yaml
python experiments/28_transformer_embedding_router.py --config configs/llmrouterbench_broad20.yaml
python experiments/29_embedllm_knn_split_aligned.py --config configs/llmrouterbench_broad20.yaml
python experiments/30_frugalgpt_split_aligned.py --config configs/llmrouterbench_broad20.yaml
python experiments/14_routellm_pairwise_alignment.py --config configs/llmrouterbench_broad20.yaml
python experiments/15_routellm_mf_assets.py --config configs/llmrouterbench_broad20.yaml
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_broad20.yaml
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_broad20.yaml
python experiments/37_avengerspro_cli_metrics.py --config configs/llmrouterbench_broad20.yaml
python experiments/40_avengerspro_upstream_metric.py --config configs/llmrouterbench_broad20.yaml
python experiments/38_graphrouter_cli_metrics.py --config configs/llmrouterbench_broad20.yaml
python experiments/39_graphrouter_split_aligned.py --config configs/llmrouterbench_broad20.yaml
python experiments/18_model_pool_scale.py --config configs/llmrouterbench_broad20.yaml
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_broad20.yaml
python experiments/20_benchmark_coverage.py --config configs/llmrouterbench_broad20.yaml
python experiments/21_external_command_readiness.py --config configs/llmrouterbench_broad20.yaml
python experiments/22_cost_quality_frontier.py --config configs/llmrouterbench_broad20.yaml
python experiments/23_stronger_direct_router_probe.py --config configs/llmrouterbench_broad20.yaml
python experiments/26_external_baseline_assets.py --config configs/llmrouterbench_broad20.yaml
python experiments/27_llmrouter_library_adapters.py --config configs/llmrouterbench_broad20.yaml
python experiments/25_provider_price_sensitivity.py --config configs/llmrouterbench_broad20.yaml
pytest -q
```

## Outputs

- `table_routability.csv`: best single, cheapest, dataset-label lookup, and query-oracle audit.
- `table_recovered_gap.csv`: compression ladder with bootstrap confidence intervals.
- `table_rate_distortion.csv`: semantic-cluster and RouteCode curves for K = 1, 2, 4, 8, 16, 32, 64, 128.
- `code_cards.md`, `code_cards.json`, and `fig_code_label_heatmap.pdf`: train-set summaries for learned route labels.
- `fig_compression_ladder.pdf` and `fig_rate_distortion.pdf`: main pilot figures.
- `table_residual_concentration.csv`, `table_residual_risk.csv`, `fig_residual_concentration.pdf`, `fig_risk_coverage.pdf`, and `phase_d5_adaptive_refinement_gate_memo.md`: residual-regret concentration and adaptive-refinement gate diagnostics.
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
- `table_transformer_embedding_router.csv` and `phase_f_g_transformer_embedding_router_memo.md`: local-files-only pretrained encoder direct-router rows or skipped/failed blocker rows.
- `table_embedllm_knn_split_aligned.csv` and `phase_e_embedllm_knn_split_aligned_memo.md`: split-aligned EmbedLLM KNN local metric-adapter evaluation.
- `table_frugalgpt_split_aligned.csv` and `phase_e_frugalgpt_split_aligned_memo.md`: split-aligned FrugalGPT local-scorer metric-adapter evaluation.
- `table_routellm_pairwise_alignment.csv` and `phase_e_routellm_pairwise_alignment_memo.md`: split-aligned RouteLLM pairwise substrate readiness audit.
- `table_routellm_mf_assets.csv` and `phase_e_routellm_mf_assets_memo.md`: split-aligned RouteLLM MF trainer asset readiness audit.
- `table_routellm_mf_split_aligned.csv` and `phase_e_routellm_mf_split_aligned_memo.md`: split-aligned RouteLLM MF training-code evaluation.
- `table_avengerspro_split_aligned.csv` and `phase_e_avengerspro_split_aligned_memo.md`: split-aligned local Avengers-Pro cluster-routing compatibility evaluation.
- `table_avengerspro_cli_metrics.csv` and `phase_e_avengerspro_cli_metrics_memo.md`: exact upstream Avengers-Pro simple-cluster accuracy/cost metrics on split-aligned assets; not RouteCode utility rows.
- `table_avengerspro_upstream_metric.csv` and `phase_e_avengerspro_upstream_metric_memo.md`: RouteCode utility metrics over Avengers-Pro simple-cluster selections captured from upstream model code; not an exact upstream command row.
- `table_graphrouter_cli_metrics.csv` and `phase_e_graphrouter_cli_metrics_memo.md`: exact upstream GraphRouter one-epoch smoke accuracy/cost metrics on split-aligned assets; not RouteCode utility rows.
- `table_graphrouter_split_aligned.csv` and `phase_e_graphrouter_split_aligned_memo.md`: RouteCode utility metrics over split-aligned GraphRouter GNN selections using upstream model code; not an exact upstream command row.
- `table_model_pool_scale.csv` and `phase_f_g_model_pool_scale_memo.md`: larger model-pool scale/composition robustness diagnostics.
- `table_model_pool_transfer.csv` and `phase_f_g_model_pool_transfer_memo.md`: held-out model-pool transfer diagnostics.
- `table_benchmark_file_coverage.csv`, `table_benchmark_dataset_coverage.csv`, `table_broad_coverage_candidates.csv`, and `phase_g_benchmark_coverage_memo.md`: raw LLMRouterBench coverage and broad complete-rectangle diagnostics.
- `table_external_command_readiness.csv` and `phase_e_external_command_readiness_memo.md`: reproducible exact upstream-command readiness audit for remaining external baselines.
- `table_cost_quality_summary.csv`, `table_cost_quality_frontier.csv`, `fig_cost_quality_frontier.pdf`, and `phase_e_cost_quality_memo.md`: fixed-quality and fixed-cost operating-point diagnostics.
- `table_stronger_direct_router_probe.csv` and `phase_e_stronger_direct_router_probe_memo.md`: bounded stronger direct-router probe for MLP and gradient-boosting retraining baselines.
- `table_external_baseline_assets.csv` and `phase_e_external_baseline_assets_memo.md`: split-aligned input assets for additional upstream external baseline command paths.
- `table_llmrouter_library_adapters.csv` and `phase_e_llmrouter_library_adapters_memo.md`: split-aligned local LLMRouter trainer-class adapter metrics.
- `table_provider_price_schedule.csv`, `table_provider_cost_quality_summary.csv`, `table_provider_cost_quality_frontier.csv`, `fig_provider_price_sensitivity.pdf`, and `phase_g_provider_pricing_memo.md`: partial provider-price sensitivity diagnostics for mapped provider models.
- `phase_c_observation_memo.md`: Phase C checkpoint memo answering the seven pilot questions.
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
- Universal Model Routing; paper: https://openreview.net/pdf?id=ka82fvJ5f1
- kNN routing; paper: https://arxiv.org/abs/2505.12601; repo: https://github.com/ulab-uiuc/LLMRouter
- Causal LLM Routing; paper: https://openreview.net/forum?id=iZC5xoQQkX

## Leakage Controls

- Train/validation/test splits are assigned by `query_id`; all model rows for a query stay in the same split.
- Best-single, dataset/topic tables, embedding clusters, kNN neighbors, and RouteCode codebooks are fit on train only.
- Query oracle uses test utility only as an upper bound.
- The leaky dataset-label diagnostic is written separately to `table_leakage_gap.csv` and is not a deployable baseline.

## Next Steps

1. Broaden real-data domain metadata beyond the current coarse dataset-to-domain map.
2. Run held-out model-pool transfer checks and direct-router retraining comparisons before making transfer claims.
3. Test whether residual predictors such as centroid distance, margin, or kNN disagreement identify the high-regret tail.
4. Add official external baselines or stronger local adapters before making method-ranking claims.
5. Keep final claims scoped to pilot evidence until broader robustness checks pass.

## External Command Readiness

Command:

```bash
python experiments/21_external_command_readiness.py --config configs/llmrouterbench_broad20.yaml
```

Outputs:

- `table_external_command_readiness.csv`: reproducible readiness table for exact upstream external-baseline commands.
- `phase_e_external_command_readiness_memo.md`: memo explaining runnable rows and blockers.

Runnable exact upstream-command rows now: `11`.

| check_id | status | runnable_now | no_api_compatible | routecode_metric_compatible | exact_upstream_command | blocking_reasons | execution_evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| routecode_local_routellm_mf_metric | available | True | True | True | False |  |  |
| routecode_local_embedllm_knn_metric | available | True | True | True | False |  |  |
| routecode_local_frugalgpt_metric | available | True | True | True | False |  |  |
| routecode_upstream_avengerspro_metric | available | True | True | True | False |  | results/llmrouterbench_broad20/avengerspro_upstream_metric/raw_routing_details.json |
| llmrouter_knn_train_cli | smoke_executed | True | True | False | True |  | results/llmrouterbench_broad20/llmrouter_library_adapters/llmrouter_knn_train_stdout.log |
| llmrouter_svm_train_cli | smoke_executed | True | True | False | True |  | results/llmrouterbench_broad20/llmrouter_library_adapters/llmrouter_svm_train_stdout.log |
| llmrouter_knn_infer_cli | executed | True | True | False | True |  | results/llmrouterbench_broad20/llmrouter_library_adapters/llmrouter_knn_full_predictions.json |
| llmrouter_svm_infer_cli | executed | True | True | False | True |  | results/llmrouterbench_broad20/llmrouter_library_adapters/llmrouter_svm_full_predictions.json |
| routellm_mf_train_cli | executed | True | True | False | True |  | results/llmrouterbench_broad20/routellm_mf_assets/mf_model.pt |
| routellm_mf_eval_cli | smoke_executed | True | True | False | True |  | results/llmrouterbench_broad20/routellm_mf_assets/routellm_mf_eval_stdout.log |
| routellm_bert_cli | blocked | False | True | False | False | missing_bert_checkpoint |  |
| avengerspro_cli | executed | True | True | False | True |  | results/llmrouterbench_broad20/avengerspro_cli_metrics/simple_cluster_full_results.json |
| graphrouter_cli | executed | True | True | False | True |  | results/llmrouterbench_broad20/graphrouter_cli_metrics/graphrouter_stdout.log |
| frugalgpt_local_scorer_cli | smoke_executed | True | True | False | True |  | results/llmrouterbench_broad20/frugalgpt_split_aligned/output/frugalgpt_smoke_stdout.log |
| embedllm_knn_cli | executed | True | True | False | True |  | results/llmrouterbench_broad20/embedllm_knn_cli_metrics/embedllm_knn_k131_stdout.log |
| embedllm_mf_cli | executed | True | True | False | True |  | results/llmrouterbench_broad20/embedllm_mf_cli_metrics/embedllm_mf_stdout.log |
| best_route_train_cli | blocked | False | True | False | True | missing_best_route_local_model_checkpoint;missing_python_modules:llm_blender |  |
| routerdc_train_cli | blocked | False | True | False | True | missing_routerdc_local_model_checkpoint;missing_python_modules:deepspeed |  |
| modelsat_train_cli | blocked | False | True | False | True | missing_modelsat_base_model_checkpoint;missing_modelsat_embedding_model_checkpoint;missing_python_modules:nltk,deepspeed |  |

## Phase E Baseline Coverage

Command:

```bash
python experiments/44_phase_e_baseline_coverage.py --result-dir results/llmrouterbench_broad20
```

Outputs:

- `table_phase_e_baseline_coverage.csv`: Research Flow Phase E baseline coverage audit.
- `phase_e_baseline_coverage_memo.md`: memo distinguishing required/conditional baseline coverage from optional checkpoint-gated external rows.

Required/conditional baseline coverage complete: `True`.

| requirement_id | requirement | requirement_type | status | evidence | notes |
| --- | --- | --- | --- | --- | --- |
| random | Random routing baseline | required | present | random | Evidence present: random. |
| cheapest | Cheapest-model baseline | required | present | cheapest | Evidence present: cheapest. |
| best_single | Best single model | required | present | best_single,best_single | Evidence present: best_single,best_single. |
| dataset_oracle | Dataset oracle | required | present | dataset_oracle | Evidence present: dataset_oracle. |
| query_oracle | Query oracle | required | present | query_oracle,query_oracle | Evidence present: query_oracle,query_oracle. |
| dataset_label_lookup | Dataset-label lookup | required | present | dataset_label_lookup,dataset_label_lookup | Evidence present: dataset_label_lookup,dataset_label_lookup. |
| predicted_topic_lookup | Predicted-topic lookup | required | present | predicted_topic_lookup | Evidence present: predicted_topic_lookup. |
| embedding_cluster_lookup | Embedding-cluster lookup | required | present | embedding_cluster_lookup | Evidence present: embedding_cluster_lookup. |
| knn | kNN router | required | present | kNN | Evidence present: kNN. |
| logistic_mlp_svm | MLP/SVM/simple learned routers | required | present | logistic_embedding_router,mlp_embedding_router,svm_embedding_router | Evidence present: logistic_embedding_router,mlp_embedding_router,svm_embedding_router. |
| route_llm_if_easy | RouteLLM baseline when locally runnable | conditional | present | routellm_mf_split_aligned | Evidence present: routellm_mf_split_aligned. |
| llmrouter_if_available | LLMRouter baselines when locally available | conditional | present | llmrouter_library_knn,llmrouter_library_svm | Evidence present: llmrouter_library_knn,llmrouter_library_svm. |
| graphrouter_if_available | GraphRouter baseline when locally available | conditional | present | graphrouter_split_aligned | Evidence present: graphrouter_split_aligned. |
| avengerspro_if_included | Avengers-Pro baseline when included in LLMRouterBench | conditional | present | avengerspro_upstream_simple_cluster | Evidence present: avengerspro_upstream_simple_cluster. |
| cost_quality_metrics | Cost-quality frontier metrics | required | present | cost_quality | Evidence present: cost_quality. |
| optional_extra_external_blockers | Extra checkpoint-heavy external baselines beyond Research Flow required list | optional | present | routellm_bert_cli,best_route_train_cli,routerdc_train_cli,modelsat_train_cli | Optional external blockers documented, not required for Phase E completion. |
