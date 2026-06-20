# RouteCode LLMRouterBench Pilot

This run uses `llmrouterbench` records converted into the RouteCode canonical schema. No model generation or external API calls are made by these evaluation scripts.

## Commands

```bash
python experiments/00_data_audit.py --config configs/llmrouterbench_scale20.yaml
python experiments/01_compression_ladder.py --config configs/llmrouterbench_scale20.yaml
python experiments/02_rate_distortion_curve.py --config configs/llmrouterbench_scale20.yaml
python experiments/03_residual_concentration.py --config configs/llmrouterbench_scale20.yaml
python experiments/04_split_sensitivity.py --config configs/llmrouterbench_scale20.yaml
python experiments/05_predictor_diagnostics.py --config configs/llmrouterbench_scale20.yaml
python experiments/06_predictability_constrained.py --config configs/llmrouterbench_scale20.yaml
python experiments/07_new_model_calibration.py --config configs/llmrouterbench_scale20.yaml
python experiments/08_ablation_summary.py --config configs/llmrouterbench_scale20.yaml
python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_scale20.yaml
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_scale20.yaml
python experiments/11_code_card_interpretability.py --config configs/llmrouterbench_scale20.yaml
python experiments/12_official_baseline_artifacts.py --config configs/llmrouterbench_scale20.yaml
python experiments/13_transformer_backbone_readiness.py --config configs/llmrouterbench_scale20.yaml
python experiments/28_transformer_embedding_router.py --config configs/llmrouterbench_scale20.yaml
python experiments/29_embedllm_knn_split_aligned.py --config configs/llmrouterbench_scale20.yaml
python experiments/30_frugalgpt_split_aligned.py --config configs/llmrouterbench_scale20.yaml
python experiments/14_routellm_pairwise_alignment.py --config configs/llmrouterbench_scale20.yaml
python experiments/15_routellm_mf_assets.py --config configs/llmrouterbench_scale20.yaml
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_scale20.yaml
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_scale20.yaml
python experiments/37_avengerspro_cli_metrics.py --config configs/llmrouterbench_scale20.yaml
python experiments/18_model_pool_scale.py --config configs/llmrouterbench_scale20.yaml
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_scale20.yaml
python experiments/20_benchmark_coverage.py --config configs/llmrouterbench_scale20.yaml
python experiments/21_external_command_readiness.py --config configs/llmrouterbench_scale20.yaml
python experiments/22_cost_quality_frontier.py --config configs/llmrouterbench_scale20.yaml
python experiments/23_stronger_direct_router_probe.py --config configs/llmrouterbench_scale20.yaml
python experiments/26_external_baseline_assets.py --config configs/llmrouterbench_scale20.yaml
python experiments/27_llmrouter_library_adapters.py --config configs/llmrouterbench_scale20.yaml
python experiments/25_provider_price_sensitivity.py --config configs/llmrouterbench_scale20.yaml
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
| random | 0.5672 | 0.3966 | -0.5436 |
| best_single | 0.7069 | 0.2569 | 0.0000 |
| dataset_oracle | 0.7759 | 0.1879 | 0.2685 |
| kNN | 0.7241 | 0.2397 | 0.0671 |
| svm_embedding_router | 0.5172 | 0.4466 | -0.7383 |
| query_oracle | 0.9638 | 0.0000 | 1.0000 |

Utility-oracle RouteCode rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.7069 | 0.2569 | 0.0000 |
| 2 | 1.0000 | 0.9767 | 0.7069 | 0.2569 | 0.0000 |
| 4 | 2.0000 | 1.9371 | 0.7431 | 0.2207 | 0.1409 |
| 8 | 3.0000 | 2.8567 | 0.7741 | 0.1897 | 0.2617 |
| 16 | 4.0000 | 3.8819 | 0.8155 | 0.1483 | 0.4228 |
| 32 | 5.0000 | 4.7004 | 0.8172 | 0.1466 | 0.4295 |
| 64 | 6.0000 | 5.5857 | 0.8603 | 0.1034 | 0.5973 |
| 128 | 7.0000 | 6.3319 | 0.9034 | 0.0603 | 0.7651 |

Regret-objective RouteCode oracle rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.7069 | 0.2569 | 0.0000 |
| 2 | 1.0000 | 0.8896 | 0.7966 | 0.1672 | 0.3490 |
| 4 | 2.0000 | 1.6603 | 0.8741 | 0.0897 | 0.6510 |
| 8 | 3.0000 | 2.2466 | 0.9276 | 0.0362 | 0.8591 |
| 16 | 4.0000 | 2.9940 | 0.9586 | 0.0052 | 0.9799 |
| 32 | 5.0000 | 3.6481 | 0.9638 | 0.0000 | 1.0000 |
| 64 | 6.0000 | 4.2367 | 0.9638 | 0.0000 | 1.0000 |
| 128 | 7.0000 | 4.8026 | 0.9638 | 0.0000 | 1.0000 |

Predicted RouteCode rows:

| K | rate_log2K | empirical_H_Z | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.7069 | 0.2569 | 0.0000 |
| 2 | 1.0000 | 0.9576 | 0.6983 | 0.2655 | -0.0336 |
| 4 | 2.0000 | 1.9813 | 0.6914 | 0.2724 | -0.0604 |
| 8 | 3.0000 | 2.9323 | 0.6845 | 0.2793 | -0.0872 |
| 16 | 4.0000 | 3.9397 | 0.6569 | 0.3069 | -0.1946 |
| 32 | 5.0000 | 4.6997 | 0.6310 | 0.3328 | -0.2953 |
| 64 | 6.0000 | 5.2475 | 0.5948 | 0.3690 | -0.4362 |
| 128 | 7.0000 | 5.5303 | 0.5759 | 0.3879 | -0.5101 |

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

## Split Sensitivity

Command:

```bash
python experiments/04_split_sensitivity.py --config configs/llmrouterbench_scale20.yaml
```

Outputs:

- `table_split_sensitivity.csv`: method metrics for each split scenario.
- `table_split_rank_correlation.csv`: ranking correlation and degradation against the random split.
- `table_split_rate_threshold.csv`: RouteCode predicted-label rate needed to recover 80% learned-router gain when reached.
- `fig_split_sensitivity.pdf`: heatmap of recovered gap vs oracle across scenarios.

Random split ranking snapshot:

| method | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- |
| query_oracle | 0.9638 | 1.0000 |
| dataset_label_lookup | 0.7586 | 0.2013 |
| predicted_topic_lookup | 0.7431 | 0.1409 |
| embedding_cluster_lookup | 0.7379 | 0.1208 |
| kNN | 0.7241 | 0.0671 |
| best_single | 0.7069 | 0.0000 |
| routecode_predicted_labels | 0.6569 | -0.1946 |
| logistic_embedding_router | 0.5172 | -0.7383 |

Lowest rank correlations vs random:

| scenario | scenario_type | rank_correlation_vs_random | mean_absolute_utility_delta_vs_random | mean_absolute_recovered_gap_delta_vs_random |
| --- | --- | --- | --- | --- |
| leave_dataset_out:aime | leave_one_dataset_out | 0.5367 | 0.0990 | 0.1779 |
| domain_homogeneous:broad_knowledge | domain_homogeneous | 0.8295 | 0.0269 | 0.0961 |
| leave_domain_out:broad_knowledge | leave_one_domain_out | 0.8383 | 0.0561 | 0.0872 |
| cluster_held_out:0 | cluster_held_out | 0.9286 | 0.0185 | 0.1507 |
| model_pool_holdout:DeepSeek-R1-0528-Qwen3-8B | model_pool_holdout | 0.9940 | 0.0067 | 0.0242 |
| model_pool_holdout:DeepHermes-3-Llama-3-8B-Preview | model_pool_holdout | 1.0000 | 0.0067 | 0.0266 |
| random | random | 1.0000 | 0.0000 | 0.0000 |

## Phase H Claim Audit

Command:

```bash
python experiments/34_claim_audit.py --config configs/llmrouterbench_scale20.yaml
```

Outputs:

- `table_claim_status.csv`: claim-level status, metric, threshold, and evidence pointers.
- `phase_h_claim_status_memo.md`: interpretation memo for supported, diagnostic, unsupported, and missing-evidence claims.

| claim_id | claim | status | primary_metric | primary_value | threshold | evidence | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low_rate_oracle_codes | Useful low-rate utility route codes exist. | diagnostic_supported | best_low_rate_oracle_recovered_gap_vs_oracle | 0.9799 | diagnostic if >= 0.80 by K<=16 | table_rate_distortion.csv, method=regret_routecode_oracle_labels, K=16.0000, recovered_gap_vs_oracle=0.9799 | Low-rate oracle code labels preserve most oracle routing gain. |
| small_inferred_labels | Small inferred route labels recover most routing performance. | not_supported | best_inferred_recovered_gap_vs_oracle | 0.1409 | supported only if recovered gap >= 0.85 and lower bootstrap CI >= 0.80 | table_recovered_gap.csv, method=predicted_topic_lookup, K=6.0000, recovered_gap_vs_oracle=0.1409 | Do not claim that small inferred labels recover most routing performance. |
| model_pool_transfer | Route labels transfer across model pools better than same-budget direct retraining. | diagnostic_alive | mean_matched_transfer_minus_direct_recovered_gap | 0.1616 | diagnostic if > 0; paper-level support requires broader split/model coverage | transfer_scenario=complementary_to_remaining_top: routecode=0.3588, direct=0.1059, diff=0.2529; transfer_scenario=dominated_to_remaining_top: routecode=0.2714, direct=0.0286, diff=0.2429; transfer_scenario=top_to_next: routecode=0.0442, direct=0.0552, diff=-0.0110 | Transfer remains alive as a bounded diagnostic. |
| new_model_calibration | New models can be integrated with fewer calibration examples than direct retraining. | diagnostic_alive | mean_matched_routecode_minus_direct_recovered_gap | 0.5096 | diagnostic if > 0; paper-level support requires stronger cost accounting and broader repeats | new_model_id=MiMo-7B-RL-0530, examples_per_label=1: routecode=-0.4295, direct=-0.4094, diff=-0.0201; new_model_id=MiMo-7B-RL-0530, examples_per_label=16: routecode=0.0671, direct=-0.4094, diff=0.4765; new_model_id=MiMo-7B-RL-0530, examples_per_label=2: routecode=0.0201, direct=-0.4094, diff=0.4295; new_model_id=MiMo-7B-RL-0530, examples_per_label=32: routecode=0.0671, direct=-0.3960, diff=0.4631; new_model_id=MiMo-7B-RL-0530, examples_per_label=4: routecode=-0.0201, direct=-0.4094, diff=0.3893; new_model_id=MiMo-7B-RL-0530, examples_per_label=64: routecode=0.0671, direct=-0.3960, diff=0.4631 | Calibration remains alive as a bounded diagnostic. |
| benchmark_diagnosis | Benchmark routing results expose compressibility or split-design artifacts. | not_supported | min_split_rank_correlation | 0.5367 | diagnostic if rank correlation < 0.50 or dataset-label recovered gap >= 0.25 | table_split_rank_correlation.csv: scenario=leave_dataset_out:aime, rank_correlation=0.5367; table_recovered_gap.csv, method=dataset_label_lookup, K=6.0000, recovered_gap_vs_oracle=0.2013 | Current split/compressibility evidence is not strong enough for benchmark diagnosis. |
| adaptive_refinement | Adaptive refinement improves cost-quality by refining uncertain queries. | not_supported | top10_regret_mass_fraction | 0.1910 | defer unless residual-risk gate is strong; current heuristic expects top-10% regret mass >= 0.30 | table_residual_risk.csv, regret_mass_fraction=0.1910 | Do not implement or claim adaptive refinement from the current residual-risk gate. |
