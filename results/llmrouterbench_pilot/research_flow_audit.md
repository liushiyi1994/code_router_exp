# Research Flow Audit

Source: `/home/liush/projects/code_router_exp/Research Flow.md`

This audit records the current state after the LLMRouterBench pilot, corrected RouteCode oracle-label evaluation, E3 predictor diagnostics, E4 code-card artifacts and interpretability audit, D2 predictability-constrained RouteCode, D4/E5 simulated new-model calibration, bounded Phase F ablations, bounded Phase G sensitivities, raw benchmark coverage/taxonomy diagnostics, broad20 B0/B1/B2/D2/residual/split/ablation/transfer/local-external-surrogate/RouteLLM-MF-official-code diagnostics, same-six-dataset 20-model pool scale diagnostics, same-six-dataset held-out model-pool transfer diagnostics, coarse LLMRouterBench domain-map split sensitivity, official RouteLLM artifact inspection, RouteLLM pairwise split-alignment substrate export, RouteLLM MF trainer asset export, split-aligned RouteLLM MF local-code evaluation, split-aligned local Avengers-Pro cluster-routing compatibility evaluation, and cache-only transformer-backbone readiness audit. It is a checkpoint, not a completion claim.

## Verified Complete

### Phase A: Synthetic Sanity

Evidence:

- `results/demo/table_routability.csv`
- `results/demo/table_recovered_gap.csv`
- `results/demo/table_rate_distortion.csv`
- `results/demo/fig_compression_ladder.pdf`
- `results/demo/fig_rate_distortion.pdf`
- `results/demo/code_cards.md`
- `results/demo/code_cards.json`
- `results/demo/fig_code_label_heatmap.pdf`

Status: complete for the local synthetic MVP. Synthetic outputs are for pipeline validation only and should not be used for scientific claims.

### Phase B: Real-Data Pilot

Evidence:

- B0 routability: `results/llmrouterbench_pilot/table_routability.csv`, `fig_model_win_distribution.pdf`, `fig_oracle_gap_by_dataset.pdf`
- B1 compression ladder: `table_recovered_gap.csv`, `table_leakage_gap.csv`, `fig_compression_ladder.pdf`
- B2 rate-distortion: `table_rate_distortion.csv`, `fig_rate_distortion.pdf`
- B3 residual concentration: `table_residual_concentration.csv`, `table_residual_queries.csv`, `table_residual_by_label.csv`, `fig_residual_concentration.pdf`
- B4 split sensitivity: `table_split_sensitivity.csv`, `table_split_rank_correlation.csv`, `table_split_rate_threshold.csv`, `fig_split_sensitivity.pdf`

Status: complete for the bounded LLMRouterBench pilot subset. The loader now applies the configured coarse LLMRouterBench `domain_map` before split construction, so leave-domain-out is no longer just leave-dataset-out. The current map remains coarse and manually configured, so it supports pilot split-sensitivity diagnosis but not a broad domain-generalization claim.

### Phase C: Observation Synthesis

Evidence:

- `results/llmrouterbench_pilot/phase_c_observation_memo.md`

Status: complete for the pilot checkpoint.

Key conclusion:

- Utility-oracle RouteCode labels are strong at K=16: mean utility `0.8897`, recovered gap vs query oracle `0.9699`.
- Current deployable label predictors are weak: best measured utility-oracle label accuracy is `0.2052`, and predicted RouteCode variants remain below best-single.
- The selected next method direction is D2 predictability-constrained RouteCode.

### Phase D3/E4: Explainable Code Cards

Evidence:

- `results/llmrouterbench_pilot/code_cards.md`
- `results/llmrouterbench_pilot/code_cards.json`
- `results/llmrouterbench_pilot/fig_code_label_heatmap.pdf`

Status: complete for the current flat RouteCode codebook. Future D2 labels should regenerate the same artifacts.

### E3: Query-to-Label Prediction Diagnostics

Evidence:

- `results/llmrouterbench_pilot/table_predictor_comparison.csv`
- `results/llmrouterbench_pilot/table_utility_weighted_confusion.csv`
- `results/llmrouterbench_pilot/table_calibration_curve.csv`
- `results/llmrouterbench_pilot/fig_utility_weighted_confusion.pdf`
- `results/llmrouterbench_pilot/fig_calibration_curve.pdf`

Status: complete for diagnosing current RouteCode label predictability.

### Phase D2: Predictability-Constrained RouteCode

Evidence:

- `results/llmrouterbench_pilot/table_predictability_constrained.csv`
- `results/llmrouterbench_pilot/fig_predictability_constrained_tradeoff.pdf`
- `results/llmrouterbench_pilot/code_cards_predictability_constrained.md`
- `results/llmrouterbench_pilot/code_cards_predictability_constrained.json`
- `results/llmrouterbench_pilot/fig_code_label_heatmap_predictability_constrained.pdf`
- `results/llmrouterbench_pilot/phase_d_method_memo.md`

Status: complete for the minimum D2 pilot at fixed K=16 and alpha sweep `[0, 0.05, 0.1, 0.3, 1, 3, 10]`.

Key result:

- Best deployable D2 row is `d2_embedding_centroid` at alpha `3`: mean utility `0.7466`, recovered gap vs query oracle `0.3459`, label accuracy vs D2 joint labels `0.9810`.
- D2 improves substantially over flat RouteCode logistic label prediction (`0.6138` mean utility), and is above kNN/semantic KMeans in this run (`0.7362` mean utility), but remains below dataset-label lookup (`0.7534` mean utility).
- Increasing alpha makes labels predictable but reduces the D2 joint-label oracle from the utility-only RouteCode oracle level; this is the expected utility/predictability tradeoff, not a final claim that inferred labels recover most oracle performance.

### Phase D4/E5: New-Model Calibration

Evidence:

- `results/llmrouterbench_pilot/table_new_model_integration.csv`
- `results/llmrouterbench_pilot/fig_transfer_calibration_curve.pdf`
- `results/llmrouterbench_pilot/phase_e5_new_model_calibration_memo.md`

Status: complete for the expanded simulated held-out-model calibration pilot using existing outcome tables and no external API calls.

Protocol implemented:

- freeze query-to-label predictor;
- treat all six configured pilot models as held-out/new models: `Qwen3-8B`, `Qwen2.5-Coder-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-7B`, `Llama-3.1-8B-Instruct`, `MiniCPM4.1-8B`, and `Intern-S1-mini`;
- sample `r = 1,2,4,8,16,32,64` examples per label;
- update label-to-model table;
- compare against direct router retraining under the same calibration budget using logistic, SVM, kNN, MLP, and gradient-boosting classifiers.

Key result:

- Mean across the six held-out models: RouteCode label calibration reaches mean utility `0.7374` at r=32 with about `411.8` new-model evaluations. The strongest mean direct retraining row among logistic/SVM/kNN/MLP/gradient-boosting is MLP at r=2, mean utility `0.6672`.
- Best individual budgeted row: held-out `Qwen2.5-Coder-7B-Instruct`, RouteCode label calibration at r=64, mean utility `0.7431` with `607` new-model evaluations.
- This supports keeping the sample-efficiency claim alive as a diagnostic. It is not yet a final transfer claim because seeds, official external baselines, transformer direct-router baselines, and broader model-pool robustness remain missing.

### Phase E: Internal Baseline Coverage

Evidence:

- `results/llmrouterbench_pilot/table_recovered_gap.csv`
- `results/llmrouterbench_pilot/table_rate_distortion.csv`

Status: complete for required local/internal baseline rows.

Current internal baselines include:

- random;
- best single;
- dataset oracle;
- query oracle;
- dataset-label lookup;
- predicted-topic lookup;
- embedding-cluster lookup;
- kNN;
- logistic/MLP/SVM embedding routers;
- flat and D2 RouteCode rows;
- simulated new-model calibration against direct logistic/SVM/kNN retraining.

Key added rows from the latest baseline hardening:

- `random`: mean utility `0.5724`, recovered gap vs query oracle `-0.4135`;
- `dataset_oracle`: mean utility `0.7638`, recovered gap vs query oracle `0.4211`;
- `svm_embedding_router`: mean utility `0.6724`, recovered gap vs query oracle `0.0226`.

### Phase E: External-Style Baseline Surrogates

Evidence:

- `results/llmrouterbench_pilot/table_external_baselines.csv`
- `results/llmrouterbench_pilot/phase_e_external_baseline_memo.md`

Status: complete for a local, no-API external-style surrogate layer. This is not complete for official external-method reproduction.

Covered in this layer:

- low-rank utility matrix-factorization router inspired by RouteLLM/EmbedLLM MF;
- RouteLLM-style binary strong/weak threshold router using local embeddings and logistic strong-win prediction;
- strong/weak pair pinned to `Qwen3-8B` vs `Qwen2.5-Coder-7B-Instruct` for the pilot;
- references and implementation mismatch recorded in the memo and README.

Key result:

- `routellm_style_mf_utility_router` reaches mean utility `0.7052` and recovered gap `0.1654`, above best-single but below kNN (`0.7362`, recovered gap `0.3008`).
- The best binary surrogate row is threshold `0.25`: mean utility `0.6931`, recovered gap `0.1128`, and strong-model selection rate `0.3672`.
- These rows help bound simple external-style methods, but they do not replace official RouteLLM-MF/BERT, GraphRouter, or Avengers-Pro runs.

### Phase E: Official RouteLLM Artifact Inspection

Evidence:

- `results/llmrouterbench_pilot/table_official_external_artifacts.csv`
- `results/llmrouterbench_pilot/phase_e_official_baseline_artifacts_memo.md`

Status: complete for parsing official upstream LLMRouterBench RouteLLM-MF artifacts that are already present in the local checkout. This is not complete for split-aligned official external-method reproduction.

Covered in this layer:

- five official RouteLLM-MF seed artifacts from `data/raw/external/LLMRouterBench/baselines/RouteLLM/results`;
- overall and per-dataset upstream rows, 70 rows total;
- enrichment from upstream `mf_selection_accuracy_by_seed.csv` and `mf_total_cost_by_seed.csv`;
- explicit compatibility flags: `split_aligned_with_routecode = False` and `routecode_metric_compatible = False`.

Key result:

- The official upstream RouteLLM-MF aggregate artifacts report sample-average selection accuracy from `0.6029` to `0.6262` and total cost from `124.6618` to `126.6179` across seeds.
- These values document the upstream artifact state and baseline readiness only. They are not directly comparable to RouteCode mean-utility or recovered-gap rows because they use a different evaluation contract.

### Phase E: RouteLLM Pairwise Split-Alignment Substrate

Evidence:

- `results/llmrouterbench_pilot/routellm_pairwise/pairwise_train.json`
- `results/llmrouterbench_pilot/routellm_pairwise/pairwise_test.json`
- `results/llmrouterbench_pilot/routellm_pairwise/metadata.json`
- `results/llmrouterbench_pilot/table_routellm_pairwise_alignment.csv`
- `results/llmrouterbench_pilot/phase_e_routellm_pairwise_alignment_memo.md`

Status: complete for exporting a RouteCode split-aligned strong/weak pairwise substrate for the configured RouteLLM-style binary pair. This is not complete for official RouteLLM-MF/BERT evaluation.

Covered in this layer:

- configured pair `Qwen3-8B` as model_a/strong and `Qwen2.5-Coder-7B-Instruct` as model_b/weak;
- 1,738 train pairwise records and 580 test pairwise records;
- train/test query overlap `0`;
- winner labels `model_a`, `model_b`, or `tie` derived from RouteCode utility on the existing split;
- explicit compatibility flags: `split_aligned_with_routecode = True`, `official_routellm_result = False`, and `routecode_metric_compatible = False`.

Key result:

- The pairwise export closes the data-substrate part of the RouteLLM split-alignment gap without external API calls or checkpoint downloads. It does not train or evaluate official RouteLLM-MF/BERT, so it should not be ranked against RouteCode utility rows as an external baseline result.

### Phase E: RouteLLM MF Trainer Assets

Evidence:

- `results/llmrouterbench_pilot/routellm_mf_assets/pairwise_train.json`
- `results/llmrouterbench_pilot/routellm_mf_assets/pairwise_test.json`
- `results/llmrouterbench_pilot/routellm_mf_assets/prompt_embeddings.npy`
- `results/llmrouterbench_pilot/routellm_mf_assets/prompt_index.json`
- `results/llmrouterbench_pilot/routellm_mf_assets/mf_train_config.local.json`
- `results/llmrouterbench_pilot/routellm_mf_assets/metadata.json`
- `results/llmrouterbench_pilot/table_routellm_mf_assets.csv`
- `results/llmrouterbench_pilot/phase_e_routellm_mf_assets_memo.md`

Status: complete for exporting official-trainer-compatible RouteLLM-MF inputs using the RouteCode split and deterministic local embeddings. This is not complete for a trained or evaluated RouteLLM-MF baseline.

Covered in this layer:

- quality-winner train records in the LLMRouterBench RouteLLM MF trainer schema with utility fields retained for later RouteCode metric conversion;
- 627 decisive train records and 580 test records, with test ties retained;
- 2,318 prompt embeddings with dimension 256 aligned to `idx`;
- local CPU training config at `routellm_mf_assets/mf_train_config.local.json`;
- pair `Qwen3-8B` / `Qwen2.5-Coder-7B-Instruct` verified present in the official RouteLLM `MODEL_IDS`;
- explicit compatibility flags: `split_aligned_with_routecode = True`, `official_trainer_compatible = True`, `official_routellm_result = False`, and `routecode_metric_compatible = False`.

Key result:

- The MF asset export closes the official-trainer input contract gap without external API calls. The follow-on split-aligned MF local-code evaluation is recorded in the next section.

### Phase E: Split-Aligned RouteLLM MF Local-Code Evaluation

Evidence:

- `results/llmrouterbench_pilot/routellm_mf_split_aligned/mf_model.pt`
- `results/llmrouterbench_pilot/routellm_mf_split_aligned/raw_metrics.json`
- `results/llmrouterbench_pilot/routellm_mf_split_aligned/train_config.json`
- `results/llmrouterbench_pilot/table_routellm_mf_split_aligned.csv`
- `results/llmrouterbench_pilot/phase_e_routellm_mf_split_aligned_memo.md`

Status: complete for a split-aligned RouteLLM-MF evaluation using the local LLMRouterBench MF training source and deterministic RouteCode embeddings. This is not the upstream published RouteLLM checkpoint and does not cover RouteLLM-BERT.

Covered in this layer:

- local LLMRouterBench `MFModel_Train` source loaded directly from `data/raw/external/LLMRouterBench/baselines/RouteLLM/routers/matrix_factorization`;
- checkpoint trained on the RouteCode split-aligned pairwise train assets;
- threshold sweep over `[0.25, 0.5, 0.75]`;
- RouteCode utility metrics plus RouteLLM-style quality/routing metrics on the RouteCode test split;
- explicit compatibility flags: `official_training_code_used = True`, `official_upstream_checkpoint = False`, `split_aligned_with_routecode = True`, and `routecode_metric_compatible = True`.

Key result:

- Best split-aligned RouteLLM-MF local-code threshold is `0.5`: mean utility `0.7259`, recovered gap vs oracle `0.2556`, selection accuracy `0.7259`, and decisive-pair routing accuracy `0.7476`.
- This is now a stronger Phase E baseline than the earlier surrogate-only layer, but it remains a local-embedding reproduction rather than an exact upstream RouteLLM command/checkpoint run.

### Phase G: Bounded Sensitivity Suite

Evidence:

- `results/llmrouterbench_pilot/table_sensitivity_summary.csv`
- `results/llmrouterbench_pilot/fig_sensitivity_summary.pdf`
- `results/llmrouterbench_pilot/phase_g_sensitivity_memo.md`

Status: complete for the bounded/local Phase G layer.

Covered in this layer:

- embedding feature variant: configured hashing features vs 64- and 256-feature local hashing embeddings;
- clustering algorithm: KMeans vs agglomerative embedding clusters;
- label noise: logistic embedding-router training labels corrupted at rates `[0.0, 0.1, 0.2]`;
- cost mis-estimation: train-time cost multipliers `[0.5, 1.0, 2.0]` under a sensitivity-local cost objective (`cost_lambda = 100.0`);
- model price-ratio objective stress: cost-ratio exponents `[0.0, 0.5, 1.0, 2.0]` that flatten or expand model-average costs before recomputing utility;
- query-length buckets: short, medium, and long query subsets with bucket-local references;
- domain-granularity buckets: configured coarse domains, curated `task_family` and `task_subtype` taxonomy groups, dataset-level groups, and train-fitted text-cluster groups with bucket-local references;
- dominated/composition checks: full model pool, top-4 model pool, drop-dominant pool, configured `qwen_pair`, `qwen_deepseek_llama`, and `compact_pair` model-pool slices, plus automatic dominated/complementary pools for sizes `[2, 3, 4, 5]`;
- bootstrap sampling: confidence intervals recomputed with `[50, 100, 300]` bootstrap replicates.
- coarse domain granularity is covered through B4 split sensitivity using configured domains: `math`, `code`, `science`, and `broad_knowledge`.

Key result:

- D2 embedding-centroid routing is stable across these bounded sensitivities relative to kNN, but sensitivity remains shallow. In the LLMRouterBench pilot summary, D2 mean recovered gap is `0.3233` across embedding feature variants, `0.3459` in the D2 clustering reference, `0.4305` under the cost-misestimation slice, `0.3699` across price-ratio objective stress rows, `0.3682` across bounded model-pool subset scenarios, `0.2246` across configured model-pool composition scenarios, `0.2839` across automatic dominated/complementary model pools, `0.1668` across domain-granularity buckets, `0.3163` across query-length buckets, and `0.3459` across bootstrap-count variants.
- Price-ratio objective stress keeps D2 above best-single across exponents `[0.0, 0.5, 1.0, 2.0]`, with recovered gap from `0.3167` to `0.4203`. kNN is slightly higher on this slice with mean recovered gap `0.3959`, so this is robustness evidence rather than a D2 dominance claim.
- Automatic model-pool construction exposes the intended contrast: D2 recovered gap ranges from `-0.0200` on dominated automatic pools to `0.6796` on complementary automatic pools, so model-pool composition strongly changes apparent routing value.
- Domain-granularity buckets expose strong heterogeneity: D2 recovered gap ranges from `-0.2000` to `0.7600`, with mean `0.1668`. The regenerated rows now include curated task-family/task-subtype taxonomy buckets in addition to coarse domains, datasets, and train-fitted text clusters. This is still a bounded pilot taxonomy, not a full benchmark-scale taxonomy proof.
- Query-length buckets expose heterogeneity: D2 recovered gap ranges from `-0.0476` to `0.6140`, so broad claims still need stratified reporting.

### Phase F/G: 20-Model Pool Scale Robustness

Evidence:

- `results/llmrouterbench_scale20/table_model_pool_scale.csv`
- `results/llmrouterbench_scale20/phase_f_g_model_pool_scale_memo.md`
- `results/llmrouterbench_scale20/README.md`

Status: complete for a same-six-dataset, 20-model LLMRouterBench scale layer. This extends model-pool size/composition diagnostics, but it is not a held-out model-pool transfer claim.

Covered in this layer:

- six pilot datasets with complete shared coverage across 20 local LLMRouterBench model result files;
- 57,940 query-model rows, 2,897 queries, 20 models, and six datasets;
- train-only construction of top, complementary, dominated, and full pool scenarios with sizes `[2, 4, 8, 12, 16, 20]`;
- best-single, kNN, and D2 embedding-centroid rows evaluated on the held-out test queries for each pool.

Key result:

- On the full 20-model pool, D2 reaches mean utility `0.7397` and recovered gap vs oracle `0.1275`; kNN reaches mean utility `0.7241` and recovered gap `0.0671`.
- Pool composition strongly changes routability: D2 recovered gap ranges from `-0.1154` on `top_2` to `0.3913` on `complementary_2`, and dominated pools can be neutral or negative.
- These rows keep model-pool robustness alive as a diagnostic, but broader dataset coverage, held-out pool protocols, and direct-router retraining comparisons remain necessary before making transfer claims.

### Phase F/G: Held-Out Model-Pool Transfer

Evidence:

- `results/llmrouterbench_scale20/table_model_pool_transfer.csv`
- `results/llmrouterbench_scale20/phase_f_g_model_pool_transfer_memo.md`
- `results/llmrouterbench_scale20/README.md`

Status: complete for a same-six-dataset, disjoint source/target 20-model transfer diagnostic. This is bounded transfer evidence, not a paper-level transfer claim.

Covered in this layer:

- three disjoint source/target protocols: `top_to_next`, `complementary_to_remaining_top`, and `dominated_to_remaining_top`;
- 8 source models and 8 target models per scenario, with maximum source-target overlap `0`;
- source D2 labels learned on source-pool train utility, then remapped to target-pool models using target train utility only;
- same target train labels used for direct retraining baselines: logistic, SVM, kNN, MLP, and gradient boosting;
- target best-single, target kNN, target native D2, target direct retraining, and transferred source-D2 rows.

Key result:

- Transferred source-D2 recovered gap ranges from `0.0442` to `0.3588` across the three disjoint 8-to-8 scenarios, with mean `0.2248`.
- Direct retraining recovered gap ranges from `-0.0286` to `0.1059` across logistic/SVM/kNN/MLP/gradient-boosting rows, with the best direct row coming from MLP on `complementary_to_remaining_top`.
- Native target-D2 recovered gap ranges from `0.0884` to `0.3588`, so transferred labels are competitive in two scenarios and weaker on `top_to_next`.
- This keeps the model-pool-transfer claim alive as a diagnostic, but the protocol is still same-six-dataset and train-label-rich; broader dataset coverage and stricter budget sweeps are still needed.

### Phase G: Benchmark Coverage And Taxonomy Audit

Evidence:

- `results/llmrouterbench_pilot/table_benchmark_file_coverage.csv`
- `results/llmrouterbench_pilot/table_benchmark_dataset_coverage.csv`
- `results/llmrouterbench_pilot/table_broad_coverage_candidates.csv`
- `results/llmrouterbench_pilot/phase_g_benchmark_coverage_memo.md`

Status: complete for raw local LLMRouterBench coverage discovery and configured taxonomy coverage. This does not run routers on the broader rectangle.

Covered in this layer:

- raw result scan before canonical equal-model schema validation;
- latest-file filtering across local LLMRouterBench JSON results;
- dataset-level domain/task taxonomy coverage;
- complete model/dataset rectangle candidates for model counts `[6, 10, 20, 32]`.

Key result:

- Local raw coverage contains `567` latest result files, `24` datasets, and `40` models.
- The broad config now provides domain and `task_family`/`task_subtype` labels for all `24` local datasets surfaced by the coverage audit.
- The largest complete-query candidate by query count uses `20` models over `18` datasets with `14,041` complete queries and `280,820` query-model rows.
- A `32`-model candidate exists, but only over `5` datasets and `2,435` complete queries. This exposes the tradeoff between model-pool size and dataset breadth.
- Next evidence should run actual router/rate-distortion metrics on the 18-dataset/20-model candidate; coverage alone is not routing evidence.

### Phase B: Broad20 Router Metrics

Evidence:

- `results/llmrouterbench_broad20/table_routability.csv`
- `results/llmrouterbench_broad20/table_recovered_gap.csv`
- `results/llmrouterbench_broad20/table_rate_distortion.csv`
- `results/llmrouterbench_broad20/fig_compression_ladder.pdf`
- `results/llmrouterbench_broad20/fig_rate_distortion.pdf`
- `results/llmrouterbench_broad20/phase_b_broad20_memo.md`
- `results/llmrouterbench_broad20/table_predictability_constrained.csv`
- `results/llmrouterbench_broad20/fig_predictability_constrained_tradeoff.pdf`
- `results/llmrouterbench_broad20/phase_d_method_memo.md`
- `results/llmrouterbench_broad20/table_residual_concentration.csv`
- `results/llmrouterbench_broad20/table_residual_risk.csv`
- `results/llmrouterbench_broad20/fig_residual_concentration.pdf`
- `results/llmrouterbench_broad20/fig_risk_coverage.pdf`
- `results/llmrouterbench_broad20/phase_d5_adaptive_refinement_gate_memo.md`
- `results/llmrouterbench_broad20/table_split_sensitivity.csv`
- `results/llmrouterbench_broad20/table_split_rank_correlation.csv`
- `results/llmrouterbench_broad20/table_split_rate_threshold.csv`
- `results/llmrouterbench_broad20/fig_split_sensitivity.pdf`
- `results/llmrouterbench_broad20/table_ablation_summary.csv`
- `results/llmrouterbench_broad20/fig_sensitivity_k_lambda.pdf`
- `results/llmrouterbench_broad20/fig_seed_stability.pdf`
- `results/llmrouterbench_broad20/phase_f_g_ablation_memo.md`
- `results/llmrouterbench_broad20/table_model_pool_transfer.csv`
- `results/llmrouterbench_broad20/phase_f_g_model_pool_transfer_memo.md`
- `results/llmrouterbench_broad20/table_external_baselines.csv`
- `results/llmrouterbench_broad20/phase_e_external_baseline_memo.md`
- `results/llmrouterbench_broad20/table_routellm_mf_split_aligned.csv`
- `results/llmrouterbench_broad20/phase_e_routellm_mf_split_aligned_memo.md`
- `results/llmrouterbench_broad20/routellm_mf_split_aligned/mf_model.pt`
- `results/llmrouterbench_broad20/table_avengerspro_split_aligned.csv`
- `results/llmrouterbench_broad20/phase_e_avengerspro_split_aligned_memo.md`
- `results/llmrouterbench_broad20/avengerspro_split_aligned/metadata.json`

Status: complete for B0/B1/B2, D2 predictability-constrained RouteCode, residual/adaptive-refinement gate diagnostics, bounded split sensitivity, bounded ablations, bounded held-out model-pool transfer, local external-style baseline surrogates, RouteLLM MF official-code local-embedding evaluation, and a local Avengers-Pro compatibility baseline on the audited 18-dataset/20-model complete rectangle. Published upstream checkpoints/full upstream-command external baselines have not yet been rerun on this rectangle; the broad split/ablation/transfer/external rows are bounded local diagnostics, not exhaustive sweeps.

Covered in this layer:

- canonical broad20 data with `280,820` query-model rows, `14,041` queries, `18` datasets, and `20` models;
- B0 routability;
- B1 compression ladder;
- B2 rate-distortion over K `[1, 2, 4, 8, 16, 32, 64, 128]`;
- D2 predictability-constrained RouteCode alpha sweep at K=16;
- residual concentration and deployable residual-risk gate diagnostics;
- bounded split sensitivity covering one representative scenario each for random, leave-one-dataset-out, leave-one-domain-out, domain-homogeneous, cluster-held-out, and model-pool-holdout;
- bounded ablations over K `[8, 16, 32]`, lambda `[0.0, 0.1]`, seeds `[3, 7]`, train fractions `[0.5, 1.0]`, and D2 beta `[0.0, 1.0]`;
- bounded disjoint 8-source/8-target model-pool transfer over top, complementary, and dominated source-pool scenarios;
- local RouteLLM/LLMRouter-style external baseline surrogates with explicit non-official implementation notes;
- local LLMRouterBench RouteLLM MF training code on broad20 split-aligned pairwise assets with local RouteCode embeddings.
- local Avengers-Pro cluster-routing contract implementation on broad20 with RouteCode deterministic embeddings and explicit non-official implementation notes.

Key result:

- Broad20 query oracle is strong: best-single mean utility is `0.7037`, query oracle is `0.9160`, and oracle regret is `0.2123`.
- Dataset-label lookup is weak on this broad split: mean utility `0.7172`, recovered gap `0.0638`.
- Embedding-cluster lookup reaches mean utility `0.7222` and recovered gap `0.0872`; kNN is slightly below best-single with recovered gap `-0.0067`.
- Utility-vector RouteCode oracle labels recover `0.3658` at K=16, `0.5151` at K=32, `0.7064` at K=64, and `0.7903` at K=128.
- Regret-objective RouteCode oracle labels are much stronger: recovered gap is `0.6510` at K=4, `0.8540` at K=8, `0.9681` at K=16, and `1.0000` by K=32.
- Current predicted RouteCode labels remain below best-single on this broad rectangle: logistic predicted K=16 has recovered gap `-0.4782`, and MLP predicted K=16 has `-0.4581`.
- Best deployable D2 at alpha `1.0` reaches mean utility `0.7229`, recovered gap `0.0906`, and label accuracy `0.7646`. This is a small gain over best-single and close to embedding KMeans recovered gap `0.0872`, not a large broad-rectangle recovery result.
- Broad residual regret concentration remains visible by oracle sorting: top 5%, 10%, and 20% high-regret queries account for `0.1600`, `0.3190`, and `0.6379` of residual regret. Deployable risk signals are weak: low route-label confidence captures only `0.0556` and `0.1056` of regret at top 5% and 10% flagged queries, with AUC `0.5370`; adaptive refinement remains deferred.
- Bounded split sensitivity shows benchmark-design effects: `leave_dataset_out:aime` rank correlation vs random is `0.7832` and `cluster_held_out:0` is `0.7904`; domain/model-pool variants are more stable in this bounded run.
- Bounded broad ablations preserve the main pattern: regret-objective oracle RouteCode has mean recovered gap `0.9360` across the configured K/lambda rows, flat utility-oracle RouteCode averages `0.3155`, semantic KMeans averages `0.0792`, and deployable D2 averages `0.0696`; D2 seed-stability mean recovered gap is `0.0990`.
- Bounded broad transfer keeps the model-pool transfer claim alive diagnostically: transferred source-D2 labels recover `0.1739` to `0.2274` of the target oracle gap across three disjoint 8-to-8 scenarios, native target-D2 recovers `0.1682` to `0.2592`, and lightweight direct retraining baselines recover only `0.0013` to `0.0394`.
- Broad local external-style surrogates are weak: the MF utility surrogate reaches mean utility `0.6934` and recovered gap `-0.0487`, while binary RouteLLM-style strong/weak surrogates range from recovered gap `-0.1560` to `-1.5034`.
- Broad RouteLLM MF official-code local-embedding rows are only slightly above best-single: validation accuracy is `0.9106`, but the best threshold `0.5` reaches mean utility `0.7073` and recovered gap `0.0168`.
- Broad local Avengers-Pro compatibility rows are below best-single: the simple K=16 cluster row reaches mean utility `0.6574` and recovered gap `-0.2181`, while the balance row reaches mean utility `0.6449` and recovered gap `-0.2768`.

## Incomplete

### Phase D5/E7: Adaptive Refinement

Optional/follow-up unless residual concentration and confidence/disagreement predict failures.

Gate evidence:

- `fig_risk_coverage.pdf`
- `results/llmrouterbench_pilot/table_residual_risk.csv`
- `results/llmrouterbench_pilot/phase_d5_adaptive_refinement_gate_memo.md`

Status: gate checked; adaptive refinement is deferred.

Key result:

- Residual regret remains moderately concentrated by oracle sorting: top 5%, 10%, and 20% high-regret queries account for `17.7%`, `35.4%`, and `70.7%` of total residual regret.
- Deployable risk signals are weak in this pilot. The best deployable trigger captures only `5.49%` of regret in the top 5% flagged queries and `12.20%` in the top 10% flagged queries; best deployable AUC is approximately `0.56`.
- Adaptive refinement should not be implemented as a core RouteCode claim from the current pilot. The missing adaptive-refinement utility outputs (`table_adaptive_refinement.csv`, `fig_refinement_utility_vs_cost.pdf`) are intentionally deferred unless a stronger deployable risk signal is found.

### Phase E: External Method Evaluation

Internal baselines, local external-style surrogates, official upstream RouteLLM artifact inspection, RouteLLM pairwise split-alignment substrate export, RouteLLM MF trainer asset export, split-aligned RouteLLM MF local-code evaluation, and a split-aligned local Avengers-Pro cluster-routing compatibility evaluation are now covered, but the full required split-aligned official external baseline set is still incomplete.

Baseline readiness evidence:

- `results/llmrouterbench_pilot/baseline_readiness_audit.md`
- `results/llmrouterbench_pilot/table_external_baselines.csv`
- `results/llmrouterbench_pilot/phase_e_external_baseline_memo.md`
- `results/llmrouterbench_pilot/table_official_external_artifacts.csv`
- `results/llmrouterbench_pilot/phase_e_official_baseline_artifacts_memo.md`
- `results/llmrouterbench_pilot/table_routellm_pairwise_alignment.csv`
- `results/llmrouterbench_pilot/phase_e_routellm_pairwise_alignment_memo.md`
- `results/llmrouterbench_pilot/table_routellm_mf_assets.csv`
- `results/llmrouterbench_pilot/phase_e_routellm_mf_assets_memo.md`
- `results/llmrouterbench_pilot/table_routellm_mf_split_aligned.csv`
- `results/llmrouterbench_pilot/phase_e_routellm_mf_split_aligned_memo.md`
- `results/llmrouterbench_pilot/table_avengerspro_split_aligned.csv`
- `results/llmrouterbench_pilot/phase_e_avengerspro_split_aligned_memo.md`

Still missing or not integrated as external/stronger baselines:

- exact upstream-command RouteLLM-MF/BERT or LLMRouterBench RouteLLM adapter output on the RouteCode pilot split; a split-aligned MF model has been trained/evaluated with local official MF source and RouteCode embeddings, but not with the full upstream environment/checkpoint path and not for BERT;
- official GraphRouter / LLMRouter adapter output where feasible;
- exact upstream-command Avengers-Pro output with its service/cache embedding path if that environment can be pinned locally; the current Avengers-Pro row is a split-aligned local implementation of the documented cluster-routing contract.

### Phase F/G: Bounded Ablation And Robustness

Evidence:

- `results/llmrouterbench_pilot/table_ablation_summary.csv`
- `results/llmrouterbench_pilot/fig_sensitivity_k_lambda.pdf`
- `results/llmrouterbench_pilot/fig_seed_stability.pdf`
- `results/llmrouterbench_pilot/phase_f_g_ablation_memo.md`
- `results/llmrouterbench_pilot/table_code_card_interpretability.csv`
- `results/llmrouterbench_pilot/phase_f_code_card_interpretability_memo.md`
- `results/llmrouterbench_pilot/table_transformer_backbone_readiness.csv`
- `results/llmrouterbench_pilot/phase_f_g_transformer_backbone_readiness_memo.md`

Status: complete for the first bounded Phase F/G ablation layer.

Covered in this layer:

- code count K over `[4, 8, 16, 32, 64, 128]`;
- code objective comparison among semantic clusters, utility-vector RouteCode, regret-objective RouteCode, and predictability-constrained D2;
- cost weight lambda over `[0.0, 0.05, 0.1, 0.2]` for the pilot;
- D2 rate-penalty beta over `[0.0, 0.1, 1.0, 3.0]`;
- random seed stability over seeds `[3, 7, 11]`;
- training data fraction for logistic and SVM embedding routers over `[0.25, 0.5, 1.0]`;
- new-model calibration examples per label from the E5 pilot.
- with/without code-card interpretability as an observability ablation for flat RouteCode and D2 RouteCode.
- cache-only transformer-backbone readiness for requested `answerdotai/ModernBERT-base` and `microsoft/deberta-v3-base` checkpoints.

Key result:

- Seed stability: D2 embedding-centroid mean recovered gap is `0.3320` with std `0.0177`; kNN mean recovered gap is `0.2723` with std `0.0356`.
- K/lambda: flat utility-oracle RouteCode reaches the query oracle on this pilot split at K=32 and remains effectively saturated at K=64 and K=128 across tested lambdas; D2 remains the stronger deployable RouteCode variant but does not close the oracle-code gap, and its K/lambda mean recovered gap declines for K above 16 in this pilot.
- Regret-objective RouteCode: oracle-label regret-code rows reach the query oracle by K=8 on the pilot rate-distortion curve; deployable embedding-centroid regret-code rows remain much lower, with K=16 recovered gap `0.1053`.
- Rate penalty: D2 beta values `[0.0, 0.1, 1.0, 3.0]` produce the same deployable recovered gap (`0.3459`) and empirical label entropy (`3.1120`) on this pilot slice, so the current balance penalty is not an active lever at the selected K/alpha.
- Code-card interpretability: label-only rows expose `1` audited explanatory field, while code-card rows expose `9` fields for both flat RouteCode and D2. Code-card coverage is `1.0000` for domain summaries, dataset summaries, representative queries, high-regret examples, utility vectors, and human-readable explanations. This is an observability result, not a routing-utility result.
- Transformer-backbone readiness: no requested lightweight encoder checkpoint is currently cached. The local HF cache contains `Qwen/Qwen3-4B`, but it is a causal LM checkpoint of about `7.5` GB and is not used as a lightweight encoder baseline in the no-download pilot.

### Remaining Phase F: Ablation Study

Required but incomplete:

- predictor type beyond lightweight logistic/SVM/MLP remains incomplete as a routing metric; the ModernBERT/DeBERTa local-cache dependency has been audited, but no encoder checkpoint is currently available for evaluation;
- broader held-out model-pool transfer protocols beyond the bounded same-six-dataset and broad20 20-model transfer layers;
- adaptive-refinement ablation only if a future residual-risk gate justifies implementing refinement;

### Remaining Phase G: Sensitivity Analysis

Partially complete for split strategy and coarse domain granularity through B4, curated task-family/task-subtype taxonomy through the local sensitivity suite, random seeds through the bounded ablation layer, and the local sensitivity suite above. Broader required sensitivity remains incomplete.

Still missing:

- published upstream-checkpoint RouteLLM/BERT, GraphRouter/LLMRouter, and exact upstream-command Avengers-Pro rows on the 18-dataset/20-model broad rectangle;
- exhaustive rather than bounded split-sensitivity, ablation, and transfer sweeps on the 18-dataset/20-model broad rectangle;
- true external/pretrained embedding-backbone metric rows rather than local hashing variants; the current readiness audit shows the requested encoder checkpoints are absent from the local cache;
- broader real pricing schedules and cost-quality operating points beyond the current multiplier/exponent stress tests;
- larger full-benchmark model-pool robustness beyond the same-six-dataset 20-model scale layer and raw coverage audit.

### Phase H: Final Paper Claims

Not complete. Current evidence supports only cautious diagnostic claims.

Currently alive:

- useful low-rate utility codes exist;
- predictability-constrained labels are much easier to infer than utility-only labels in the pilot;
- simulated held-out-model calibration favors RouteCode over matched-budget direct logistic/SVM/kNN/MLP/gradient-boosting baselines across six held-out models;
- local external-style MF and binary RouteLLM-style surrogates are below kNN/D2 in the pilot;
- official upstream RouteLLM-MF aggregate artifacts have been parsed, but are not split-aligned RouteCode metrics;
- RouteLLM pairwise train/test substrate and MF trainer assets are now split-aligned with RouteCode for one configured strong/weak pair;
- RouteLLM-MF local-code checkpoint and RouteCode utility rows now exist for that pair, while exact upstream-command MF/BERT coverage remains missing;
- split-aligned RouteLLM-MF local-code evaluation is now available with local RouteCode embeddings; best threshold `0.5` reaches mean utility `0.7259` and recovered gap vs oracle `0.2556`;
- split-aligned local Avengers-Pro compatibility rows now exist; the simple K=16 cluster row reaches mean utility `0.7397` and recovered gap vs oracle `0.3158`, while the balance row reaches mean utility `0.7241` and recovered gap vs oracle `0.2481`;
- same-six-dataset 20-model scale rows now exist; on the full 20-model pool D2 reaches mean utility `0.7397` and recovered gap vs oracle `0.1275`, while complementary/top/dominated pools expose strong composition sensitivity;
- same-six-dataset held-out model-pool transfer rows now exist; transferred source-D2 labels have mean recovered gap `0.2248` across three disjoint 8-to-8 scenarios and remain competitive with same-budget direct retraining in this bounded diagnostic;
- raw LLMRouterBench coverage is now audited: `24` local datasets and `40` local models are present, taxonomy covers all `24` local datasets, and an 18-dataset/20-model complete rectangle is available for the next broad router run;
- broad20 B0/B1/B2, D2, residual-gate, bounded split-sensitivity, bounded ablation, bounded transfer, local external-surrogate, RouteLLM MF official-code, and local Avengers-Pro compatibility metrics now exist: utility-vector RouteCode oracle labels recover `0.7903` by K=128, regret-objective labels recover `0.9681` by K=16, best deployable D2 recovered gap is `0.0906`, broad residual risk signals remain weak, bounded split diagnostics show ranking changes for leave-dataset/cluster-held-out scenarios, transferred source-D2 labels recover `0.1739` to `0.2274` across three disjoint broad20 8-to-8 model-pool scenarios, local external-style surrogates remain below best-single, split-aligned RouteLLM MF official-code rows recover only `0.0168` at the best threshold, and broad local Avengers-Pro rows remain below best-single;
- transformer-backbone readiness is now reproducibly audited with no downloads, but no ModernBERT/DeBERTa metric row exists yet;
- benchmark split design changes rankings;
- dataset/topic labels are strong enough to motivate benchmark diagnosis.

Not yet supported:

- small inferred route labels recover most routing performance;
- paper-level route labels transfer across model pools beyond the bounded same-six-dataset and broad20 transfer diagnostics;
- new models need fewer calibration examples as a paper-level claim;
- adaptive refinement improves cost-quality.

## Next Action

Run the remaining split-aligned official external/stronger baseline and deeper robustness layer: exact upstream-command RouteLLM-MF/BERT if the full dependency/embedding environment is installed, exact upstream-command Avengers-Pro if its embedding service/cache path can be pinned, GraphRouter/LLMRouter adapter outputs after PyG and graph-data dependencies are available, local-files-only ModernBERT/DeBERTa embedding/direct-router rows after a suitable encoder checkpoint is cached, stricter held-out model-pool transfer checks on larger benchmark-scale model pools, and expanded broad20 split/ablation/transfer sweeps if runtime permits. Keep adaptive refinement deferred unless a future deployable residual-risk signal is substantially stronger. The D4/E5 and bounded F/G pilots keep the calibration, transfer, and robustness claims alive but do not yet prove them.
