# Research Flow Audit

Source: `/home/liush/projects/code_router_exp/Research Flow.md`

This audit records the current state after the LLMRouterBench pilot, corrected RouteCode oracle-label evaluation, E3 predictor diagnostics, E4 code-card artifacts and interpretability audit, D2 predictability-constrained RouteCode, D4/E5 simulated new-model calibration, bounded Phase F ablations, bounded Phase G sensitivities, raw benchmark coverage/taxonomy diagnostics, broad20 B0/B1/B2/D2/residual/split/expanded-ablation/transfer/local-external-surrogate/RouteLLM-MF-official-code diagnostics, same-six-dataset 20-model pool scale diagnostics, same-six-dataset held-out model-pool transfer diagnostics, coarse LLMRouterBench domain-map split sensitivity, official RouteLLM artifact inspection, RouteLLM pairwise split-alignment substrate export, RouteLLM MF trainer asset export, split-aligned RouteLLM MF local-code evaluation, split-aligned local Avengers-Pro cluster-routing compatibility evaluation, full-split cache-backed exact upstream Avengers-Pro CLI accuracy/cost metrics, upstream Avengers-Pro model-code RouteCode utility metrics, external baseline split-aligned input asset export, local LLMRouter trainer-class adapter metrics, split-aligned local EmbedLLM KNN and FrugalGPT metric-adapter execution, exact upstream EmbedLLM KNN full-split correctness metrics, exact upstream EmbedLLM MF full-split router-accuracy metrics, RouteCode-postprocessed exact FrugalGPT CLI saved-scorer metrics, cache-only transformer-backbone readiness audit, local-files-only transformer direct-router execution wiring expanded to MiniLM, BGE-small, E5-small, ModernBERT, and DeBERTa, per-run Phase H claim-status gating, and global cross-run Phase H claim aggregation. It is a checkpoint, not a completion claim.

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
- `results/llmrouterbench_broad20/phase_c_broad20_observation_memo.md`

Status: complete for the pilot checkpoint and refreshed for the broad20/32-model evidence now driving claim decisions.

Key conclusion:

- Utility-oracle RouteCode labels are strong at K=16: mean utility `0.8897`, recovered gap vs query oracle `0.9699`.
- Current deployable label predictors are weak: best measured utility-oracle label accuracy is `0.2052`, and predicted RouteCode variants remain below best-single.
- The selected next method direction is D2 predictability-constrained RouteCode.
- Broad20 preserves the low-rate oracle-code result but weakens the deployable claim: regret-objective oracle labels recover `96.8%` by K=16, while the best deployable D2 embedding-centroid row recovers only `9.1%` of oracle gap and is only slightly above semantic embedding k-means.
- The 32-model stress rectangle is mixed-to-negative for deployable transfer and scale because the model pool is highly dominated; it supports model-pool composition as a key diagnostic rather than a general transfer claim.

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
- `results/llmrouterbench_pilot/table_stronger_direct_router_probe.csv`
- `results/llmrouterbench_pilot/phase_e_stronger_direct_router_probe_memo.md`

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
- The bounded stronger direct-router probe over held-out `Qwen2.5-Coder-7B-Instruct` and `Qwen3-8B` keeps the same pattern: RouteCode label calibration averages mean utility `0.7371` and recovered gap `0.3045` at r=64, while the best direct-router average is MLP at r=8 with mean utility `0.6422` and recovered gap `-0.1090`.
- This supports keeping the sample-efficiency claim alive as a diagnostic. It is not yet a final transfer claim because seeds, official external baselines, broader model-pool robustness, and stronger direct-router comparisons beyond the now-executed requested encoder rows remain incomplete.

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

### Phase F/G: 32-Model Pool Scale Stress Test

Evidence:

- `configs/llmrouterbench_32model.yaml`
- `results/llmrouterbench_32model/table_routability.csv`
- `results/llmrouterbench_32model/table_model_pool_scale.csv`
- `results/llmrouterbench_32model/table_model_pool_transfer.csv`
- `results/llmrouterbench_32model/phase_f_g_model_pool_scale_memo.md`
- `results/llmrouterbench_32model/phase_f_g_model_pool_transfer_memo.md`
- `results/llmrouterbench_32model/README.md`

Status: complete for a narrow 32-model/5-dataset complete-rectangle scale and held-out transfer stress test. This is a bounded robustness diagnostic, not a transfer claim.

Covered in this layer:

- audited complete rectangle with `77,920` query-model rows, `2,435` queries, `5` datasets, and `32` models;
- datasets `aime`, `gpqa`, `livecodebench`, `livemathbench`, and `mmlupro`;
- best-single, dataset-label, and query-oracle routability audit;
- train-only top, complementary, dominated, and full model-pool scenarios over sizes `[2, 4, 8, 16, 32]`;
- best-single, kNN, and D2 embedding-centroid rows for each pool scenario.
- disjoint 16-source/16-target transfer scenarios `top_to_next`, `complementary_to_remaining_top`, and `dominated_to_remaining_top`, with source-target overlap `0`;
- target best-single, target kNN, native target-D2, direct logistic/SVM/kNN retraining, and transferred source-D2 rows.

Key result:

- The 32-model rectangle has oracle headroom: best-single mean utility is `0.8501`, query oracle is `0.9384`, and oracle regret is `0.0883`.
- It is also highly dominated on test: the full 32-model pool has test dominance ratio `0.8172`.
- On the full 32-model pool, D2 reaches mean utility `0.8152` and recovered gap `-0.3953`; kNN reaches mean utility `0.8111` and recovered gap `-0.4419`.
- Across the 32-model scale rows, D2 recovered gap ranges from `-0.5294` to `0.0000`. This is negative robustness evidence for deployable compressed routing under this narrow high-model-count rectangle and reinforces model-pool composition as a major diagnostic factor.
- In disjoint 16-source/16-target transfer, transferred source-D2 recovered gap ranges from `-0.1429` to `0.0462`; same-budget direct logistic/SVM/kNN retraining ranges from `-0.2245` to `0.0462`.
- This high-model-count transfer layer is mixed-to-negative: transferred labels match the best lightweight direct row only in the dominated-source scenario and remain below best-single in the top-to-next and complementary scenarios.

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

### Phase F/G: Broad10 Fixed-Rectangle Robustness

Evidence:

- `configs/llmrouterbench_broad10.yaml`
- `results/llmrouterbench_broad10/table_routability.csv`
- `results/llmrouterbench_broad10/table_recovered_gap.csv`
- `results/llmrouterbench_broad10/table_rate_distortion.csv`
- `results/llmrouterbench_broad10/table_predictability_constrained.csv`
- `results/llmrouterbench_broad10/table_model_pool_scale.csv`
- `results/llmrouterbench_broad10/table_model_pool_transfer.csv`
- `results/llmrouterbench_broad10/phase_f_g_broad10_robustness_memo.md`
- `results/llmrouterbench_broad10/README.md`

Status: complete for a fixed 10-model/18-dataset complete-rectangle robustness checkpoint. This is bounded robustness evidence, not a paper-level transfer claim.

Covered in this layer:

- audited complete rectangle with `140,410` query-model rows, `14,041` queries, `18` datasets, and `10` models;
- B0/B1/B2-style routability, compression ladder, and rate-distortion rows;
- D2 predictability-constrained RouteCode alpha sweep over `[0.0, 0.1, 1.0, 3.0]`;
- train-only top, complementary, dominated, and full model-pool scale scenarios over sizes `[2, 4, 8, 10]`;
- six disjoint held-out model-pool transfer scenarios over source/target sizes `4x4` and `6x4`, with source-target overlap `0`;
- same-budget direct retraining baselines logistic, SVM, kNN, MLP, and gradient boosting.

Key result:

- Broad10 has routing headroom: best-single mean utility is `0.6656`, query oracle is `0.8821`, and oracle gap is `0.2165`.
- Utility-oracle RouteCode reaches recovered gap `0.6941` at K=16 and `0.9441` at K=64; regret-objective oracle labels reach recovered gap `0.9507` at K=8 and `1.0000` at K=16.
- Flat predicted RouteCode labels remain weak at K=16 with recovered gap `-0.5822`.
- D2 at alpha `1.0` is modestly positive: embedding-centroid mean utility `0.7048`, recovered gap `0.1809`, and label accuracy `0.9224`.
- Full 10-model D2 reaches mean utility `0.7001` and recovered gap `0.1595`; across D2 scale rows, recovered gap ranges from `0.0000` to `0.2348`.
- Transferred source-D2 recovered gap ranges from `0.1596` to `0.2917` across six disjoint transfer scenarios, with mean `0.2088`; native target-D2 mean recovered gap is `0.2116`; same-budget direct retraining mean recovered gap is `0.0324`.
- This broad fixed-rectangle layer strengthens the model-pool transfer diagnostic relative to broad20, but it remains bounded local evidence and does not replace exact upstream external-router baselines.

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
- `results/llmrouterbench_broad20/table_new_model_integration.csv`
- `results/llmrouterbench_broad20/fig_transfer_calibration_curve.pdf`
- `results/llmrouterbench_broad20/phase_e5_new_model_calibration_memo.md`
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
- `results/llmrouterbench_broad20/table_sensitivity_summary.csv`
- `results/llmrouterbench_broad20/fig_sensitivity_summary.pdf`
- `results/llmrouterbench_broad20/phase_g_sensitivity_memo.md`
- `results/llmrouterbench_broad20/table_model_pool_scale.csv`
- `results/llmrouterbench_broad20/phase_f_g_model_pool_scale_memo.md`
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
- `results/llmrouterbench_pilot/table_avengerspro_upstream_metric.csv`
- `results/llmrouterbench_pilot/phase_e_avengerspro_upstream_metric_memo.md`
- `results/llmrouterbench_broad20/table_avengerspro_upstream_metric.csv`
- `results/llmrouterbench_broad20/phase_e_avengerspro_upstream_metric_memo.md`
- `results/llmrouterbench_pilot/table_cost_quality_summary.csv`
- `results/llmrouterbench_pilot/table_cost_quality_frontier.csv`
- `results/llmrouterbench_pilot/fig_cost_quality_frontier.pdf`
- `results/llmrouterbench_pilot/phase_e_cost_quality_memo.md`
- `results/llmrouterbench_broad20/table_cost_quality_summary.csv`
- `results/llmrouterbench_broad20/table_cost_quality_frontier.csv`
- `results/llmrouterbench_broad20/fig_cost_quality_frontier.pdf`
- `results/llmrouterbench_broad20/phase_e_cost_quality_memo.md`
- `results/llmrouterbench_broad20/table_stronger_direct_router_probe.csv`
- `results/llmrouterbench_broad20/phase_e_stronger_direct_router_probe_memo.md`

Status: complete for B0/B1/B2, D2 predictability-constrained RouteCode, all-model held-out/new-model calibration, bounded stronger direct-router probing, residual/adaptive-refinement gate diagnostics, expanded split sensitivity, expanded bounded ablations, bounded Phase G sensitivity, model-pool scale/composition, expanded local held-out model-pool transfer, local external-style baseline surrogates, RouteLLM MF official-code local-embedding evaluation, local LLMRouter trainer-class adapter metrics, a local Avengers-Pro compatibility baseline, exact upstream Avengers-Pro CLI accuracy/cost rows, upstream Avengers-Pro model-code RouteCode utility rows, benchmark-metadata cost-quality operating points, and a partial provider-price sensitivity diagnostic on the audited 18-dataset/20-model complete rectangle. Most published upstream checkpoints/full upstream-command external baselines have not yet been rerun on this rectangle; the broad ablation/sensitivity/transfer/external/provider rows remain local or partial diagnostics, not exhaustive sweeps.

Covered in this layer:

- canonical broad20 data with `280,820` query-model rows, `14,041` queries, `18` datasets, and `20` models;
- B0 routability;
- B1 compression ladder;
- B2 rate-distortion over K `[1, 2, 4, 8, 16, 32, 64, 128]`;
- D2 predictability-constrained RouteCode alpha sweep at K=16;
- all-model D4/E5 held-out/new-model calibration over all `20` broad20 models and r `[1, 2, 4, 8, 16, 32, 64]`, with `620` total rows;
- bounded stronger direct-router probe over two held-out models, r `[8, 64]`, and direct methods logistic, SVM, kNN, MLP, and gradient boosting;
- residual concentration and deployable residual-risk gate diagnostics;
- expanded split sensitivity covering `65` scenarios and `520` method rows: random, all `18` leave-dataset-out cases, all `11` leave-domain-out cases, all `11` domain-homogeneous cases, all `4` configured cluster-held-out cases, and all `20` single-model model-pool-holdout cases;
- expanded bounded ablations with `134` rows over K `[4, 8, 16, 32, 64, 128]`, lambda `[0.0, 0.05, 0.1, 0.2]`, seeds `[3, 7, 11]`, train fractions `[0.25, 0.5, 1.0]` for best-single/kNN/logistic/SVM/D2, and D2 beta `[0.0, 0.1, 1.0, 3.0]`;
- bounded Phase G sensitivity over embedding feature variants, clustering algorithms, label noise, cost mis-estimation, price-ratio stress, model-pool subset/composition, automatic dominated/complementary pools, domain granularity, query-length buckets, and bootstrap counts;
- broad model-pool scale/composition over train-only top, complementary, dominated, and full pools at sizes `[2, 4, 8, 12, 16, 20]`;
- expanded local disjoint model-pool transfer covering `18` source/target scenarios and `162` method rows over source/target size pairs `4x4`, `4x8`, `8x4`, `8x8`, `12x4`, and `12x8`;
- local RouteLLM/LLMRouter-style external baseline surrogates with explicit non-official implementation notes;
- local LLMRouterBench RouteLLM MF training code on broad20 split-aligned pairwise assets with local RouteCode embeddings.
- local LLMRouter KNN/SVM trainer-class adapter metrics on broad20 split-aligned RouteCode embeddings, with separate exact upstream train-CLI, full-test route-only infer-CLI evidence, and RouteCode postprocessed metrics over exact CLI outputs.
- local Avengers-Pro cluster-routing contract implementation on broad20 with RouteCode deterministic embeddings and explicit non-official implementation notes.
- upstream Avengers-Pro `SimpleClusterRouter` model-code execution on split-aligned broad20 assets, with RouteCode utility postprocessing over captured `routing_details` and explicit non-exact-command notes.
- fixed-quality and fixed-cost operating-point diagnostics over lambda `[0.0, 0.05, 0.1, 0.2]` using released benchmark cost metadata, with both all-method and deployable-method frontier rows.

Key result:

- Broad20 query oracle is strong: best-single mean utility is `0.7037`, query oracle is `0.9160`, and oracle regret is `0.2123`.
- Dataset-label lookup is weak on this broad split: mean utility `0.7172`, recovered gap `0.0638`.
- Embedding-cluster lookup reaches mean utility `0.7222` and recovered gap `0.0872`; kNN is slightly below best-single with recovered gap `-0.0067`.
- Utility-vector RouteCode oracle labels recover `0.3658` at K=16, `0.5151` at K=32, `0.7064` at K=64, and `0.7903` at K=128.
- Regret-objective RouteCode oracle labels are much stronger: recovered gap is `0.6510` at K=4, `0.8540` at K=8, `0.9681` at K=16, and `1.0000` by K=32.
- Current predicted RouteCode labels remain below best-single on this broad rectangle: logistic predicted K=16 has recovered gap `-0.4782`, and MLP predicted K=16 has `-0.4581`.
- Best deployable D2 at alpha `1.0` reaches mean utility `0.7229`, recovered gap `0.0906`, and label accuracy `0.7646`. This is a small gain over best-single and close to embedding KMeans recovered gap `0.0872`, not a large broad-rectangle recovery result.
- Broad all-model new-model calibration favors RouteCode over lightweight direct retraining: mean RouteCode label-calibration utility reaches `0.7183` and recovered gap `0.0687` at r=64 across all `20` held-out broad20 models, while the strongest lightweight direct retraining mean row is kNN at r=32/r=64 with mean utility about `0.5388` and recovered gap about `-0.777`. The all-model run uses fast logistic/SVM direct-router settings for tractability, so this keeps the sample-efficiency claim alive diagnostically but still does not cover transformer direct routers or official external routers.
- Broad stronger-direct probe adds MLP and gradient boosting under a bounded two-held-out-model protocol: RouteCode label calibration averages mean utility `0.7167` and recovered gap `0.0612` at r=64, while the best stronger direct-router average is logistic at r=64 with mean utility `0.5504` and recovered gap `-0.7223`; MLP at r=64 reaches mean utility `0.5486` and recovered gap `-0.7307`. This narrows the local direct-router comparison gap but is still not a full calibration proof.
- Broad residual regret concentration remains visible by oracle sorting: top 5%, 10%, and 20% high-regret queries account for `0.1600`, `0.3190`, and `0.6379` of residual regret. Deployable risk signals are weak: low route-label confidence captures only `0.0556` and `0.1056` of regret at top 5% and 10% flagged queries, with AUC `0.5370`; adaptive refinement remains deferred.
- Expanded split sensitivity now covers `65` broad20 scenarios with no skipped rows. It shows stronger benchmark-design effects than the initial shallow run: `leave_dataset_out:mbpp` rank correlation vs random is `0.1198`, `leave_dataset_out:humaneval` is `0.4551`, `cluster_held_out:1` and `leave_domain_out:dialogue` are both `0.4791`, and `leave_dataset_out:emorynlp` is `0.6988`. This keeps benchmark split design as a major diagnostic thread and now includes all `20` broad20 single-model pool holdouts. The completed split-rate threshold table covers all `65` scenarios with the full K ladder `[1, 2, 4, 8, 16, 32, 64, 128]`; only `leave_dataset_out:mbpp` reaches the 80% learned-router-gain threshold, at `rate_log2K_to_80pct_learned_gain = 5.0`.
- Expanded bounded broad ablations preserve the main pattern after adding D2/kNN/best-single train-size rows and making the broad KMeans/direct-router fits explicitly bounded: regret-objective oracle RouteCode has mean recovered gap `0.9081` across the configured K/lambda rows, flat utility-oracle RouteCode averages `0.3810`, semantic KMeans averages `0.0569`, and deployable D2 averages `0.0390`; D2 seed-stability mean recovered gap is `0.0901`. In the train-fraction slice, D2 averages `0.0185`, kNN averages `0.0089`, and the fast bounded logistic/SVM rows are negative on this broad20 rectangle.
- Bounded broad Phase G sensitivity adds `226` rows across `11` sensitivity families. D2 recovered gap averages `0.0850` across embedding-backbone variants, `0.1159` under cost mis-estimation, `0.0872` under price-ratio stress, `0.0797` across model-pool subsets, and `0.0627` across automatic dominated/complementary pools. Domain and length buckets remain heterogeneous: D2 ranges from `-0.0727` to `0.5000` across domain-granularity buckets and from `-0.0335` to `0.2263` across query-length buckets.
- Broad model-pool scale adds `48` rows over model counts `[2, 4, 8, 12, 16, 20]`. D2 recovered gap ranges from `0.0000` to `0.0980`; the best D2 row is `complementary_12` with mean utility `0.7229`, recovered gap `0.0980`, test oracle gap `0.1962`, and test dominance ratio `0.8038`. On the full 20-model broad pool, D2 reaches mean utility `0.7158` and recovered gap `0.0570`, while kNN is slightly below best-single at recovered gap `-0.0067`.
- Expanded local broad transfer keeps the model-pool transfer claim alive diagnostically: transferred source-D2 labels recover `0.0584` to `0.2592` of the target oracle gap across `18` disjoint source/target scenarios, with mean `0.1858`; native target-D2 recovers `0.0730` to `0.2668`, with mean `0.1965`; same-budget direct retraining over logistic/SVM/kNN/MLP/gradient boosting recovers only `-0.0511` to `0.0738`, with mean about `0.0130`.
- Broad local external-style surrogates are weak: the MF utility surrogate reaches mean utility `0.6934` and recovered gap `-0.0487`, while binary RouteLLM-style strong/weak surrogates range from recovered gap `-0.1560` to `-1.5034`.
- Broad RouteLLM MF official-code local-embedding rows are only slightly above best-single: validation accuracy is `0.9106`, but the best threshold `0.5` reaches mean utility `0.7073` and recovered gap `0.0168`.
- Broad local LLMRouter trainer-class adapter rows are below best-single: KNN reaches mean utility `0.5214` and recovered gap `-0.8591`, while SVM reaches mean utility `0.4833` and recovered gap `-1.0386`.
- Broad local Avengers-Pro compatibility rows are below best-single: the simple K=16 cluster row reaches mean utility `0.6574` and recovered gap `-0.2181`, while the balance row reaches mean utility `0.6449` and recovered gap `-0.2768`. The upstream model-code utility row is similar, with mean utility `0.6567`, recovered gap `-0.2215`, upstream accuracy `0.5988`, and `2,808` test predictions.
- Broad cost-quality operating points expose deployability limits under fixed-quality targets: no deployable broad20 method reaches the tested oracle-fraction quality thresholds. Under fixed-cost budgets, deployable frontier winners vary among cheapest, kNN, embedding-cluster lookup, and D2; at lambda `0.0`, D2 is the deployable winner at the largest tested budget with mean quality `0.7229` and mean cost `0.0004`, while kNN wins the next lower budget with mean quality `0.7023` and mean cost `0.0004`. These rows use released benchmark cost metadata and are not provider-price claims.
- Provider-price sensitivity rows now include two source-checked schedules from `2026-06-15`: the original OpenRouter-only snapshot and a mixed public-provider subset with per-model provider attribution. The OpenRouter schedule covers `2/6` pilot models and `3/20` broad20 models (`Qwen3-8B`, `Llama-3.1-8B-Instruct`, and OpenRouter's free `NVIDIA-Nemotron-Nano-9B-v2` route). The mixed schedule covers `4/6` pilot models and `12/20` broad20 models by adding `DeepSeek-R1-Distill-Qwen-7B`, `Qwen2.5-Coder-7B-Instruct`, `gemma-2-9b-it`, `granite-3.3-8b-instruct`, `DeepSeek-R1-0528-Qwen3-8B`, `GLM-Z1-9B-0414`, `NVIDIA-Nemotron-Nano-9B-v2`, `OpenThinker3-7B`, `cogito-v1-preview-llama-8B`, and `glm-4-9b-chat` where available. The remaining mixed-schedule gaps now carry source-checked unmapped notes rather than generic omissions: `DeepHermes-3-Llama-3-8B-Preview` and `Llama-3.1-Nemotron-Nano-8B-v1` have catalog pages without active per-token prices in the checked snapshot; `Fin-R1`, `Llama-3.1-8B-UltraMedical`, and `MiMo-7B-RL-0530` have flat-rate hosted pages without per-token prices; `Intern-S1-mini`, `MiniCPM4.1-8B`, and `internlm3-8b-instruct` have exact model-card/self-host sources but no provider token price. On broad20 mixed-provider rows at lambda `0.0`, best-single mean quality is `0.7037`, embedding-cluster lookup is `0.7222`, D2 is `0.7172`, and the twelve-model query oracle is `0.8996`. This remains a sensitivity diagnostic, not a full provider-cost claim.

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

Internal baselines, local external-style surrogates, official upstream RouteLLM artifact inspection, RouteLLM pairwise split-alignment substrate export, RouteLLM MF trainer asset export, split-aligned RouteLLM MF local-code evaluation, local LLMRouter trainer-class adapter metrics, a split-aligned local Avengers-Pro cluster-routing compatibility evaluation, full-split cache-backed exact upstream Avengers-Pro simple-cluster accuracy/cost metrics, and upstream Avengers-Pro model-code RouteCode utility metrics are now covered, but the full required split-aligned official external baseline set is still incomplete.

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
- `results/llmrouterbench_pilot/table_external_command_readiness.csv`
- `results/llmrouterbench_pilot/phase_e_external_command_readiness_memo.md`
- `results/llmrouterbench_broad20/table_routellm_mf_assets.csv`
- `results/llmrouterbench_broad20/phase_e_routellm_mf_assets_memo.md`
- `results/llmrouterbench_pilot/table_graphrouter_assets.csv`
- `results/llmrouterbench_pilot/phase_e_graphrouter_assets_memo.md`
- `results/llmrouterbench_broad20/table_graphrouter_assets.csv`
- `results/llmrouterbench_broad20/phase_e_graphrouter_assets_memo.md`
- `results/llmrouterbench_broad20/table_external_command_readiness.csv`
- `results/llmrouterbench_broad20/phase_e_external_command_readiness_memo.md`
- `results/llmrouterbench_pilot/table_external_baseline_assets.csv`
- `results/llmrouterbench_pilot/phase_e_external_baseline_assets_memo.md`
- `results/llmrouterbench_broad20/table_external_baseline_assets.csv`
- `results/llmrouterbench_broad20/phase_e_external_baseline_assets_memo.md`
- `results/llmrouterbench_pilot/table_llmrouter_library_adapters.csv`
- `results/llmrouterbench_pilot/phase_e_llmrouter_library_adapters_memo.md`
- `results/llmrouterbench_pilot/table_llmrouter_cli_metrics.csv`
- `results/llmrouterbench_pilot/phase_e_llmrouter_cli_metrics_memo.md`
- `results/llmrouterbench_broad20/table_llmrouter_library_adapters.csv`
- `results/llmrouterbench_broad20/phase_e_llmrouter_library_adapters_memo.md`
- `results/llmrouterbench_broad20/table_llmrouter_cli_metrics.csv`
- `results/llmrouterbench_broad20/phase_e_llmrouter_cli_metrics_memo.md`
- `results/llmrouterbench_pilot/table_frugalgpt_cli_metrics.csv`
- `results/llmrouterbench_pilot/phase_e_frugalgpt_cli_metrics_memo.md`
- `results/llmrouterbench_broad20/table_frugalgpt_cli_metrics.csv`
- `results/llmrouterbench_broad20/phase_e_frugalgpt_cli_metrics_memo.md`
- `results/llmrouterbench_pilot/table_embedllm_knn_cli_metrics.csv`
- `results/llmrouterbench_pilot/phase_e_embedllm_knn_cli_metrics_memo.md`
- `results/llmrouterbench_broad20/table_embedllm_knn_cli_metrics.csv`
- `results/llmrouterbench_broad20/phase_e_embedllm_knn_cli_metrics_memo.md`
- `results/llmrouterbench_pilot/table_embedllm_mf_cli_metrics.csv`
- `results/llmrouterbench_pilot/phase_e_embedllm_mf_cli_metrics_memo.md`
- `results/llmrouterbench_broad20/table_embedllm_mf_cli_metrics.csv`
- `results/llmrouterbench_broad20/phase_e_embedllm_mf_cli_metrics_memo.md`
- `results/llmrouterbench_pilot/table_provider_price_schedule.csv`
- `results/llmrouterbench_pilot/table_provider_cost_quality_summary.csv`
- `results/llmrouterbench_pilot/table_provider_cost_quality_frontier.csv`
- `results/llmrouterbench_pilot/phase_g_provider_pricing_memo.md`
- `results/llmrouterbench_broad20/table_provider_price_schedule.csv`
- `results/llmrouterbench_broad20/table_provider_cost_quality_summary.csv`
- `results/llmrouterbench_broad20/table_provider_cost_quality_frontier.csv`
- `results/llmrouterbench_broad20/phase_g_provider_pricing_memo.md`
- `results/llmrouterbench_pilot/table_claim_status.csv`
- `results/llmrouterbench_pilot/phase_h_claim_status_memo.md`
- `results/llmrouterbench_pilot/table_model_pool_transfer.csv`
- `results/llmrouterbench_pilot/phase_f_g_model_pool_transfer_memo.md`
- `results/llmrouterbench_broad20/table_claim_status.csv`
- `results/llmrouterbench_broad20/phase_h_claim_status_memo.md`
- `results/table_claim_status_by_run.csv`
- `results/table_claim_status_global.csv`
- `results/phase_h_global_claim_status_memo.md`
- `results/table_paper_evidence_summary.csv`
- `results/phase_h_paper_evidence_summary.md`
- `results/table_research_flow_completion.csv`
- `results/phase_h_research_flow_completion_audit.md`
- `results/README.md`
- `paper_notes.md`

Still missing or not integrated as external/stronger baselines:

- RouteLLM-BERT output on the RouteCode pilot/broad20 splits; the exact upstream RouteLLM-MF training and evaluation CLIs now run on split-aligned pilot and broad20 local assets with cache-backed RouteCode embeddings and no API calls, and separate split-aligned MF models have been trained/evaluated with local official MF source and RouteCode embeddings. BERT remains missing;
- official GraphRouter metric output where feasible; one-epoch exact upstream GraphRouter command-path smoke metrics now exist for pilot and broad20, but they report upstream accuracy/cost under GraphRouter's internal split rather than RouteCode utility;
- BEST-Route, RouterDC, and MODEL-SAT metric outputs; split-aligned input assets now exist for pilot and broad20. FrugalGPT now has a local encoder checkpoint path, bounded pilot and broad20 smoke execution, a full one-epoch pilot RouteCode-compatible local metric-adapter table, a bounded one-step broad20 RouteCode-compatible local metric-adapter table, and RouteCode-postprocessed metrics over saved scorer directories emitted by the exact FrugalGPT command. EmbedLLM KNN now has patched local CLI wiring, split-aligned full-train/full-test exact upstream tensor executions with correctness metrics for pilot and broad20, and a RouteCode-compatible local metric-adapter table for pilot and broad20. EmbedLLM MF now has upstream-compatible padded `question_embeddings_3584.pth` assets plus full-split upstream router-mode executions for pilot and broad20; the remaining blockers are the heavier dependency/checkpoint stacks for other external baselines.

Current exact command-path readiness:

- `routecode_local_routellm_mf_metric` is available and RouteCode-metric-compatible, but it is not an exact upstream command.
- `routecode_local_embedllm_knn_metric` is available and RouteCode-metric-compatible for pilot and broad20, but it is a local metric adapter rather than an exact upstream command or published checkpoint.
- `routecode_local_frugalgpt_metric` is available and RouteCode-metric-compatible for pilot and broad20, but it is a local metric adapter rather than an exact upstream command or published checkpoint; the broad20 row is a bounded one-step scorer run. `table_frugalgpt_cli_metrics.csv` separately scores saved scorer directories emitted by the exact FrugalGPT command with RouteCode utility.
- `routecode_upstream_avengerspro_metric` is available and RouteCode-metric-compatible for pilot and broad20, but it is not an exact upstream command output because the upstream CLI JSON omits per-query routing details.
- The readiness table now covers `19` rows for pilot and broad20: RouteCode local MF metric, RouteCode local EmbedLLM KNN metric, RouteCode local FrugalGPT metric, upstream-code Avengers-Pro RouteCode metric, LLMRouter KNN/SVM train CLI, LLMRouter KNN/SVM route-only infer CLI, RouteLLM MF train/eval, RouteLLM-BERT, Avengers-Pro CLI, GraphRouter, FrugalGPT, EmbedLLM KNN, EmbedLLM MF, BEST-Route, RouterDC, and MODEL-SAT.
- Runnable rows now: `15`; runnable exact upstream-command rows now: `11`.
- `llmrouter_knn_train_cli` and `llmrouter_svm_train_cli` now have split-aligned YAML/JSONL/tensor assets and bounded exact upstream training CLI smoke executions for pilot and broad20. Evidence lives at `results/llmrouterbench_pilot/llmrouter_library_adapters/llmrouter_knn_train_stdout.log`, `results/llmrouterbench_pilot/llmrouter_library_adapters/llmrouter_svm_train_stdout.log`, `results/llmrouterbench_broad20/llmrouter_library_adapters/llmrouter_knn_train_stdout.log`, and `results/llmrouterbench_broad20/llmrouter_library_adapters/llmrouter_svm_train_stdout.log`. These are exact training command-path checks, not exact upstream metric rows.
- `llmrouter_knn_infer_cli` and `llmrouter_svm_infer_cli` now have full-test exact upstream route-only inference executions for pilot and broad20 using `query_embedding_lookup.pt` to avoid Longformer downloads and external API calls. Evidence lives at `results/llmrouterbench_pilot/llmrouter_library_adapters/llmrouter_knn_full_predictions.json`, `results/llmrouterbench_pilot/llmrouter_library_adapters/llmrouter_svm_full_predictions.json`, `results/llmrouterbench_broad20/llmrouter_library_adapters/llmrouter_knn_full_predictions.json`, and `results/llmrouterbench_broad20/llmrouter_library_adapters/llmrouter_svm_full_predictions.json`.
- `table_llmrouter_cli_metrics.csv` now scores those exact upstream LLMRouter full-test route-only outputs with RouteCode utility. Pilot KNN/SVM both select `Qwen3-8B` and recover `0.0000` vs oracle over `580` predictions. Broad20 KNN reaches mean utility `0.5221` and recovered gap `-0.8557` over `2808` predictions; broad20 SVM reaches mean utility `0.4843` and recovered gap `-1.0336`. These are RouteCode post-processed metrics over exact upstream command outputs, not published LLMRouter benchmark artifacts.
- `routellm_mf_train_cli` executed locally without API calls and wrote `results/llmrouterbench_pilot/routellm_mf_assets/mf_model.pt` and `results/llmrouterbench_broad20/routellm_mf_assets/mf_model.pt`.
- `routellm_mf_eval_cli` now has bounded cache-backed exact upstream evaluation smoke executions for pilot and broad20 using `mf_eval_config.local.json`, `embedding_config.local.yaml`, and `embedding_cache.jsonl` to avoid embedding API calls. Evidence lives at `results/llmrouterbench_pilot/routellm_mf_assets/routellm_mf_eval_stdout.log` and `results/llmrouterbench_broad20/routellm_mf_assets/routellm_mf_eval_stdout.log`. The pilot eval covers `580` pairwise test samples with selection accuracy `0.7207`; broad20 covers `2808` pairwise test samples with selection accuracy `0.7037`. These are exact evaluator command-path checks, while RouteCode utility metrics remain in `table_routellm_mf_split_aligned.csv`.
- `routellm_bert_cli` is blocked by a missing local BERT checkpoint.
- `avengerspro_cli` now has full-split cache-backed exact upstream simple-cluster executions for pilot and broad20, using RouteCode-generated train/test JSONL plus `full_embedding_cache.jsonl` and `simple_cluster_config.full.json` to avoid embedding API calls. Evidence lives at `results/llmrouterbench_pilot/table_avengerspro_cli_metrics.csv`, `results/llmrouterbench_pilot/avengerspro_cli_metrics/simple_cluster_full_results.json`, `results/llmrouterbench_broad20/table_avengerspro_cli_metrics.csv`, and `results/llmrouterbench_broad20/avengerspro_cli_metrics/simple_cluster_full_results.json`. Pilot dataset-level accuracy is `0.7043` over `580` test queries; broad20 dataset-level accuracy is `0.6035` over `2808` test queries. These exact CLI artifacts are upstream accuracy/cost rows because the saved JSON omits per-query `routing_details`. Separate RouteCode utility rows over routing decisions captured from the upstream `SimpleClusterRouter` class now exist at `results/llmrouterbench_pilot/table_avengerspro_upstream_metric.csv` and `results/llmrouterbench_broad20/table_avengerspro_upstream_metric.csv`; these use upstream model code but are not exact command outputs.
- `graphrouter_cli` now has generated router-data and LLM-description embedding assets plus one-epoch exact upstream smoke executions for pilot and broad20. Pilot reaches dataset-level accuracy `0.6378`, sample-level accuracy `0.5747`, and total cost `1.4794`; broad20 reaches dataset-level accuracy `0.2997`, sample-level accuracy `0.3530`, and total cost `0.0187`. These are upstream accuracy/cost smoke metrics, not RouteCode utility rows.
- `frugalgpt_local_scorer_cli` now has split-aligned train/test JSONL assets and a local MiniLM encoder checkpoint path for pilot and broad20. The pilot has a bounded upstream-command smoke execution at `results/llmrouterbench_pilot/frugalgpt_split_aligned/output/frugalgpt_smoke_stdout.log` with record accuracy `0.5181`, prompt accuracy `0.7621`, and macro dataset accuracy `0.5268`; broad20 has a bounded max-sample smoke execution at `results/llmrouterbench_broad20/frugalgpt_split_aligned/output/frugalgpt_smoke_stdout.log` with record accuracy `0.4789`, prompt accuracy `0.4940`, and macro dataset accuracy `0.5016`. These are runtime evidence for the exact command path. A separate one-epoch RouteCode-compatible local metric-adapter row now exists at `results/llmrouterbench_pilot/table_frugalgpt_split_aligned.csv`: mean utility `0.7517`, recovered gap vs oracle `0.3684`, record accuracy `0.6417`, and prompt accuracy `0.8966`. A bounded one-step broad20 local metric-adapter row exists at `results/llmrouterbench_broad20/table_frugalgpt_split_aligned.csv`: mean utility `0.5791`, recovered gap vs oracle `-0.5872`, record accuracy `0.5568`, and prompt accuracy `0.7336`. RouteCode-postprocessed exact FrugalGPT CLI saved-scorer rows now also exist: pilot mean utility `0.5983`, recovered gap `-0.3008` over `580` predictions; broad20 mean utility `0.6695`, recovered gap `-0.1611` over `2808` predictions.
- `embedllm_knn_cli` and `embedllm_mf_cli` now have split-aligned CSV/tensor assets. KNN has patched local argparse wiring, local `sentence_transformers`, and full train/test exact upstream tensor executions at `results/llmrouterbench_pilot/embedllm_knn_cli_metrics/embedllm_knn_k131_stdout.log` and `results/llmrouterbench_broad20/embedllm_knn_cli_metrics/embedllm_knn_k131_stdout.log`; the best pilot correctness row is k=15 with mean correctness accuracy `0.6718`, and the best broad20 correctness row is k=15 with mean correctness accuracy `0.6919`. These are exact upstream correctness metrics, not RouteCode routing utility. Separate RouteCode-compatible local metric-adapter rows now exist at `results/llmrouterbench_pilot/table_embedllm_knn_split_aligned.csv` and `results/llmrouterbench_broad20/table_embedllm_knn_split_aligned.csv`; the best pilot row is k=131 with mean utility `0.7500` and recovered gap vs oracle `0.3609`, and the best broad20 row is k=131 with mean utility `0.7304` and recovered gap vs oracle `0.1258`. MF now uses a patched local no-op `wandb` fallback, points to `question_embeddings_3584.pth`, and has full-split upstream router-mode executions at `results/llmrouterbench_pilot/embedllm_mf_cli_metrics/embedllm_mf_stdout.log` and `results/llmrouterbench_broad20/embedllm_mf_cli_metrics/embedllm_mf_stdout.log`; pilot best dataset-level accuracy is `0.6345` with final test sample accuracy `0.6138`, and broad20 best dataset-level accuracy is `0.5427` with final test sample accuracy `0.5502`. These are exact upstream router-accuracy metrics, not RouteCode routing utility. Earlier bounded smoke logs remain command-path evidence only.
- `best_route_train_cli`, `routerdc_train_cli`, and `modelsat_train_cli` now have split-aligned training/eval assets and remain blocked by local model checkpoints plus required training dependencies such as `llm_blender`, `deepspeed`, `wandb`, and `nltk`.
- Local LLMRouter trainer-class adapter metric rows and exact-CLI postprocessed metric rows now exist: pilot KNN/SVM collapse to best-single (`0.6672` mean utility), while broad20 KNN/SVM remain below best-single. Exact upstream LLMRouter KNN/SVM train-CLI and full-test route-only infer-CLI rows validate the upstream command entrypoints on the same split-aligned assets.

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
- `results/llmrouterbench_pilot/table_transformer_embedding_router.csv`
- `results/llmrouterbench_pilot/phase_f_g_transformer_embedding_router_memo.md`

Status: complete for the first bounded Phase F/G ablation layer.

Covered in this layer:

- code count K over `[4, 8, 16, 32, 64, 128]`;
- code objective comparison among semantic clusters, utility-vector RouteCode, regret-objective RouteCode, and predictability-constrained D2;
- cost weight lambda over `[0.0, 0.05, 0.1, 0.2]` for the pilot;
- D2 rate-penalty beta over `[0.0, 0.1, 1.0, 3.0]`;
- random seed stability over seeds `[3, 7, 11]`;
- training data fraction for best-single, kNN, logistic, SVM, and D2 RouteCode over `[0.25, 0.5, 1.0]`;
- new-model calibration examples per label from the E5 pilot.
- with/without code-card interpretability as an observability ablation for flat RouteCode and D2 RouteCode.
- cache-only transformer-backbone readiness for requested `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`, `answerdotai/ModernBERT-base`, and `microsoft/deberta-v3-base` checkpoints.
- local-files-only transformer direct-router execution for cached `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`, `answerdotai/ModernBERT-base`, and `microsoft/deberta-v3-base`.

Key result:

- Seed stability: D2 embedding-centroid mean recovered gap is `0.3320` with std `0.0177`; kNN mean recovered gap is `0.2723` with std `0.0356`.
- K/lambda: flat utility-oracle RouteCode reaches the query oracle on this pilot split at K=32 and remains effectively saturated at K=64 and K=128 across tested lambdas; D2 remains the stronger deployable RouteCode variant but does not close the oracle-code gap, and its K/lambda mean recovered gap declines for K above 16 in this pilot.
- Regret-objective RouteCode: oracle-label regret-code rows reach the query oracle by K=8 on the pilot rate-distortion curve; deployable embedding-centroid regret-code rows remain much lower, with K=16 recovered gap `0.1053`.
- Rate penalty: D2 beta values `[0.0, 0.1, 1.0, 3.0]` produce the same deployable recovered gap (`0.3459`) and empirical label entropy (`3.1120`) on this pilot slice, so the current balance penalty is not an active lever at the selected K/alpha.
- Code-card interpretability: label-only rows expose `1` audited explanatory field, while code-card rows expose `9` fields for both flat RouteCode and D2. Code-card coverage is `1.0000` for domain summaries, dataset summaries, representative queries, high-regret examples, utility vectors, and human-readable explanations. This is an observability result, not a routing-utility result.
- Transformer-backbone readiness and execution wiring: all requested encoder backbones now produce split-aligned direct-router rows with no API calls, including BGE-small and E5-small retrieval-style encoders. On the pilot, the strongest requested-encoder row is DeBERTa logistic (`0.6741` mean utility, recovered gap `0.0301`), followed by E5-small logistic/SVM (`0.6724`, `0.0226`) and BGE-small kNN (`0.6690`, `0.0075`); MiniLM kNN equals best-single (`0.6672`, `0.0000`). On broad20, all requested encoder direct routers remain substantially below best-single: the strongest row is ModernBERT logistic (`0.5427`, recovered gap `-0.7584`), followed by ModernBERT SVM (`0.5417`, `-0.7634`), MiniLM kNN (`0.5402`, `-0.7701`), and E5-small logistic (`0.5370`, `-0.7852`). This closes the requested BGE/E5/ModernBERT/DeBERTa execution gap but remains negative evidence for simple pretrained-encoder direct routers on the broad rectangle.

### Remaining Phase F: Ablation Study

Required but incomplete:

- predictor type beyond local lightweight logistic/SVM/kNN/MLP/gradient-boosting probes and requested cached encoders remains incomplete for claim purposes; MiniLM, BGE-small, E5-small, ModernBERT, and DeBERTa direct-router metric rows now exist, but tuned text encoders and exact external-router command paths remain missing;
- broader held-out model-pool transfer protocols beyond the same-six-dataset, expanded local broad20 20-model, and 32-model stress transfer layers;
- adaptive-refinement ablation only if a future residual-risk gate justifies implementing refinement;

### Remaining Phase G: Sensitivity Analysis

Partially complete for split strategy and coarse domain granularity through B4, curated task-family/task-subtype taxonomy through the local and broad sensitivity suites, random seeds through the bounded ablation layer, and the local/broad sensitivity suites above. Broader required sensitivity remains incomplete.

Still missing:

- published upstream-checkpoint RouteLLM/BERT and native exact-command Avengers-Pro RouteCode utility output on the 18-dataset/20-model broad rectangle; exact upstream Avengers-Pro accuracy/cost rows, split-aligned local Avengers-Pro rows, and upstream model-code Avengers-Pro utility rows now exist, while native exact CLI utility output remains unavailable because the upstream JSON omits per-query routing details. Exact upstream GraphRouter accuracy/cost smoke rows and split-aligned GraphRouter GNN RouteCode utility adapter rows now exist for pilot and broad20, while native upstream utility metrics are still not emitted by the unmodified GraphRouter command;
- exhaustive rather than bounded sensitivity and transfer sweeps on the 18-dataset/20-model broad rectangle; broad20 split sensitivity is now expanded across all dataset/domain groups, configured clusters, all `20` single-model pool holdouts, and a completed full-ladder split-rate threshold table, but these are still bounded local diagnostics rather than exhaustive robustness proof;
- stronger external/pretrained embedding-backbone metric rows beyond the requested MiniLM/BGE-small/E5-small/ModernBERT/DeBERTa pass, such as tuned task encoders;
- complete provider-specific pricing coverage for all pilot/broad20 models; the current OpenRouter snapshot maps `3/20` broad models and the mixed public-provider schedule maps `12/20` broad models with per-model source URLs, while the remaining `8/20` broad models now have source-checked unmapped/blocker notes;
- model-pool robustness beyond the current same-six-dataset 20-model, broad10 18-dataset/10-model, broad20 18-dataset/20-model, and 32-model/5-dataset complete-rectangle diagnostics.

### Phase H: Final Paper Claims

Not complete. Current evidence supports only cautious diagnostic claims. Phase H now has reproducible claim-status artifacts:

- `results/llmrouterbench_pilot/table_claim_status.csv`
- `results/llmrouterbench_pilot/phase_h_claim_status_memo.md`
- `results/llmrouterbench_broad20/table_claim_status.csv`
- `results/llmrouterbench_broad20/phase_h_claim_status_memo.md`
- `results/table_paper_evidence_summary.csv`
- `results/phase_h_paper_evidence_summary.md`
- `results/table_research_flow_completion.csv`
- `results/phase_h_research_flow_completion_audit.md`
- `paper_notes.md`

Current claim-gate status:

- `low_rate_oracle_codes`: diagnostic-supported on all five audited runs: pilot (`1.0000` recovered gap by K<=16), broad10 (`1.0000`), broad20 (`0.9681`), scale20 (`0.9799`), and 32-model (`0.9535`).
- `small_inferred_labels`: not supported on all five audited runs; best inferred recovered gap is pilot (`0.3459`), followed by broad10 (`0.1891`), scale20 (`0.1409`), broad20 (`0.0906`), and 32-model (`0.0233`), all below the pre-committed `0.85` threshold and without recovered-gap CI support.
- `model_pool_transfer`: diagnostic-alive on pilot with matched-scenario mean transfer-minus-direct recovered gap `0.3083`, and diagnostic-alive on broad20 with matched-scenario mean transfer-minus-direct recovered gap `0.1472`.
- `new_model_calibration`: diagnostic-alive on pilot with matched held-out-model/budget mean RouteCode-minus-direct recovered gap `0.2339`, broad10 with matched mean difference `0.4106`, broad20 with matched mean difference `0.7402`, scale20 with matched mean difference `0.5096`, and 32-model with matched mean difference `0.8140`.
- `benchmark_diagnosis`: mixed evidence; diagnostic-supported on pilot and broad20 from split-ranking changes, including broad20 minimum rank correlation `0.1198`, but not supported by the bounded broad10, scale20, or 32-model gates.
- `adaptive_refinement`: not supported on all five audited runs because the residual-risk gates remain weak; `table_adaptive_refinement.csv` remains intentionally absent.

Global cross-run claim-gate status across pilot, broad10, broad20, scale20, and 32-model stress outputs:

- `low_rate_oracle_codes`: diagnostic-supported across all five audited runs with rate-distortion evidence; best recovered gap is `1.0000`, and worst recovered gap is `0.9535`.
- `small_inferred_labels`: not supported globally across all five audited runs; best inferred-label recovered gap is only `0.3459`, far below the `0.85` threshold.
- `model_pool_transfer`: mixed evidence; pilot, broad10, broad20, and scale20 are diagnostic-alive with matched transfer-minus-direct recovered-gap means from `0.1472` to `0.3083`, while the 32-model stress run is not supported at `-0.0537`.
- `new_model_calibration`: diagnostic-alive on all five audited result directories, with matched RouteCode-minus-direct recovered-gap means `0.2339`, `0.4106`, `0.7402`, `0.5096`, and `0.8140` for pilot, broad10, broad20, scale20, and 32-model respectively.
- `benchmark_diagnosis`: mixed evidence; pilot and broad20 are diagnostic-supported through split-ranking changes, while broad10, scale20, and 32-model are not supported by their current bounded gates.
- `adaptive_refinement`: not supported globally across all five audited runs; residual-risk evidence remains weak and no adaptive-refinement utility table exists.

Paper evidence summary:

- `results/table_paper_evidence_summary.csv` and `results/phase_h_paper_evidence_summary.md` now translate the current global claim gates and pilot/broad20 external readiness tables into paper-facing posture rows.
- The generated recommendation is `information_frontier_diagnostic`: do not claim that few inferred bits are enough; frame current evidence around low-rate oracle structure, modest deployable inferred-label recovery, and diagnostic calibration/transfer threads.
- `paper_notes.md` is generated from the same evidence table and records guardrails for paper wording and remaining external-baseline blockers.

Research Flow completion audit:

- `results/table_research_flow_completion.csv` and `results/phase_h_research_flow_completion_audit.md` now check explicit `Research Flow.md` phases against current artifact evidence.
- Current status is `8` complete phases, `1` deferred phase, `1` blocked phase, and `1` incomplete phase.
- Remaining non-complete phases are `phase_e_external_methods` due to blocked `routellm_bert_cli`, `best_route_train_cli`, `routerdc_train_cli`, and `modelsat_train_cli`, and `phase_h_final_claims` because small inferred labels and adaptive refinement are not supported while transfer and benchmark diagnosis are mixed.

Currently alive:

- useful low-rate utility codes exist;
- predictability-constrained labels are much easier to infer than utility-only labels in the pilot;
- simulated held-out-model calibration favors RouteCode over matched-budget direct logistic/SVM/kNN/MLP/gradient-boosting baselines across six pilot held-out models, over lightweight logistic/SVM/kNN baselines across all `20` broad20 held-out models, and over matched lightweight direct baselines in the broad10, scale20, and 32-model bounded calibration diagnostics;
- local external-style MF and binary RouteLLM-style surrogates are below kNN/D2 in the pilot;
- official upstream RouteLLM-MF aggregate artifacts have been parsed, but are not split-aligned RouteCode metrics;
- RouteLLM pairwise train/test substrate and MF trainer assets are now split-aligned with RouteCode for one configured strong/weak pair;
- RouteLLM-MF local-code checkpoint and RouteCode utility rows now exist for the pilot and broad20 pairs; the exact upstream MF training and cache-backed evaluation CLIs have executed on the split-aligned assets without API calls, while BERT coverage remains missing;
- split-aligned RouteLLM-MF local-code evaluation is now available with local RouteCode embeddings; best threshold `0.5` reaches mean utility `0.7259` and recovered gap vs oracle `0.2556`;
- split-aligned EmbedLLM KNN local metric-adapter rows now exist for pilot and broad20 using local `all-mpnet-base-v2` sentence-transformer embeddings and no API calls; best pilot k=131 reaches mean utility `0.7500` and recovered gap vs oracle `0.3609`, while best broad20 k=131 reaches mean utility `0.7304` and recovered gap vs oracle `0.1258`. Exact upstream EmbedLLM KNN CLI correctness rows also now exist on full split-aligned tensor assets: pilot k=15 reaches mean correctness accuracy `0.6718`, and broad20 k=15 reaches `0.6919`. These are metric-bearing adapters and exact upstream correctness rows around the upstream per-model kNN correctness idea, not upstream published checkpoints;
- split-aligned FrugalGPT local scorer metric-adapter rows now exist for the pilot and broad20 using the LLMRouterBench FrugalGPT local scorer source and no API calls; the one-epoch pilot row reaches mean utility `0.7517` and recovered gap vs oracle `0.3684`, while the bounded one-step broad20 row reaches mean utility `0.5791` and recovered gap vs oracle `-0.5872`. RouteCode-postprocessed exact FrugalGPT CLI saved-scorer rows also exist: pilot mean utility `0.5983` and recovered gap `-0.3008`, broad20 mean utility `0.6695` and recovered gap `-0.1611`. These are not upstream published checkpoints;
- split-aligned local Avengers-Pro compatibility rows now exist; the simple K=16 cluster row reaches mean utility `0.7397` and recovered gap vs oracle `0.3158`, while the balance row reaches mean utility `0.7241` and recovered gap vs oracle `0.2481`. Separate full-split cache-backed exact upstream Avengers-Pro simple-cluster accuracy/cost rows now exist for pilot and broad20 without embedding API calls: pilot dataset-level accuracy `0.7043` over `580` test queries, broad20 dataset-level accuracy `0.6035` over `2808` test queries. RouteCode utility rows over captured upstream `SimpleClusterRouter` routing decisions also now exist: pilot mean utility `0.7397`, recovered gap `0.3158`, upstream accuracy `0.7059`, `580` predictions; broad20 mean utility `0.6567`, recovered gap `-0.2215`, upstream accuracy `0.5988`, `2,808` predictions;
- local LLMRouter trainer-class adapter metrics now exist for KNN/SVM without API calls or Longformer embedding extraction; pilot rows collapse to best-single (`0.6672` mean utility), and broad20 rows remain below best-single (`0.5214` and `0.4833` mean utility). Exact upstream LLMRouter KNN/SVM training CLI and full-test route-only inference CLI executions now also exist for pilot and broad20, and RouteCode postprocessed metric rows over those exact CLI outputs are recorded in `table_llmrouter_cli_metrics.csv`;
- same-six-dataset 20-model scale rows now exist; on the full 20-model pool D2 reaches mean utility `0.7397` and recovered gap vs oracle `0.1275`, while complementary/top/dominated pools expose strong composition sensitivity;
- pilot and same-six-dataset held-out model-pool transfer rows now exist; pilot transferred source-D2 labels have mean recovered gap `0.3792` across three disjoint 3-to-3 scenarios and beat same-budget direct retraining by mean recovered gap `0.3083`, while scale20 transferred source-D2 labels have mean recovered gap `0.2248` across three disjoint 8-to-8 scenarios and remain competitive with same-budget direct retraining in this bounded diagnostic;
- raw LLMRouterBench coverage is now audited: `24` local datasets and `40` local models are present, taxonomy covers all `24` local datasets, an 18-dataset/10-model complete rectangle is evaluated in broad10, an 18-dataset/20-model complete rectangle is evaluated in broad20, and a 32-model/5-dataset complete rectangle is evaluated as a model-pool scale and disjoint transfer stress test;
- broad10 and broad20 B0/B1/B2, D2, broad10/scale20/32-model bounded new-model calibration, broad20 all-model new-model calibration, bounded stronger direct-router probe, residual-gate, expanded split-sensitivity, expanded bounded ablation, bounded sensitivity, broad model-pool scale, expanded local transfer, local external-surrogate, RouteLLM MF official-code, local and upstream-code Avengers-Pro utility rows, benchmark-metadata cost-quality operating-point metrics, and partial provider-price sensitivity metrics now exist: broad20 utility-vector RouteCode oracle labels recover `0.7903` by K=128, broad20 regret-objective labels recover `0.9681` by K=16, best broad20 deployable D2 recovered gap is `0.0906`, broad10 D2 recovered gap is `0.1809` at alpha `1.0`, broad expanded-ablation D2 seed-stability mean recovered gap is `0.0901`, broad20 scale D2 recovered gap ranges from `0.0000` to `0.0980` and reaches `0.0570` on the full 20-model pool, broad10 transferred source-D2 mean recovered gap is `0.2088` across six disjoint scenarios while same-budget direct retraining averages `0.0324`, broad10/scale20/32-model RouteCode label calibration is diagnostic-alive with matched RouteCode-minus-direct recovered-gap means `0.4106`, `0.5096`, and `0.8140`, broad20 all-model RouteCode label calibration reaches mean recovered gap `0.0687` at r=64 while lightweight direct retraining remains much lower, the broad stronger-direct probe keeps RouteCode above MLP and gradient-boosting direct retraining on two held-out models, broad residual risk signals remain weak, expanded split diagnostics show severe ranking changes such as `leave_dataset_out:mbpp` rank correlation `0.1198` vs random, broad sensitivity rows expose small positive D2 gaps with domain/query/model-pool heterogeneity, transferred source-D2 labels recover `0.0584` to `0.2592` across 18 disjoint broad20 model-pool transfer scenarios while stronger same-budget direct retraining recovers only `-0.0511` to `0.0738`, 32-model transferred source-D2 labels recover `-0.1429` to `0.0462` across three disjoint 16-to-16 scenarios, local external-style surrogates remain below best-single, split-aligned RouteLLM MF official-code rows recover only `0.0168` at the best threshold, broad local and upstream-code Avengers-Pro rows remain below best-single, deployable broad20 methods do not reach the tested high fixed-quality thresholds in the cost-quality frontier, and the provider-priced broad subsets remain partial (`3/20` OpenRouter-only, `12/20` mixed public-provider with source-checked notes for the remaining `8/20`);
- `results/llmrouterbench_broad20/phase_c_broad20_observation_memo.md` now records the broad claim-status checkpoint: broad20 supports useful low-rate oracle labels, benchmark split sensitivity, and model-pool-composition diagnostics; it does not support "small inferred labels recover most routing performance" because the best deployable D2 broad20 row recovers only `0.0906` of oracle gap.
- exact upstream external-command readiness is now reproducibly audited for pilot and broad20 across `19` rows: the current local environment has `15` runnable rows, including `11` runnable/executed exact upstream-command rows, `llmrouter_knn_train_cli`, `llmrouter_svm_train_cli`, `llmrouter_knn_infer_cli`, `llmrouter_svm_infer_cli`, `routellm_mf_train_cli`, `routellm_mf_eval_cli`, `avengerspro_cli`, `graphrouter_cli`, `frugalgpt_local_scorer_cli`, `embedllm_knn_cli`, and `embedllm_mf_cli`, on each audited output directory; LLMRouter KNN/SVM training and full-test route-only inference, RouteLLM MF train/eval, Avengers-Pro full-split accuracy/cost, GraphRouter one-epoch smoke accuracy/cost, FrugalGPT, full-split EmbedLLM KNN correctness, and full-split EmbedLLM MF router accuracy now have execution evidence for both pilot and broad20. RouteCode-compatible local or postprocessed metric rows are available for RouteLLM MF, EmbedLLM KNN, FrugalGPT, local LLMRouter trainer-class adapters, RouteCode-postprocessed exact LLMRouter CLI outputs, RouteCode-postprocessed exact FrugalGPT CLI saved-scorer outputs, upstream-code Avengers-Pro `SimpleClusterRouter` selections, and split-aligned GraphRouter GNN selections on pilot and broad20; exact upstream EmbedLLM KNN/MF, Avengers-Pro CLI, and GraphRouter CLI accuracy/cost metrics are tracked separately from RouteCode utility, because those native outputs do not emit RouteCode utility selections. RouteLLM-BERT is blocked by a missing checkpoint, and BEST-Route/RouterDC/MODEL-SAT now have split-aligned input assets with remaining checkpoint/dependency blockers recorded;
- transformer-backbone readiness and local-files-only direct-router execution are now reproducibly audited with no API calls for MiniLM, BGE-small, E5-small, ModernBERT, and DeBERTa; the resulting broad20 rows are weak/negative evidence for simple pretrained-encoder direct routers;
- benchmark split design changes rankings;
- dataset/topic labels are strong enough to motivate benchmark diagnosis.

Not yet supported:

- small inferred route labels recover most routing performance;
- paper-level route labels transfer across model pools beyond the bounded pilot, same-six-dataset, broad10/broad20 transfer, and 32-model stress diagnostics;
- new models need fewer calibration examples as a paper-level claim beyond the bounded diagnostics now present for pilot, broad10, broad20, scale20, and 32-model;
- adaptive refinement improves cost-quality.

## Next Action

Run the remaining split-aligned official external/stronger baseline and deeper robustness layer: RouteLLM-BERT if a local checkpoint is installed, BEST-Route/RouterDC/MODEL-SAT if local checkpoints/dependencies become available, native exact-command Avengers-Pro utility output only if the upstream CLI is modified or wrapped to emit per-query routing details, tuned or multi-epoch GraphRouter variants only if claim-critical and selected without test leakage, native published-checkpoint or native upstream-metric LLMRouter artifacts if they become necessary beyond the exact CLI selection outputs already postprocessed here, tuned/task-specific encoder families if claim-critical, fuller provider-price schedules if reliable prices can be pinned for more benchmark models, additional held-out transfer checks across broader dataset/model rectangles if new complete-coverage candidates become available, and expanded broad20 split/ablation/transfer sweeps where claim-critical. Keep adaptive refinement deferred unless a future deployable residual-risk signal is substantially stronger. The D4/E5 and bounded F/G pilots keep the calibration, transfer, and robustness claims alive but do not yet prove them.
