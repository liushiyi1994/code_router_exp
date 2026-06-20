# Baseline Readiness Audit

Source docs:

- `REFERENCES.md`
- `PAPERS_AND_BASELINES.md`
- `Research Flow.md` Phase E

Date: 2026-06-14

This audit records the external baseline repositories inspected before implementing additional Phase E baselines. It now also records official upstream RouteLLM artifact parsing, RouteCode split-aligned RouteLLM pairwise substrate export, RouteLLM MF trainer asset export, split-aligned RouteLLM MF local-code evaluation, local LLMRouter trainer-class adapter metrics, split-aligned local Avengers-Pro cluster-routing compatibility evaluation, and RouteCode utility scoring over routing decisions captured from upstream Avengers-Pro model code. It does not claim the upstream published RouteLLM checkpoint or a full exact upstream Avengers-Pro command-path RouteCode utility row has been run on the RouteCode pilot split.

## Repositories Checked

| Source | Repo | Current HEAD inspected | Local checkout |
|---|---|---:|---|
| LLMRouterBench | `https://github.com/ynulihao/LLMRouterBench` | `c77cb0506949d8f959e97967d2fefca0e8ff1b05` | `data/raw/external/LLMRouterBench` |
| RouteLLM | `https://github.com/lm-sys/routellm` | `0b64fdafe049e596a3f5657c219329f24af24198` | `data/raw/external/routellm` |
| LLMRouter | `https://github.com/ulab-uiuc/LLMRouter` | `c65a32b1435bacdb1488280effef28a6ff89edf6` | `data/raw/external/LLMRouter` |
| RouterBench | `https://github.com/withmartian/routerbench` | `cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c` | `data/raw/external/routerbench` |
| BEST-Route | `https://github.com/microsoft/best-route-llm` | `fc9b896a814acf7c00565fbea0803a8c20812702` | HEAD checked only |
| IRT-Router | `https://github.com/Mercidaiha/IRT-Router` | `e8f258ced4ec3c40d795403603acd8c1cdfb994d` | HEAD checked only |

## LLMRouterBench Baseline Framework

LLMRouterBench is the most direct Phase E integration target because it already ships a unified baseline loader, split logic, and adaptors.

Inspected files:

- `data/raw/external/LLMRouterBench/README.md`
- `data/raw/external/LLMRouterBench/baselines/README.md`
- `data/raw/external/LLMRouterBench/baselines/adaptors/*.py`
- `data/raw/external/LLMRouterBench/baselines/RouteLLM/README.md`
- `data/raw/external/LLMRouterBench/baselines/GraphRouter/README.md`
- `data/raw/external/LLMRouterBench/baselines/AvengersPro/README.md`

Useful built-in/adapted baselines:

- RouterDC
- EmbedLLM
- MODEL-SAT
- Avengers / Avengers-Pro
- HybridLLM
- FrugalGPT
- RouteLLM
- GraphRouter

Important implementation note:

- LLMRouterBench adaptors already enforce prompt-level splitting for many baselines. Any RouteCode adapter should preserve the same no-leakage invariant: all model outcomes for one prompt/query must stay in the same split.

## RouteLLM

Inspected upstream README and router files under `data/raw/external/routellm/routellm`.

Useful methods:

- `mf`
- `bert`
- `sw_ranking`
- `causal_llm`
- `random`

Practicality:

- The LLMRouterBench embedded RouteLLM adaptor is the safest first integration path for RouteLLM-MF.
- Upstream RouteLLM is primarily binary strong/weak routing. A fair adapter for the current 6-model RouteCode pilot should either:
  - evaluate selected strong/weak pairs only; or
  - use LLMRouterBench's prepared RouteLLM baseline outputs/configs where available.
- Upstream MF and similarity-weighted routers may require embeddings and RouteLLM-specific pairwise data assets.

Recommended first RouteLLM action:

1. Treat `results/llmrouterbench_pilot/table_routellm_mf_split_aligned.csv` as the current split-aligned RouteLLM-MF local-code baseline.
2. If exact upstream reproduction is needed, install/pin the full LLMRouterBench baseline environment and rerun the upstream command path on the exported assets.
3. Add RouteLLM-BERT only after local checkpoint/dependency requirements are pinned.
4. Record exact config, model pair, threshold, split, and embedding source.

## LLMRouter

Inspected upstream README and router docs under `data/raw/external/LLMRouter/llmrouter/models`.

Useful methods:

- `knnrouter`
- `svmrouter`
- `mlprouter`
- `mfrouter`
- `elorouter`
- `routerdc`
- `graphrouter`
- `causallm_router`

Practicality:

- kNN, SVM, and MLP are easy to mirror locally because RouteCode already has deterministic local embeddings and train/test query splits. SVM is the main missing simple learned router in the current codebase.
- GraphRouter requires GNN dependencies and query/model graph data. It is better treated as an external adapter after D2 and simple baselines are stable.
- Causal LLM router and Router-R1 are not appropriate for the no-API/no-fine-tuning stage.

Recommended first LLMRouter-aligned action:

1. Add a local SVM embedding router matching the LLMRouter SVM router concept.
2. Add it to `table_recovered_gap.csv`, `table_rate_distortion.csv`, and split sensitivity where feasible.
3. Treat upstream LLMRouter GraphRouter as a later adapter.

## RouterBench

Inspected upstream README under `data/raw/external/routerbench`.

Practicality:

- RouterBench is useful as a secondary benchmark/data source after LLMRouterBench diagnostics stabilize.
- Its pipeline expects processed RouterBench data and optional MongoDB/local caches.
- Do not switch to RouterBench until the D2 method and LLMRouterBench Phase E tables are stable.

## BEST-Route and IRT-Router

Current HEADs resolve, but these were not cloned in this pass.

Practicality:

- BEST-Route is relevant mainly for adaptive compute / model-plus-sampling action spaces, not the current no-adaptive-refinement RouteCode stage.
- IRT-Router is potentially useful for interpretability and difficulty/model-ability comparisons, but optional for the first paper direction.

## Immediate Baseline Implementation Priority

1. Local random router and dataset oracle as named rows, if not already separated in every table.
2. Local SVM embedding router, matching LLMRouter's SVM router concept.
3. D2 predictability-constrained RouteCode after explicit design approval.
4. RouteLLM-MF binary strong/weak pilot via LLMRouterBench adaptor.
5. LLMRouterBench built-in baselines such as AvengersPro / EmbedLLM / GraphRouter only after adapter commands and dependencies are pinned.

## Required Recording for Future Runs

For every external baseline run, record in the run README:

- paper link;
- repo link;
- exact commit hash;
- exact command;
- dataset/config;
- split definition;
- model pool or strong/weak pair;
- hyperparameters;
- hardware/runtime;
- leakage check;
- implementation mismatch from the paper or upstream repo.

## Current Status

Baseline sources have been inspected and pinned. Local internal baselines have been implemented, `experiments/10_external_baseline_surrogates.py` now runs no-API external-style surrogate rows in `results/llmrouterbench_pilot/table_external_baselines.csv`, `experiments/12_official_baseline_artifacts.py` now parses official upstream RouteLLM-MF artifacts into `results/llmrouterbench_pilot/table_official_external_artifacts.csv`, `experiments/14_routellm_pairwise_alignment.py` now exports a RouteCode split-aligned RouteLLM-style pairwise substrate, `experiments/15_routellm_mf_assets.py` now exports official-trainer-compatible MF assets, `experiments/16_routellm_mf_split_aligned.py` now trains/evaluates a split-aligned MF checkpoint using the local LLMRouterBench MF source with RouteCode embeddings, `experiments/27_llmrouter_library_adapters.py` now trains/evaluates local LLMRouter KNN/SVM trainer-class adapters on RouteCode embeddings, `experiments/17_avengerspro_split_aligned.py` now evaluates a split-aligned local implementation of the Avengers-Pro cluster-routing contract with RouteCode embeddings, `experiments/40_avengerspro_upstream_metric.py` now captures routing decisions from upstream Avengers-Pro `SimpleClusterRouter` code and scores them on RouteCode utility, `experiments/24_graphrouter_assets.py` now exports GraphRouter-compatible data-contract assets, and `experiments/21_external_command_readiness.py` now writes a reproducible exact upstream-command readiness table.

The surrogate rows are not official external baseline reproductions. The current surrogate layer includes:

- `routellm_style_mf_utility_router`: low-rank utility-factor router inspired by RouteLLM/EmbedLLM MF;
- `routellm_binary_logistic_surrogate_t*`: RouteLLM-style strong/weak threshold router using local embeddings.

The official artifact inspection layer includes:

- five upstream RouteLLM-MF seed JSON files from `data/raw/external/LLMRouterBench/baselines/RouteLLM/results`;
- upstream per-seed selection-accuracy and total-cost CSVs;
- 70 compatibility-tagged rows, including five overall rows and per-dataset rows;
- explicit `split_aligned_with_routecode = False` and `routecode_metric_compatible = False` flags.

The RouteLLM pairwise substrate layer includes:

- `results/llmrouterbench_pilot/routellm_pairwise/pairwise_train.json` and `pairwise_test.json`;
- metadata with train/test query overlap `0`;
- 1,738 train records and 580 test records for `Qwen3-8B` vs `Qwen2.5-Coder-7B-Instruct`;
- explicit `split_aligned_with_routecode = True`, `official_routellm_result = False`, and `routecode_metric_compatible = False` flags.

The RouteLLM MF trainer asset layer includes:

- `results/llmrouterbench_pilot/routellm_mf_assets/pairwise_train.json` and `pairwise_test.json`;
- `prompt_embeddings.npy`, `prompt_index.json`, and `mf_train_config.local.json`;
- 627 decisive train records, 580 test records, and 2,318 aligned prompt embeddings with dimension 256;
- pair `Qwen3-8B` / `Qwen2.5-Coder-7B-Instruct` present in the official RouteLLM `MODEL_IDS`;
- explicit `split_aligned_with_routecode = True`, `official_trainer_compatible = True`, `official_routellm_result = False`, and `routecode_metric_compatible = False` flags.

The split-aligned RouteLLM MF local-code evaluation layer includes:

- `results/llmrouterbench_pilot/routellm_mf_split_aligned/mf_model.pt`;
- `results/llmrouterbench_pilot/table_routellm_mf_split_aligned.csv`;
- threshold rows for `[0.25, 0.5, 0.75]`;
- best threshold `0.5`, mean utility `0.7259`, recovered gap vs oracle `0.2556`, selection accuracy `0.7259`, and decisive-pair routing accuracy `0.7476`;
- explicit `official_training_code_used = True`, `official_upstream_checkpoint = False`, `split_aligned_with_routecode = True`, and `routecode_metric_compatible = True` flags.

The local LLMRouter trainer-class adapter layer includes:

- `results/llmrouterbench_pilot/table_llmrouter_library_adapters.csv`;
- `results/llmrouterbench_pilot/llmrouter_library_adapters/knn_model.pkl` and `svm_model.pkl`;
- KNN/SVM trainer classes imported from `data/raw/external/LLMRouter`;
- local prediction from saved sklearn artifacts on RouteCode deterministic embeddings, avoiding LLMRouter Longformer/API route methods;
- pilot rows that collapse to best-single: mean utility `0.6672`, recovered gap vs oracle `0.0000`;
- explicit `split_aligned_with_routecode = True`, `routecode_metric_compatible = True`, `upstream_training_class_used = True`, `exact_upstream_command = False`, and `external_api_calls = False` flags.

The split-aligned local Avengers-Pro compatibility layer includes:

- `results/llmrouterbench_pilot/avengerspro_split_aligned/train.jsonl` and `test.jsonl`;
- `results/llmrouterbench_pilot/avengerspro_split_aligned/baseline_scores.json`;
- `results/llmrouterbench_pilot/table_avengerspro_split_aligned.csv`;
- simple K-means and balance-aware K-means cluster-routing rows using train-only cluster rankings and deterministic RouteCode embeddings;
- best row `avengerspro_simple_cluster_k16`, mean utility `0.7397`, recovered gap vs oracle `0.3158`;
- explicit `official_command_path = False`, `official_upstream_checkpoint = False`, `split_aligned_with_routecode = True`, `routecode_metric_compatible = True`, and `no_api_calls = True` flags.

The Avengers-Pro upstream model-code utility layer includes:

- `results/llmrouterbench_pilot/table_avengerspro_upstream_metric.csv`;
- `results/llmrouterbench_pilot/avengerspro_upstream_metric/raw_routing_details.json`;
- `results/llmrouterbench_pilot/avengerspro_upstream_metric/simple_cluster_config.local.json`;
- upstream `SimpleClusterRouter` imported from `data/raw/external/LLMRouterBench/baselines/AvengersPro`;
- local embedding cache and RouteCode train/test assets, with no external API calls;
- pilot row `avengerspro_upstream_simple_cluster_postprocessed`, mean utility `0.7397`, recovered gap vs oracle `0.3158`, upstream accuracy `0.7059`, and `580` test predictions;
- explicit `upstream_model_code_used = True`, `exact_upstream_command = False`, `split_aligned_with_routecode = True`, and `routecode_metric_compatible = True` flags.

The exact upstream Avengers-Pro CLI JSON still omits `routing_details`, so the exact command-path row remains accuracy/cost-only. The upstream model-code utility layer is the current RouteCode-compatible Avengers-Pro evidence, not a native exact CLI utility output.

GraphRouter now has exact upstream command-path smoke metrics for pilot and broad20, plus a split-aligned RouteCode utility adapter that reuses the upstream GraphRouter GNN/model code with RouteCode train/validation/test masks. The generated GraphRouter data-contract assets by themselves are not RouteCode-utility-compatible because the unmodified upstream runner performs its own internal split and reports upstream accuracy/cost metrics, but the runtime blocker has been removed in the current environment and RouteCode-compatible postprocessing now exists through the split-aligned adapter.

The GraphRouter asset layer includes:

- `results/llmrouterbench_pilot/graphrouter_assets/router_data.csv`, `LLM_Descriptions.json`, `llm_description_embedding.pkl`, and `config.local.yaml`;
- `results/llmrouterbench_pilot/table_graphrouter_assets.csv`;
- `results/llmrouterbench_pilot/table_graphrouter_cli_metrics.csv`;
- `results/llmrouterbench_pilot/table_graphrouter_split_aligned.csv`;
- `results/llmrouterbench_pilot/graphrouter_cli_metrics/graphrouter_stdout.log`;
- `results/llmrouterbench_pilot/graphrouter_split_aligned/raw_predictions.json`;
- `2,897` pilot queries, `17,382` query-model rows, and `6` models;
- explicit `split_aligned_with_routecode = True`, `official_graphrouter_result = False`, and `routecode_metric_compatible = False` flags.
- pilot one-epoch exact upstream smoke row: dataset-level accuracy `0.6378`, sample-level accuracy `0.5747`, total cost `1.4794`.
- pilot split-aligned RouteCode utility row: mean utility `0.5897`, recovered gap vs oracle `-0.3383`, `580` test predictions, checkpoint selected on the RouteCode validation split.

The exact upstream-command readiness layer includes:

- `results/llmrouterbench_pilot/table_external_command_readiness.csv`;
- `results/llmrouterbench_pilot/phase_e_external_command_readiness_memo.md`;
- available RouteCode-metric-compatible rows for local RouteLLM-MF, EmbedLLM KNN, FrugalGPT, and upstream-code Avengers-Pro adapters;
- executed exact upstream-command rows include RouteLLM-MF train/eval, LLMRouter KNN/SVM train and route-only inference, Avengers-Pro simple-cluster, GraphRouter one-epoch smoke, FrugalGPT local scorer, EmbedLLM KNN, and EmbedLLM MF;
- upstream RouteLLM-MF train checkpoint evidence: `results/llmrouterbench_pilot/routellm_mf_assets/mf_model.pt`;
- full-split Avengers-Pro exact CLI accuracy/cost evidence: `results/llmrouterbench_pilot/table_avengerspro_cli_metrics.csv` and `results/llmrouterbench_pilot/avengerspro_cli_metrics/simple_cluster_full_results.json`;
- upstream-code Avengers-Pro RouteCode utility evidence: `results/llmrouterbench_pilot/table_avengerspro_upstream_metric.csv` and `results/llmrouterbench_pilot/avengerspro_upstream_metric/raw_routing_details.json`;
- GraphRouter exact CLI smoke evidence: `results/llmrouterbench_pilot/table_graphrouter_cli_metrics.csv` and `results/llmrouterbench_pilot/graphrouter_cli_metrics/graphrouter_stdout.log`;
- `routellm_bert_cli` blocked by a missing local BERT checkpoint;
- `best_route_train_cli`, `routerdc_train_cli`, and `modelsat_train_cli` remain blocked by checkpoint/dependency constraints.

The broad20 exact upstream-command readiness layer now mirrors that checkpoint status on the 18-dataset/20-model rectangle:

- `results/llmrouterbench_broad20/table_external_command_readiness.csv`;
- `results/llmrouterbench_broad20/phase_e_external_command_readiness_memo.md`;
- `results/llmrouterbench_broad20/table_routellm_mf_assets.csv`;
- `results/llmrouterbench_broad20/phase_e_routellm_mf_assets_memo.md`;
- `results/llmrouterbench_broad20/table_graphrouter_assets.csv`;
- `results/llmrouterbench_broad20/phase_e_graphrouter_assets_memo.md`;
- `results/llmrouterbench_broad20/table_graphrouter_cli_metrics.csv`;
- `results/llmrouterbench_broad20/phase_e_graphrouter_cli_metrics_memo.md`;
- `results/llmrouterbench_broad20/table_graphrouter_split_aligned.csv`;
- `results/llmrouterbench_broad20/phase_e_graphrouter_split_aligned_memo.md`;
- `results/llmrouterbench_broad20/table_avengerspro_upstream_metric.csv`;
- `results/llmrouterbench_broad20/phase_e_avengerspro_upstream_metric_memo.md`;
- available RouteCode-metric-compatible rows for local RouteLLM-MF, EmbedLLM KNN, FrugalGPT, and upstream-code Avengers-Pro adapters;
- executed exact upstream-command rows mirror the pilot checkpoint, including full-split Avengers-Pro simple-cluster accuracy/cost evidence at `results/llmrouterbench_broad20/table_avengerspro_cli_metrics.csv`, `results/llmrouterbench_broad20/avengerspro_cli_metrics/simple_cluster_full_results.json`, and GraphRouter smoke evidence at `results/llmrouterbench_broad20/table_graphrouter_cli_metrics.csv`;
- upstream RouteLLM-MF train checkpoint evidence: `results/llmrouterbench_broad20/routellm_mf_assets/mf_model.pt`;
- broad20 upstream-code Avengers-Pro RouteCode utility row: mean utility `0.6567`, recovered gap vs oracle `-0.2215`, upstream accuracy `0.5988`, and `2,808` test predictions.
- broad20 one-epoch GraphRouter exact upstream smoke row: dataset-level accuracy `0.2997`, sample-level accuracy `0.3530`, total cost `0.0187`.
- broad20 split-aligned GraphRouter RouteCode utility row: mean utility `0.3511`, recovered gap vs oracle `-1.6611`, `2,808` test predictions, checkpoint selected on the RouteCode validation split.
- `routellm_bert_cli` remains blocked by a missing local BERT checkpoint; BEST-Route, RouterDC, and MODEL-SAT remain blocked by local checkpoint/dependency constraints recorded in the broad20 readiness table.

Remaining official-baseline priority:

1. RouteLLM-BERT coverage if a local checkpoint is installed.
2. Native exact-command Avengers-Pro RouteCode utility output only if the upstream CLI is changed or wrapped to emit per-query `routing_details`; the current upstream-code wrapper already provides RouteCode-compatible utility evidence.
3. BEST-Route, RouterDC, and MODEL-SAT command-path execution if local checkpoints/dependencies are installed.
