# Broad20 RouteCode Checkpoint Memo

Phase B command set:

```bash
python experiments/00_data_audit.py --config configs/llmrouterbench_broad20.yaml
python experiments/01_compression_ladder.py --config configs/llmrouterbench_broad20.yaml
python experiments/02_rate_distortion_curve.py --config configs/llmrouterbench_broad20.yaml
python experiments/06_predictability_constrained.py --config configs/llmrouterbench_broad20.yaml
python experiments/07_new_model_calibration.py --config configs/llmrouterbench_broad20.yaml
python experiments/03_residual_concentration.py --config configs/llmrouterbench_broad20.yaml
python experiments/04_split_sensitivity.py --config configs/llmrouterbench_broad20.yaml
python experiments/08_ablation_summary.py --config configs/llmrouterbench_broad20.yaml
python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_broad20.yaml
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_broad20.yaml
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_broad20.yaml
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_broad20.yaml
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_broad20.yaml
python experiments/40_avengerspro_upstream_metric.py --config configs/llmrouterbench_broad20.yaml
python experiments/22_cost_quality_frontier.py --config configs/llmrouterbench_broad20.yaml
python experiments/23_stronger_direct_router_probe.py --config configs/llmrouterbench_broad20.yaml
python experiments/25_provider_price_sensitivity.py --config configs/llmrouterbench_broad20.yaml
```

This run evaluates the 18-dataset/20-model complete rectangle identified by the raw LLMRouterBench coverage audit. It uses existing local outcome JSONs only and makes no external API calls.

## Data Rectangle

- Query-model rows: `280,820`.
- Queries: `14,041`.
- Datasets: `18`.
- Models: `20`.
- Train/test query counts: `8,425` / `2,808`.

## B0 Routability

| method | mean_utility | oracle_regret | recovered_gap_vs_oracle |
| --- | --- | --- | --- |
| best_single | 0.7037 | 0.2123 | 0.0000 |
| dataset_label_lookup | 0.7172 | 0.1987 | 0.0638 |
| query_oracle | 0.9160 | 0.0000 | 1.0000 |

## B1 Compression Readout

| method | K | mean_utility | recovered_gap_vs_oracle |
| --- | --- | --- | --- |
| embedding_cluster_lookup | 16 | 0.7222 | 0.0872 |
| kNN |  | 0.7023 | -0.0067 |
| routecode_oracle_labels | 16 | 0.7813 | 0.3658 |
| routecode_predicted_labels | 16 | 0.6022 | -0.4782 |
| routecode_mlp_predicted_labels | 16 | 0.6065 | -0.4581 |

## B2 Rate-Distortion Readout

Utility-vector RouteCode oracle labels recover `0.3658` of the query-oracle gap at K=16, `0.5151` at K=32, `0.7064` at K=64, and `0.7903` at K=128.

Regret-objective RouteCode oracle labels recover `0.6510` at K=4, `0.8540` at K=8, `0.9681` at K=16, and reach the query oracle by K=32.

Semantic embedding KMeans remains low in this broad rectangle: recovered gap is `0.0872` at K=16 and `0.0889` at K=32.

## D2 Predictability-Constrained Readout

Command:

```bash
python experiments/06_predictability_constrained.py --config configs/llmrouterbench_broad20.yaml
```

Best deployable D2 row in this sweep is `d2_embedding_centroid` at alpha `1`: mean utility `0.7229`, recovered gap `0.0906`, label accuracy `0.7646`, and empirical label entropy `3.6436`.

D2 improves sharply over flat RouteCode logistic prediction at K=16 (`0.6022` mean utility, recovered gap `-0.4782`), but it is only slightly above embedding KMeans (`0.7222`, recovered gap `0.0872`) and well below the broad query oracle (`0.9160`).

The D2 joint-label oracle has a utility/predictability tradeoff: alpha `0` reaches mean utility `0.7899` and recovered gap `0.4060`, while alpha `1` gives deployable centroid routing near `0.7229` but joint-oracle recovered gap only `0.1124`.

## All-Model New-Model Calibration Readout

Command:

```bash
python experiments/07_new_model_calibration.py --config configs/llmrouterbench_broad20.yaml
```

This expanded run holds out every model in the broad20 rectangle as a simulated new model: `20` held-out models, `620` total rows, and `140` RouteCode label-calibration rows over r `[1, 2, 4, 8, 16, 32, 64]`. It uses D2 labels with K=16, alpha=1, beta=0, and compares label calibration to same-budget logistic, SVM, and kNN direct retraining. It makes no external API calls. To keep the all-model sweep practical, direct logistic uses the configured `saga` solver and direct SVM uses the configured SGD linear-SVM backend with max_iter `100`; the separate stronger-direct probe still covers MLP and gradient boosting on a bounded two-model slice.

Mean across all `20` held-out models: RouteCode label calibration reaches mean utility `0.7183` and recovered gap `0.0687` at r=64, using about `1008.6` new-model evaluations. The strongest lightweight direct retraining mean row is kNN at r=32/r=64 with mean utility about `0.5388` and recovered gap about `-0.777`. The best individual budgeted row is RouteCode label calibration for `DeepHermes-3-Llama-3-8B-Preview` at r=16, mean utility `0.7272`, recovered gap `0.1107`, and `256` new-model evaluations. This strengthens the sample-efficiency diagnostic on the broad rectangle, but it still does not cover transformer direct routers or official external routers.

## Stronger Direct-Router Probe Readout

Command:

```bash
python experiments/23_stronger_direct_router_probe.py --config configs/llmrouterbench_broad20.yaml
```

This is a bounded two-held-out-model probe over r `[8, 64]`. It complements the all-model logistic/SVM/kNN calibration sweep by separately comparing RouteCode label calibration to logistic, SVM, kNN, MLP, and gradient-boosting direct retraining under the same sampled new-model budgets.

Mean across the probe rows: RouteCode label calibration reaches mean utility `0.7158` and recovered gap `0.0570` at r=8, and mean utility `0.7167` with recovered gap `0.0612` at r=64. The best stronger direct-router average is logistic at r=64 with mean utility `0.5504` and recovered gap `-0.7223`; MLP at r=64 reaches mean utility `0.5486` and recovered gap `-0.7307`. This extends the direct-router comparison beyond logistic/SVM/kNN but remains a bounded probe, not a full calibration proof.

## Residual Gate Readout

Command:

```bash
python experiments/03_residual_concentration.py --config configs/llmrouterbench_broad20.yaml
```

Residual regret is moderately concentrated: top 5%, 10%, and 20% oracle-sorted residual queries account for `0.1600`, `0.3190`, and `0.6379` of total residual regret.

Deployable residual-risk signals are weak. The best deployable signal is low route-label confidence, capturing only `0.0556` of regret in the top 5% flagged queries and `0.1056` in the top 10%, with AUC `0.5370`.

Adaptive refinement remains deferred on broad20.

## Split Sensitivity Readout

Command:

```bash
python experiments/04_split_sensitivity.py --config configs/llmrouterbench_broad20.yaml
```

This expanded diagnostic now covers `65` split scenarios and `520` method rows: random, all `18` leave-dataset-out cases, all `11` leave-domain-out cases, all `11` domain-homogeneous cases, all `4` configured cluster-held-out cases, and all `20` single-model model-pool-holdout cases. The lowest rank correlations vs the random split are `leave_dataset_out:mbpp` at `0.1198`, `leave_dataset_out:humaneval` at `0.4551`, `cluster_held_out:1` at `0.4791`, `leave_domain_out:dialogue` at `0.4791`, and `leave_dataset_out:emorynlp` at `0.6988`.

Dataset/domain/cluster splits create much larger method-ranking changes than model-pool holdout splits, supporting the benchmark-diagnosis thread. The expanded completed method table covers all broad20 single-model pool holdouts. The split-rate threshold table now also covers all `65` scenarios with the full RouteCode K ladder `[1, 2, 4, 8, 16, 32, 64, 128]`, using checkpointed partial writes, rate-sweep fit controls, and a rate-only resume path. The diagnostic logistic router emitted nonfatal scikit-learn convergence warnings.

RouteCode predicted labels reach 80% learned-router gain in only `1/65` full-ladder split scenarios: `leave_dataset_out:mbpp`, at `rate_log2K_to_80pct_learned_gain = 5.0`. That row has best RouteCode predicted-label recovered gap `2.8333` vs learned and `0.0447` vs oracle, so the threshold result is a split-diagnostic outlier rather than broad evidence of deployable recovery.

## Expanded Bounded Ablation Readout

Command:

```bash
python experiments/08_ablation_summary.py --config configs/llmrouterbench_broad20.yaml
```

The broad20 ablation is expanded but still bounded, not exhaustive. The regenerated table has `125` rows: `100` K/lambda rows, `15` seed-stability rows, `6` training-fraction rows, and `4` D2 rate-penalty rows. Across the configured K/lambda rows, regret-objective oracle labels have mean recovered gap `0.9126`, flat utility-oracle labels average `0.4094`, semantic KMeans averages `0.0592`, and deployable D2 averages `0.0481`.

D2 seed stability remains narrow but low-ceiling: mean recovered gap is `0.1046` over seeds `[3, 7, 11]`. D2 beta values `[0.0, 0.1, 1.0, 3.0]` produce recovered gaps from `0.0772` to `0.0906`, so the current rate penalty is only a weak lever on this broad run.

## Bounded Phase G Sensitivity Readout

Command:

```bash
python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_broad20.yaml
```

This run adds 226 broad20 sensitivity rows over embedding feature variants, clustering algorithms, label noise, cost mis-estimation, price-ratio stress, model-pool subset/composition, automatic dominated/complementary pools, domain granularity, query-length buckets, and bootstrap counts.

Mean D2 recovered gap remains small but positive across most global sensitivity families: `0.0850` for embedding-backbone variants, `0.1159` under cost mis-estimation, `0.0872` under price-ratio stress, `0.0797` across broad model-pool subsets, and `0.0627` across automatic dominated/complementary model pools. Domain and length buckets expose heterogeneity: D2 recovered gap ranges from `-0.0727` to `0.5000` across domain-granularity buckets and from `-0.0335` to `0.2263` across query-length buckets.

This strengthens Phase G coverage on the broad rectangle, but it remains bounded. External pretrained embedding backbones, exact upstream-command external baselines, and exhaustive split/transfer sweeps remain open.

## Model-Pool Scale Readout

Command:

```bash
python experiments/18_model_pool_scale.py --config configs/llmrouterbench_broad20.yaml
```

This adds 48 rows over train-only top, complementary, dominated, and full model-pool scenarios with sizes `[2, 4, 8, 12, 16, 20]`.

D2 recovered gap ranges from `0.0000` to `0.0980`. The best D2 row is `complementary_12`, with mean utility `0.7229`, recovered gap `0.0980`, test oracle gap `0.1962`, and test dominance ratio `0.8038`. On the full 20-model broad pool, D2 reaches mean utility `0.7158` and recovered gap `0.0570`; kNN is slightly below best-single with recovered gap `-0.0067`.

This reinforces model-pool composition as a diagnostic: broad20 is routable, but deployable compressed routing gains stay modest and depend on pool construction.

## Bounded Transfer Readout

Command:

```bash
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_broad20.yaml
```

This expanded local transfer diagnostic covers `18` disjoint source/target scenarios and `162` method rows over source/target size pairs `4x4`, `4x8`, `8x4`, `8x8`, `12x4`, and `12x8`. Source/target overlap is `0` in all scenarios.

Transferred source-D2 labels recover `0.0584` to `0.2592` of the target query-oracle gap, with mean `0.1858`. Native target-D2 recovers `0.0730` to `0.2668`, with mean `0.1965`. Same-budget direct retraining baselines over logistic, SVM, kNN, MLP, and gradient boosting are much lower in this local run, with recovered-gap range `-0.0511` to `0.0738` and mean about `0.0130`.

This keeps the transfer claim alive as a diagnostic, but it does not prove paper-level transfer because the protocol is still bounded and does not cover upstream transformer direct routers.

## Local External-Style Baseline Readout

Command:

```bash
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_broad20.yaml
```

These are local, split-aligned, no-API surrogates inspired by RouteLLM/LLMRouter-style baselines, not official upstream-command reproductions. The automatic broad20 strong/weak pair is strong `Qwen3-8B`, weak `MiMo-7B-RL-0530`.

The local MF utility surrogate reaches mean utility `0.6934` and recovered gap `-0.0487`, below best-single. Binary logistic strong/weak surrogates range from recovered gap `-0.1560` at threshold `0.25` to `-1.5034` at threshold `0.75`.

Official upstream-command external baselines remain a separate gap.

## LLMRouter Library Adapter Readout

Command:

```bash
python experiments/27_llmrouter_library_adapters.py --config configs/llmrouterbench_broad20.yaml
```

This trains local LLMRouter KNN/SVM trainer classes on RouteCode deterministic embeddings and evaluates the saved sklearn artifacts on the RouteCode test split. It does not call LLMRouter route methods, does not compute Longformer embeddings, and is not an exact upstream command-path run.

The LLMRouter KNN adapter reaches mean utility `0.5214` and recovered gap `-0.8591`; the SVM adapter reaches mean utility `0.4833` and recovered gap `-1.0386`. These broad20 rows are below best-single and should be treated as compatibility evidence, not competitive external-baseline evidence.

## RouteLLM MF Official-Code Readout

Command:

```bash
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_broad20.yaml
```

This trains the local LLMRouterBench RouteLLM MF model class on RouteCode split-aligned pairwise assets and evaluates it on the RouteCode test split. It uses local deterministic RouteCode embeddings and is not the upstream published RouteLLM checkpoint.

Training reaches validation accuracy `0.9106` on `1,230` decisive validation records. The best threshold row is `0.5`, with mean utility `0.7073` and recovered gap `0.0168`. Threshold `0.25` is near best-single with recovered gap `0.0017`; threshold `0.75` falls below best-single with recovered gap `-0.0168`.

The high pairwise accuracy but low routing recovered gap is useful evidence that the broad20 strong/weak pair does not by itself explain much multi-model routing utility.

The exact LLMRouterBench RouteLLM-MF training CLI has also executed on broad20 split-aligned trainer assets without API calls and wrote `results/llmrouterbench_broad20/routellm_mf_assets/mf_model.pt`. The exact upstream MF evaluation path remains blocked by its `HUOSHAN_API_KEY` embedding-service config, and RouteLLM-BERT remains blocked by a missing local checkpoint.

## Avengers-Pro Split-Aligned Readout

Command:

```bash
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_broad20.yaml
```

This uses a local implementation of the Avengers-Pro cluster-routing contract with RouteCode deterministic embeddings and the RouteCode train/test split. It is not the upstream command-path run and should be treated as a split-aligned compatibility baseline only.

The simple K=16 cluster row reaches mean utility `0.6574` and recovered gap `-0.2181`; the balance row reaches mean utility `0.6449` and recovered gap `-0.2768`. On broad20, this local cluster-routing adapter is below best-single and below the RouteLLM MF official-code local-embedding row.

## Avengers-Pro Upstream-Code Utility Readout

Command:

```bash
python experiments/40_avengerspro_upstream_metric.py --config configs/llmrouterbench_broad20.yaml
```

This calls the upstream Avengers-Pro `SimpleClusterRouter` class on RouteCode split-aligned assets with a local embedding cache, captures `routing_details`, and scores selected models with RouteCode test-split utility. It is not an exact upstream command output because the exact CLI JSON omits per-query routing details.

The upstream-code row reaches mean utility `0.6567`, recovered gap `-0.2215`, upstream accuracy `0.5988`, and `2,808` test predictions. This is RouteCode-metric-compatible evidence over upstream model code, but it remains below best-single on broad20.

## Cost-Quality Operating-Point Readout

Command:

```bash
python experiments/22_cost_quality_frontier.py --config configs/llmrouterbench_broad20.yaml
```

This adds fixed-quality and fixed-cost operating-point diagnostics over lambda values `[0.0, 0.05, 0.1, 0.2]` using released benchmark cost metadata. It writes `table_cost_quality_summary.csv`, `table_cost_quality_frontier.csv`, `fig_cost_quality_frontier.pdf`, and `phase_e_cost_quality_memo.md`.

Broad20 deployable frontier rows show that no deployable method reaches the tested high fixed-quality targets, which are fractions of the oracle-level method quality. Under fixed-cost budgets, deployable winners are kNN, D2, embedding-cluster lookup, or cheapest depending on the budget and lambda. At lambda `0.0`, D2 is the deployable winner at the largest tested budget with mean quality `0.7229` and mean cost `0.0004`; kNN is the deployable winner at the next lower budget with mean quality `0.7023` and mean cost `0.0004`.

These are benchmark-metadata operating points, not provider-price claims, because the released LLMRouterBench metadata assigns zero cost to some local/open model rows.

## Provider-Price Sensitivity Readout

Command:

```bash
python experiments/25_provider_price_sensitivity.py --config configs/llmrouterbench_broad20.yaml
```

This adds a partial OpenRouter price snapshot checked on `2026-06-15` for `Qwen3-8B` and `Llama-3.1-8B-Instruct`. It covers `2/20` broad20 models, recomputes token costs from static input/output prices, and writes `table_provider_price_schedule.csv`, `table_provider_cost_quality_summary.csv`, `table_provider_cost_quality_frontier.csv`, `fig_provider_price_sensitivity.pdf`, and `phase_g_provider_pricing_memo.md`.

At lambda `0.0` on this two-model provider-priced subset, best-single mean quality is `0.7037`, embedding-cluster lookup is `0.7101`, D2 is `0.7076`, and the query oracle is `0.7710`. These rows are provider-price sensitivity diagnostics, not full provider-cost claims.

## Interpretation

The broad rectangle confirms non-trivial routability and a strong oracle-code rate-distortion curve, especially for regret-objective labels. It also makes the deployability gap sharper: current predicted flat RouteCode labels and simple supervised embedding routers are below best-single on this broad split.

Broad D2 now gives a small deployable gain over best-single but not a strong inferred-label recovery claim. Broad all-model new-model calibration favors RouteCode label calibration over lightweight same-budget direct retraining, and the bounded stronger-direct probe keeps that pattern when MLP and gradient-boosting direct routers are added for two held-out models; this is still not paper-level calibration evidence. Residual concentration does not justify adaptive refinement. Expanded split sensitivity shows ranking changes for leave-dataset and cluster-held-out scenarios, supporting the benchmark-diagnosis thread while now covering all `20` broad20 single-model pool holdouts. Expanded bounded ablations, the broad sensitivity suite, cost-quality operating-point rows, and partial provider-price rows preserve the oracle-code-versus-deployable-code gap and expose domain/query/model-pool/cost heterogeneity. The provider-priced rows remain sensitivity diagnostics rather than full provider-cost evidence. Expanded local transfer rows favor source-D2 label transfer over lightweight direct retraining across 18 disjoint broad20 source/target pool scenarios. Local external-style surrogates, local LLMRouter trainer-class adapters, the local and upstream-code Avengers-Pro rows, and the split-aligned GraphRouter GNN adapter are below best-single, while the RouteLLM MF official-code row is only slightly above best-single at its best threshold. Exact RouteLLM-MF train/eval CLI evidence, exact LLMRouter command-path route-only outputs, exact GraphRouter one-epoch upstream accuracy/cost metrics, split-aligned GraphRouter RouteCode utility metrics, exact Avengers-Pro command-path accuracy/cost rows, and upstream-code Avengers-Pro RouteCode utility rows now exist; RouteLLM-BERT and BEST-Route/RouterDC/MODEL-SAT remain blocked by local checkpoint/dependency requirements.
