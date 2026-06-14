# Broad20 RouteCode Checkpoint Memo

Phase B command set:

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

This is a bounded diagnostic covering one representative scenario for each split family. The lowest rank correlations vs the random split are `leave_dataset_out:aime` at `0.7832` and `cluster_held_out:0` at `0.7904`. Leave-domain-out, domain-homogeneous, and model-pool-holdout rows are more stable in this bounded run, with rank correlations `0.9581`, `0.9759`, and `0.9762`.

RouteCode predicted labels do not reach 80% learned-router gain under the tested split-specific K sweep `[1, 4, 16]`.

## Bounded Ablation Readout

Command:

```bash
python experiments/08_ablation_summary.py --config configs/llmrouterbench_broad20.yaml
```

The broad20 ablation is bounded, not exhaustive. Across the configured K/lambda rows, regret-objective oracle labels have mean recovered gap `0.9360`, flat utility-oracle labels average `0.3155`, semantic KMeans averages `0.0792`, and deployable D2 averages `0.0696`.

D2 seed stability remains narrow but low-ceiling: mean recovered gap is `0.0990` over seeds `[3, 7]`. D2 beta values `[0.0, 1.0]` produce the same recovered gap `0.0906`, so the current rate penalty is not an active lever on this broad run.

## Bounded Transfer Readout

Command:

```bash
python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_broad20.yaml
```

This is a bounded disjoint 8-source/8-target diagnostic over `top_to_next`, `complementary_to_remaining_top`, and `dominated_to_remaining_top` source-pool scenarios. Source/target overlap is `0` in all three scenarios.

Transferred source-D2 labels recover `0.1739` to `0.2274` of the target query-oracle gap. Native target-D2 recovers `0.1682` to `0.2592`. Lightweight same-budget direct retraining baselines are much lower in this bounded run, with recovered-gap range `0.0013` to `0.0394`.

This keeps the transfer claim alive as a diagnostic, but it does not prove paper-level transfer because the direct baseline set is lightweight and the protocol is still bounded.

## Local External-Style Baseline Readout

Command:

```bash
python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_broad20.yaml
```

These are local, split-aligned, no-API surrogates inspired by RouteLLM/LLMRouter-style baselines, not official upstream-command reproductions. The automatic broad20 strong/weak pair is strong `Qwen3-8B`, weak `MiMo-7B-RL-0530`.

The local MF utility surrogate reaches mean utility `0.6934` and recovered gap `-0.0487`, below best-single. Binary logistic strong/weak surrogates range from recovered gap `-0.1560` at threshold `0.25` to `-1.5034` at threshold `0.75`.

Official upstream-command external baselines remain a separate gap.

## RouteLLM MF Official-Code Readout

Command:

```bash
python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_broad20.yaml
```

This trains the local LLMRouterBench RouteLLM MF model class on RouteCode split-aligned pairwise assets and evaluates it on the RouteCode test split. It uses local deterministic RouteCode embeddings and is not the upstream published RouteLLM checkpoint.

Training reaches validation accuracy `0.9106` on `1,230` decisive validation records. The best threshold row is `0.5`, with mean utility `0.7073` and recovered gap `0.0168`. Threshold `0.25` is near best-single with recovered gap `0.0017`; threshold `0.75` falls below best-single with recovered gap `-0.0168`.

The high pairwise accuracy but low routing recovered gap is useful evidence that the broad20 strong/weak pair does not by itself explain much multi-model routing utility.

## Avengers-Pro Split-Aligned Readout

Command:

```bash
python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_broad20.yaml
```

This uses a local implementation of the Avengers-Pro cluster-routing contract with RouteCode deterministic embeddings and the RouteCode train/test split. It is not the upstream command-path run and should be treated as a split-aligned compatibility baseline only.

The simple K=16 cluster row reaches mean utility `0.6574` and recovered gap `-0.2181`; the balance row reaches mean utility `0.6449` and recovered gap `-0.2768`. On broad20, this local cluster-routing adapter is below best-single and below the RouteLLM MF official-code local-embedding row.

## Interpretation

The broad rectangle confirms non-trivial routability and a strong oracle-code rate-distortion curve, especially for regret-objective labels. It also makes the deployability gap sharper: current predicted flat RouteCode labels and simple supervised embedding routers are below best-single on this broad split.

Broad D2 now gives a small deployable gain over best-single but not a strong inferred-label recovery claim. Residual concentration does not justify adaptive refinement. Bounded split sensitivity shows ranking changes for leave-dataset and cluster-held-out scenarios, supporting the benchmark-diagnosis thread. Bounded ablations preserve the oracle-code-versus-deployable-code gap. Bounded transfer rows favor source-D2 label transfer over lightweight direct retraining. Local external-style surrogates and the local Avengers-Pro compatibility baseline are below best-single, and the RouteLLM MF official-code row is only slightly above best-single at its best threshold. Exact upstream external baselines and broader transfer checks remain missing.
