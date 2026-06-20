from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.codes.code_cards import write_code_cards, write_code_cards_json
from routecode.codes.regret import RegretOptimizedRouteCode
from routecode.codes.routecode import RouteCodeCodebook
from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_code_label_heatmap, save_rate_distortion
from routecode.predictors.classifiers import (
    LogisticModelRouter,
    MLPModelRouter,
    MLPRouteCodeLabelClassifier,
    RouteCodeLabelClassifier,
    SVMModelRouter,
)
from routecode.routers.cluster_lookup import EmbeddingClusterRouter
from routecode.routers.dataset_lookup import DatasetOracleRouter
from routecode.routers.knn import KNNRouter
from routecode.routers.oracle import OracleRouter
from routecode.routers.random import RandomRouter
from routecode.routers.single_best import BestSingleRouter


REFERENCE_LINKS = [
    ("LLMRouterBench", "https://arxiv.org/abs/2601.07206", "https://github.com/ynulihao/LLMRouterBench"),
    ("RouteLLM", "https://arxiv.org/abs/2406.18665", "https://github.com/lm-sys/routellm"),
    ("LLMRouter", "", "https://github.com/ulab-uiuc/LLMRouter"),
    ("RouterBench", "https://arxiv.org/abs/2403.12031", "https://github.com/withmartian/routerbench"),
    ("WebRouter", "https://arxiv.org/abs/2510.11221", ""),
    ("FineRouter", "https://arxiv.org/abs/2603.19415", ""),
    ("BEST-Route", "https://openreview.net/forum?id=tFBIbCVXkG", "https://github.com/microsoft/best-route-llm"),
    ("GraphRouter", "https://openreview.net/forum?id=eU39PDsZtT", "https://github.com/ulab-uiuc/LLMRouter"),
    ("Universal Model Routing", "https://openreview.net/pdf?id=ka82fvJ5f1", ""),
    ("kNN routing", "https://arxiv.org/abs/2505.12601", "https://github.com/ulab-uiuc/LLMRouter"),
    ("Causal LLM Routing", "https://openreview.net/forum?id=iZC5xoQQkX", ""),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]
    embeddings = prepared.embeddings
    seed = int(config.get("run", {}).get("random_seed", 0))
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))
    route_config = config.get("routecode", {})
    k_values = [int(k) for k in route_config.get("k_values", [1, 2, 4, 8, 16, 32])]

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = selected_values(test.utility, best_single).mean()
    random_selected = RandomRouter(random_state=seed).fit(train.query_info, train.utility).predict(test.query_info)
    oracle_selected = OracleRouter().predict(test.utility)
    oracle_mean = test.utility.max(axis=1).mean()
    dataset_oracle_selected = DatasetOracleRouter("dataset").fit(test.query_info, test.utility).predict(test.query_info)
    knn_selected = KNNRouter(int(config.get("routers", {}).get("knn_k", 15))).fit(
        train.query_info,
        train.utility,
        embeddings,
    ).predict(test.query_info, embeddings)
    logistic_selected = LogisticModelRouter(random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    ).predict(test.query_info, embeddings)
    mlp_selected = MLPModelRouter(random_state=seed, hidden_layer_sizes=(32,), max_iter=1000).fit(
        train.query_info,
        train.utility,
        embeddings,
    ).predict(test.query_info, embeddings)
    svm_selected = SVMModelRouter(random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    ).predict(test.query_info, embeddings)
    learned_reference_mean = max(
        selected_values(test.utility, knn_selected).mean(),
        selected_values(test.utility, logistic_selected).mean(),
        selected_values(test.utility, mlp_selected).mean(),
        selected_values(test.utility, svm_selected).mean(),
    )

    rows = [
        evaluate_selection(
            method="random",
            selected_models=random_selected,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            k=len(train.model_ids),
            labels=random_selected,
        ),
        evaluate_selection(
            method="best_single",
            selected_models=best_single,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        ),
        evaluate_selection(
            method="dataset_oracle",
            selected_models=dataset_oracle_selected,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            k=int(test.query_info["dataset"].nunique()) if "dataset" in test.query_info.columns else None,
            labels=test.query_info["dataset"] if "dataset" in test.query_info.columns else None,
        ),
        evaluate_selection(
            method="kNN",
            selected_models=knn_selected,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        ),
        evaluate_selection(
            method="logistic_embedding_router",
            selected_models=logistic_selected,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        ),
        evaluate_selection(
            method="mlp_embedding_router",
            selected_models=mlp_selected,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        ),
        evaluate_selection(
            method="svm_embedding_router",
            selected_models=svm_selected,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        ),
        evaluate_selection(
            method="query_oracle",
            selected_models=oracle_selected,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        ),
    ]

    last_routecode: RouteCodeCodebook | None = None
    selected_k = int(route_config.get("selected_k_for_cards", k_values[-1]))
    for k in k_values:
        semantic_router = EmbeddingClusterRouter(k, random_state=seed).fit(train.query_info, train.utility, embeddings)
        semantic_labels = semantic_router.predict_labels(embeddings.loc[test.utility.index])
        semantic_selected = semantic_router.predict(test.query_info, embeddings)
        rows.append(
            evaluate_selection(
                method="semantic_embedding_kmeans",
                selected_models=semantic_selected,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + k,
                k=k,
                labels=semantic_labels,
            )
        )

        routecode = RouteCodeCodebook(k, random_state=seed, max_iter=int(route_config.get("max_iter", 25))).fit(
            train.query_info,
            train.utility,
            embeddings,
        )
        route_labels = routecode.predict_utility_labels(test.utility)
        route_selected = routecode.predict_from_labels(route_labels)
        rows.append(
            evaluate_selection(
                method="routecode_oracle_labels",
                selected_models=route_selected,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 100 + k,
                k=k,
                labels=route_labels,
            )
        )
        regret_routecode = RegretOptimizedRouteCode(
            k,
            random_state=seed,
            max_iter=int(route_config.get("max_iter", 25)),
        ).fit(
            train.query_info,
            train.utility,
            embeddings,
        )
        regret_labels = regret_routecode.predict_utility_labels(test.utility)
        regret_selected = regret_routecode.predict_from_labels(regret_labels)
        rows.append(
            evaluate_selection(
                method="regret_routecode_oracle_labels",
                selected_models=regret_selected,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 150 + k,
                k=k,
                labels=regret_labels,
            )
        )
        predicted_regret_labels = regret_routecode.predict_labels(embeddings.loc[test.utility.index])
        predicted_regret_selected = regret_routecode.predict_from_labels(predicted_regret_labels)
        rows.append(
            evaluate_selection(
                method="regret_routecode_predicted_labels",
                selected_models=predicted_regret_selected,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 250 + k,
                k=k,
                labels=predicted_regret_labels,
            )
        )
        routecode_classifier = RouteCodeLabelClassifier(random_state=seed).fit(routecode, embeddings)
        predicted_route_labels = routecode_classifier.predict_labels(embeddings.loc[test.utility.index])
        predicted_route_selected = routecode_classifier.predict(test.query_info, embeddings)
        rows.append(
            evaluate_selection(
                method="routecode_predicted_labels",
                selected_models=predicted_route_selected,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 200 + k,
                k=k,
                labels=predicted_route_labels,
            )
        )
        mlp_routecode_classifier = MLPRouteCodeLabelClassifier(
            random_state=seed,
            hidden_layer_sizes=(8,),
            max_iter=2000,
        ).fit(routecode, embeddings)
        mlp_predicted_route_labels = mlp_routecode_classifier.predict_labels(embeddings.loc[test.utility.index])
        mlp_predicted_route_selected = mlp_routecode_classifier.predict(test.query_info, embeddings)
        rows.append(
            evaluate_selection(
                method="routecode_mlp_predicted_labels",
                selected_models=mlp_predicted_route_selected,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 300 + k,
                k=k,
                labels=mlp_predicted_route_labels,
            )
        )
        if k == selected_k:
            last_routecode = routecode

    if last_routecode is None:
        last_routecode = routecode

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_rate_distortion.csv", index=False)
    save_rate_distortion(table, out_dir / "fig_rate_distortion.pdf")
    write_code_cards(str(out_dir / "code_cards.md"), last_routecode, train.query_info, train.utility)
    write_code_cards_json(out_dir / "code_cards.json", last_routecode, train.query_info, train.utility)
    if last_routecode.label_utility_ is not None:
        save_code_label_heatmap(last_routecode.label_utility_, out_dir / "fig_code_label_heatmap.pdf")
    write_readme(out_dir, config, table, args.config)
    print(f"Wrote rate-distortion outputs to {out_dir}")


def write_readme(out_dir: Path, config: dict, table: pd.DataFrame, config_path: str) -> None:
    source = config.get("data", {}).get("source", "synthetic")
    title = "RouteCode Synthetic Demo" if source == "synthetic" else "RouteCode LLMRouterBench Pilot"
    best_rows = table[
        table["method"].isin(
            [
                "random",
                "best_single",
                "kNN",
                "svm_embedding_router",
                "dataset_oracle",
                "query_oracle",
            ]
        )
    ]
    route_rows = table[table["method"] == "routecode_predicted_labels"].sort_values("K")
    oracle_route_rows = table[table["method"] == "routecode_oracle_labels"].sort_values("K")
    regret_oracle_rows = table[table["method"] == "regret_routecode_oracle_labels"].sort_values("K")
    lines = [
        f"# {title}",
        "",
        _readme_intro(config),
        "",
        "## Commands",
        "",
        "```bash",
        f"python experiments/00_data_audit.py --config {config_path}",
        f"python experiments/01_compression_ladder.py --config {config_path}",
        f"python experiments/02_rate_distortion_curve.py --config {config_path}",
        *_extra_commands(config, config_path),
        "pytest -q",
        "```",
        "",
        "## Outputs",
        "",
        "- `table_routability.csv`: best single, cheapest, dataset-label lookup, and query-oracle audit.",
        "- `table_recovered_gap.csv`: compression ladder with bootstrap confidence intervals.",
        "- `table_rate_distortion.csv`: semantic-cluster and RouteCode curves for K = "
        + ", ".join(str(k) for k in config.get("routecode", {}).get("k_values", []))
        + ".",
        "- `code_cards.md`, `code_cards.json`, and `fig_code_label_heatmap.pdf`: train-set summaries for learned route labels.",
        "- `fig_compression_ladder.pdf` and `fig_rate_distortion.pdf`: main pilot figures.",
        *_extra_outputs(config),
        "- `outcomes.csv` and `query_embeddings.csv`: canonical input rows and deterministic local query features used for this run.",
        "",
        "## First Results",
        "",
        _markdown_table(best_rows[["method", "mean_utility", "oracle_regret", "recovered_gap_vs_oracle"]]),
        "",
        "Utility-oracle RouteCode rows:",
        "",
        _markdown_table(
            oracle_route_rows[
                ["K", "rate_log2K", "empirical_H_Z", "mean_utility", "oracle_regret", "recovered_gap_vs_oracle"]
            ]
        ),
        "",
        "Regret-objective RouteCode oracle rows:",
        "",
        _markdown_table(
            regret_oracle_rows[
                ["K", "rate_log2K", "empirical_H_Z", "mean_utility", "oracle_regret", "recovered_gap_vs_oracle"]
            ]
        ),
        "",
        "Predicted RouteCode rows:",
        "",
        _markdown_table(
            route_rows[["K", "rate_log2K", "empirical_H_Z", "mean_utility", "oracle_regret", "recovered_gap_vs_oracle"]]
        ),
        "",
        _claim_caution(config),
        "",
        "## External References Checked",
        "",
        _reference_intro(config),
        "",
    ]
    for name, paper, repo in REFERENCE_LINKS:
        parts = [f"- {name}"]
        if paper:
            parts.append(f"paper: {paper}")
        if repo:
            parts.append(f"repo: {repo}")
        lines.append("; ".join(parts))
    lines.extend(
        [
            "",
            "## Leakage Controls",
            "",
            "- Train/validation/test splits are assigned by `query_id`; all model rows for a query stay in the same split.",
            "- Best-single, dataset/topic tables, embedding clusters, kNN neighbors, and RouteCode codebooks are fit on train only.",
            "- Query oracle uses test utility only as an upper bound.",
            "- The leaky dataset-label diagnostic is written separately to `table_leakage_gap.csv` and is not a deployable baseline.",
            "",
            "## Next Steps",
            "",
            *_next_steps(config),
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_markdown_cell(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _readme_intro(config: dict) -> str:
    source = config.get("data", {}).get("source", "synthetic")
    if source == "synthetic":
        return (
            "This run is a zero-API synthetic pilot. It is intended to verify code paths, "
            "leakage controls, metrics, and plot generation before any real benchmark download."
        )
    return (
        f"This run uses `{source}` records converted into the RouteCode canonical schema. "
        "No model generation or external API calls are made by these evaluation scripts."
    )


def _claim_caution(config: dict) -> str:
    if config.get("data", {}).get("source", "synthetic") == "synthetic":
        return "These values come from synthetic data only and should not be used as novelty or paper-evidence claims."
    return (
        "These values come from a configured LLMRouterBench pilot subset. Treat them as pilot observations, "
        "not full-benchmark or paper-level claims."
    )


def _reference_intro(config: dict) -> str:
    if config.get("data", {}).get("source", "synthetic") == "synthetic":
        return (
            "The synthetic run did not execute external repositories or download benchmark data. "
            "These papers/repos were inspected as novelty boundaries and future baseline sources:"
        )
    return (
        "This run used released LLMRouterBench outcome JSONs and did not call external model APIs. "
        "These papers/repos define the data source, novelty boundaries, and future baseline sources:"
    )


def _next_steps(config: dict) -> list[str]:
    if config.get("data", {}).get("source", "synthetic") == "synthetic":
        return [
            "1. Add real loaders only after this synthetic path remains green.",
            "2. Convert LLMRouterBench outcomes to the canonical schema.",
            "3. Add real-benchmark loaders and keep logistic/MLP learned-router rows as first internal baselines.",
            "4. Run split robustness before making benchmark compressibility claims.",
            "5. Add held-out model calibration simulation for the sample-efficiency claim.",
        ]
    return [
        "1. Broaden real-data domain metadata beyond the current coarse dataset-to-domain map.",
        "2. Run held-out model-pool transfer checks and direct-router retraining comparisons before making transfer claims.",
        "3. Test whether residual predictors such as centroid distance, margin, or kNN disagreement identify the high-regret tail.",
        "4. Add official external baselines or stronger local adapters before making method-ranking claims.",
        "5. Keep final claims scoped to pilot evidence until broader robustness checks pass.",
    ]


def _extra_commands(config: dict, config_path: str) -> list[str]:
    if config.get("data", {}).get("source", "synthetic") == "synthetic":
        return [
            f"python experiments/06_predictability_constrained.py --config {config_path}",
            f"python experiments/07_new_model_calibration.py --config {config_path}",
            f"python experiments/08_ablation_summary.py --config {config_path}",
            f"python experiments/09_sensitivity_suite.py --config {config_path}",
            f"python experiments/10_external_baseline_surrogates.py --config {config_path}",
            f"python experiments/11_code_card_interpretability.py --config {config_path}",
        ]
    return [
        f"python experiments/03_residual_concentration.py --config {config_path}",
        f"python experiments/04_split_sensitivity.py --config {config_path}",
        f"python experiments/05_predictor_diagnostics.py --config {config_path}",
        f"python experiments/06_predictability_constrained.py --config {config_path}",
        f"python experiments/07_new_model_calibration.py --config {config_path}",
        f"python experiments/08_ablation_summary.py --config {config_path}",
        f"python experiments/09_sensitivity_suite.py --config {config_path}",
        f"python experiments/10_external_baseline_surrogates.py --config {config_path}",
        f"python experiments/11_code_card_interpretability.py --config {config_path}",
        f"python experiments/12_official_baseline_artifacts.py --config {config_path}",
        f"python experiments/13_transformer_backbone_readiness.py --config {config_path}",
        f"python experiments/28_transformer_embedding_router.py --config {config_path}",
        f"python experiments/29_embedllm_knn_split_aligned.py --config {config_path}",
        f"python experiments/30_frugalgpt_split_aligned.py --config {config_path}",
        f"python experiments/14_routellm_pairwise_alignment.py --config {config_path}",
        f"python experiments/15_routellm_mf_assets.py --config {config_path}",
        f"python experiments/16_routellm_mf_split_aligned.py --config {config_path}",
        f"python experiments/17_avengerspro_split_aligned.py --config {config_path}",
        f"python experiments/37_avengerspro_cli_metrics.py --config {config_path}",
        f"python experiments/40_avengerspro_upstream_metric.py --config {config_path}",
        f"python experiments/38_graphrouter_cli_metrics.py --config {config_path}",
        f"python experiments/39_graphrouter_split_aligned.py --config {config_path}",
        f"python experiments/18_model_pool_scale.py --config {config_path}",
        f"python experiments/19_model_pool_transfer.py --config {config_path}",
        f"python experiments/20_benchmark_coverage.py --config {config_path}",
        f"python experiments/21_external_command_readiness.py --config {config_path}",
        f"python experiments/22_cost_quality_frontier.py --config {config_path}",
        f"python experiments/23_stronger_direct_router_probe.py --config {config_path}",
        f"python experiments/26_external_baseline_assets.py --config {config_path}",
        f"python experiments/27_llmrouter_library_adapters.py --config {config_path}",
        f"python experiments/25_provider_price_sensitivity.py --config {config_path}",
    ]


def _extra_outputs(config: dict) -> list[str]:
    if config.get("data", {}).get("source", "synthetic") == "synthetic":
        return [
            "- `table_predictability_constrained.csv`, `fig_predictability_constrained_tradeoff.pdf`, and `code_cards_predictability_constrained.md`: predictability-constrained RouteCode diagnostics.",
            "- `table_new_model_integration.csv` and `fig_transfer_calibration_curve.pdf`: simulated held-out/new-model calibration diagnostics.",
            "- `table_ablation_summary.csv`, `fig_sensitivity_k_lambda.pdf`, and `fig_seed_stability.pdf`: bounded ablation and robustness diagnostics.",
            "- `table_sensitivity_summary.csv`, `fig_sensitivity_summary.pdf`, and `phase_g_sensitivity_memo.md`: bounded Phase G sensitivity diagnostics.",
            "- `table_external_baselines.csv` and `phase_e_external_baseline_memo.md`: local external-style baseline surrogate diagnostics.",
            "- `table_code_card_interpretability.csv` and `phase_f_code_card_interpretability_memo.md`: label-only versus code-card observability diagnostics.",
        ]
    return [
        "- `table_residual_concentration.csv`, `table_residual_risk.csv`, `fig_residual_concentration.pdf`, `fig_risk_coverage.pdf`, and `phase_d5_adaptive_refinement_gate_memo.md`: residual-regret concentration and adaptive-refinement gate diagnostics.",
        "- `table_split_sensitivity.csv`, `table_split_rank_correlation.csv`, `table_split_rate_threshold.csv`, and `fig_split_sensitivity.pdf`: split-sensitivity diagnostics.",
        "- `table_predictor_comparison.csv`, `table_utility_weighted_confusion.csv`, `table_calibration_curve.csv`, `fig_utility_weighted_confusion.pdf`, and `fig_calibration_curve.pdf`: RouteCode label-predictor diagnostics.",
        "- `table_predictability_constrained.csv`, `fig_predictability_constrained_tradeoff.pdf`, `code_cards_predictability_constrained.md`, and `phase_d_method_memo.md`: predictability-constrained RouteCode diagnostics.",
        "- `table_new_model_integration.csv`, `fig_transfer_calibration_curve.pdf`, and `phase_e5_new_model_calibration_memo.md`: simulated held-out/new-model calibration diagnostics.",
        "- `table_ablation_summary.csv`, `fig_sensitivity_k_lambda.pdf`, `fig_seed_stability.pdf`, and `phase_f_g_ablation_memo.md`: bounded ablation and robustness diagnostics.",
        "- `table_sensitivity_summary.csv`, `fig_sensitivity_summary.pdf`, and `phase_g_sensitivity_memo.md`: bounded Phase G sensitivity diagnostics.",
        "- `table_external_baselines.csv` and `phase_e_external_baseline_memo.md`: local external-style baseline surrogate diagnostics.",
        "- `table_code_card_interpretability.csv` and `phase_f_code_card_interpretability_memo.md`: label-only versus code-card observability diagnostics.",
        "- `table_official_external_artifacts.csv` and `phase_e_official_baseline_artifacts_memo.md`: official upstream baseline artifact inspection, not split-aligned RouteCode metrics.",
        "- `table_transformer_backbone_readiness.csv` and `phase_f_g_transformer_backbone_readiness_memo.md`: cache-only transformer backbone readiness audit.",
        "- `table_transformer_embedding_router.csv` and `phase_f_g_transformer_embedding_router_memo.md`: local-files-only pretrained encoder direct-router rows or skipped/failed blocker rows.",
        "- `table_embedllm_knn_split_aligned.csv` and `phase_e_embedllm_knn_split_aligned_memo.md`: split-aligned EmbedLLM KNN local metric-adapter evaluation.",
        "- `table_frugalgpt_split_aligned.csv` and `phase_e_frugalgpt_split_aligned_memo.md`: split-aligned FrugalGPT local-scorer metric-adapter evaluation.",
        "- `table_routellm_pairwise_alignment.csv` and `phase_e_routellm_pairwise_alignment_memo.md`: split-aligned RouteLLM pairwise substrate readiness audit.",
        "- `table_routellm_mf_assets.csv` and `phase_e_routellm_mf_assets_memo.md`: split-aligned RouteLLM MF trainer asset readiness audit.",
        "- `table_routellm_mf_split_aligned.csv` and `phase_e_routellm_mf_split_aligned_memo.md`: split-aligned RouteLLM MF training-code evaluation.",
        "- `table_avengerspro_split_aligned.csv` and `phase_e_avengerspro_split_aligned_memo.md`: split-aligned local Avengers-Pro cluster-routing compatibility evaluation.",
        "- `table_avengerspro_cli_metrics.csv` and `phase_e_avengerspro_cli_metrics_memo.md`: exact upstream Avengers-Pro simple-cluster accuracy/cost metrics on split-aligned assets; not RouteCode utility rows.",
        "- `table_avengerspro_upstream_metric.csv` and `phase_e_avengerspro_upstream_metric_memo.md`: RouteCode utility metrics over Avengers-Pro simple-cluster selections captured from upstream model code; not an exact upstream command row.",
        "- `table_graphrouter_cli_metrics.csv` and `phase_e_graphrouter_cli_metrics_memo.md`: exact upstream GraphRouter one-epoch smoke accuracy/cost metrics on split-aligned assets; not RouteCode utility rows.",
        "- `table_graphrouter_split_aligned.csv` and `phase_e_graphrouter_split_aligned_memo.md`: RouteCode utility metrics over split-aligned GraphRouter GNN selections using upstream model code; not an exact upstream command row.",
        "- `table_model_pool_scale.csv` and `phase_f_g_model_pool_scale_memo.md`: larger model-pool scale/composition robustness diagnostics.",
        "- `table_model_pool_transfer.csv` and `phase_f_g_model_pool_transfer_memo.md`: held-out model-pool transfer diagnostics.",
        "- `table_benchmark_file_coverage.csv`, `table_benchmark_dataset_coverage.csv`, `table_broad_coverage_candidates.csv`, and `phase_g_benchmark_coverage_memo.md`: raw LLMRouterBench coverage and broad complete-rectangle diagnostics.",
        "- `table_external_command_readiness.csv` and `phase_e_external_command_readiness_memo.md`: reproducible exact upstream-command readiness audit for remaining external baselines.",
        "- `table_cost_quality_summary.csv`, `table_cost_quality_frontier.csv`, `fig_cost_quality_frontier.pdf`, and `phase_e_cost_quality_memo.md`: fixed-quality and fixed-cost operating-point diagnostics.",
        "- `table_stronger_direct_router_probe.csv` and `phase_e_stronger_direct_router_probe_memo.md`: bounded stronger direct-router probe for MLP and gradient-boosting retraining baselines.",
        "- `table_external_baseline_assets.csv` and `phase_e_external_baseline_assets_memo.md`: split-aligned input assets for additional upstream external baseline command paths.",
        "- `table_llmrouter_library_adapters.csv` and `phase_e_llmrouter_library_adapters_memo.md`: split-aligned local LLMRouter trainer-class adapter metrics.",
        "- `table_provider_price_schedule.csv`, `table_provider_cost_quality_summary.csv`, `table_provider_cost_quality_frontier.csv`, `fig_provider_price_sensitivity.pdf`, and `phase_g_provider_pricing_memo.md`: partial provider-price sensitivity diagnostics for mapped provider models.",
        "- `phase_c_observation_memo.md`: Phase C checkpoint memo answering the seven pilot questions.",
    ]


if __name__ == "__main__":
    main()
