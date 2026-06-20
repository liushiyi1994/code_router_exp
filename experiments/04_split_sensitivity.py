from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from routecode.codes.routecode import RouteCodeCodebook
from routecode.config import load_config, output_dir
from routecode.data.splits import split_by_query
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.split_sensitivity import (
    cluster_heldout_split,
    compression_rate_to_reach,
    domain_homogeneous_split,
    leave_one_group_split,
    ranking_correlation,
)
from routecode.matrix import build_matrices
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_split_sensitivity
from routecode.predictors.classifiers import (
    LogisticModelRouter,
    PredictedLabelLookupRouter,
    RouteCodeLabelClassifier,
)
from routecode.reporting import upsert_markdown_section
from routecode.routers.cluster_lookup import EmbeddingClusterRouter
from routecode.routers.dataset_lookup import DatasetLabelRouter
from routecode.routers.knn import KNNRouter
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rate-only", action="store_true", help="Only compute missing split-rate threshold rows.")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Maximum new scenarios to compute in this invocation.")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    outcomes = prepared.outcomes.drop(columns=["split"], errors="ignore")
    embeddings = prepared.embeddings
    seed = int(config.get("run", {}).get("random_seed", 0))
    lambda_cost = float(config.get("utility", {}).get("lambda_cost", 0.0))
    split_config = config.get("split", {})
    sensitivity_config = config.get("split_sensitivity", {})

    scenarios = build_scenarios(outcomes, embeddings, config)
    if args.rate_only:
        run_rate_only(
            out_dir=out_dir,
            scenarios=scenarios,
            embeddings=embeddings,
            config=config,
            lambda_cost=lambda_cost,
            max_scenarios=args.max_scenarios,
        )
        return

    resume_partial = bool(sensitivity_config.get("resume_partial", False))
    checkpoint_partial = bool(sensitivity_config.get("checkpoint_partial", resume_partial))
    if resume_partial:
        all_rows, rate_rows, completed_scenarios = load_partial_tables(out_dir)
    else:
        all_rows, rate_rows, completed_scenarios = [], [], set()
    for scenario_name, scenario_type, split_outcomes, model_subset in scenarios:
        if scenario_name in completed_scenarios:
            continue
        try:
            metrics, rate = evaluate_scenario(
                scenario_name,
                scenario_type,
                split_outcomes,
                embeddings,
                config,
                lambda_cost,
                model_subset=model_subset,
            )
        except ValueError as exc:
            all_rows.append({"scenario": scenario_name, "scenario_type": scenario_type, "method": "SKIPPED", "error": str(exc)})
            completed_scenarios.add(scenario_name)
            if checkpoint_partial:
                write_partial_tables(out_dir, all_rows, rate_rows)
            continue
        all_rows.extend(metrics)
        rate_rows.append(rate)
        completed_scenarios.add(scenario_name)
        if checkpoint_partial:
            write_partial_tables(out_dir, all_rows, rate_rows)

    sensitivity = pd.DataFrame(all_rows)
    sensitivity.to_csv(out_dir / "table_split_sensitivity.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(out_dir / "table_split_rate_threshold.csv", index=False)
    rank_corr = split_rank_correlations(sensitivity)
    rank_corr.to_csv(out_dir / "table_split_rank_correlation.csv", index=False)
    save_split_sensitivity(sensitivity[sensitivity["method"] != "SKIPPED"], out_dir / "fig_split_sensitivity.pdf")
    append_readme(out_dir, args.config, sensitivity, rank_corr)
    clear_partial_tables(out_dir)
    print(f"Wrote split sensitivity outputs to {out_dir}")


def run_rate_only(
    out_dir: Path,
    scenarios: list[tuple[str, str, pd.DataFrame, list[str] | None]],
    embeddings: pd.DataFrame,
    config: dict,
    lambda_cost: float,
    max_scenarios: int | None = None,
) -> None:
    sensitivity_path = out_dir / "table_split_sensitivity.csv"
    if not sensitivity_path.exists():
        raise FileNotFoundError(f"Rate-only mode requires existing split sensitivity metrics: {sensitivity_path}")
    sensitivity = pd.read_csv(sensitivity_path)
    _, rate_path = partial_table_paths(out_dir)
    if rate_path.exists() and rate_path.stat().st_size > 0:
        rate_rows = pd.read_csv(rate_path).to_dict("records")
    else:
        rate_rows = []
    completed = {str(row["scenario"]) for row in rate_rows if pd.notna(row.get("scenario"))}
    computed = 0
    for scenario_name, scenario_type, split_outcomes, model_subset in scenarios:
        if scenario_name in completed:
            continue
        references = reference_values_for_scenario(sensitivity, scenario_name)
        rate = evaluate_scenario_rate(
            scenario_name,
            scenario_type,
            split_outcomes,
            embeddings,
            config,
            lambda_cost,
            references,
            model_subset=model_subset,
        )
        rate_rows.append(rate)
        completed.add(scenario_name)
        pd.DataFrame(rate_rows).to_csv(rate_path, index=False)
        computed += 1
        if max_scenarios is not None and computed >= max_scenarios:
            break

    if completed.issuperset({scenario_name for scenario_name, _, _, _ in scenarios}):
        final_rate = pd.DataFrame(rate_rows)
        final_rate.to_csv(out_dir / "table_split_rate_threshold.csv", index=False)
        rate_path.unlink(missing_ok=True)
        print(f"Wrote complete split-rate threshold outputs to {out_dir}")
    else:
        print(f"Wrote partial split-rate threshold outputs to {rate_path}")


def build_scenarios(outcomes: pd.DataFrame, embeddings: pd.DataFrame, config: dict) -> list[tuple[str, str, pd.DataFrame, list[str] | None]]:
    seed = int(config.get("run", {}).get("random_seed", 0))
    split_config = config.get("split", {})
    sensitivity = config.get("split_sensitivity", {})
    train_frac = float(split_config.get("train_frac", 0.6))
    val_frac = float(split_config.get("val_frac", 0.2))
    test_frac = float(split_config.get("test_frac", 0.2))
    scenarios: list[tuple[str, str, pd.DataFrame, list[str] | None]] = [
        (
            "random",
            "random",
            split_by_query(outcomes, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed),
            None,
        )
    ]

    datasets = sorted(outcomes["dataset"].astype(str).unique())
    domains = sorted(outcomes["domain"].astype(str).unique())
    max_groups = sensitivity.get("max_group_scenarios")
    if max_groups is not None:
        datasets = datasets[: int(max_groups)]
        domains = domains[: int(max_groups)]
    for dataset in datasets:
        scenarios.append(
            (
                f"leave_dataset_out:{dataset}",
                "leave_one_dataset_out",
                leave_one_group_split(outcomes, "dataset", dataset, seed=seed),
                None,
            )
        )
    for domain in domains:
        scenarios.append(
            (
                f"leave_domain_out:{domain}",
                "leave_one_domain_out",
                leave_one_group_split(outcomes, "domain", domain, seed=seed),
                None,
            )
        )
        scenarios.append(
            (
                f"domain_homogeneous:{domain}",
                "domain_homogeneous",
                domain_homogeneous_split(outcomes, domain, seed=seed),
                None,
            )
        )

    cluster_count = int(sensitivity.get("cluster_count", 4))
    holdout_count = int(sensitivity.get("cluster_holdout_count", min(3, cluster_count)))
    for cluster_id in range(holdout_count):
        split, heldout = cluster_heldout_split(outcomes, embeddings, cluster_count, cluster_id, seed=seed)
        scenarios.append((f"cluster_held_out:{heldout}", "cluster_held_out", split, None))

    models = sorted(outcomes["model_id"].astype(str).unique())
    max_model_scenarios = sensitivity.get("max_model_pool_scenarios")
    if max_model_scenarios is not None:
        models = models[: int(max_model_scenarios)]
    for holdout_model in models:
        model_subset = [model for model in sorted(outcomes["model_id"].astype(str).unique()) if model != holdout_model]
        split = split_by_query(outcomes, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed)
        scenarios.append((f"model_pool_holdout:{holdout_model}", "model_pool_holdout", split, model_subset))
    return scenarios


def evaluate_scenario(
    scenario: str,
    scenario_type: str,
    split_outcomes: pd.DataFrame,
    embeddings: pd.DataFrame,
    config: dict,
    lambda_cost: float,
    model_subset: list[str] | None = None,
) -> tuple[list[dict], dict]:
    if model_subset is not None:
        split_outcomes = split_outcomes[split_outcomes["model_id"].isin(model_subset)].copy()
    train_outcomes = split_outcomes[split_outcomes["split"] == "train"]
    test_outcomes = split_outcomes[split_outcomes["split"] == "test"]
    if train_outcomes["query_id"].nunique() < 2 or test_outcomes["query_id"].nunique() < 1:
        raise ValueError("Scenario lacks enough train/test queries")
    train = build_matrices(train_outcomes, lambda_cost=lambda_cost)
    test = build_matrices(test_outcomes, lambda_cost=lambda_cost)
    seed = int(config.get("run", {}).get("random_seed", 0))
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))
    router_config = config.get("routers", {})
    route_config = config.get("routecode", {})
    sensitivity_config = config.get("split_sensitivity", {})
    cluster_k = min(int(router_config.get("embedding_clusters", 16)), max(1, len(train.utility)))
    knn_k = min(int(router_config.get("knn_k", 15)), max(1, len(train.utility)))
    route_k = int(route_config.get("selected_k_for_cards", 16))
    classifier_max_iter = int(sensitivity_config.get("classifier_max_iter", 1000))
    rate_fit = split_rate_fit_controls(config)

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = selected_values(test.utility, best_single).mean()
    oracle_selected = OracleRouter().predict(test.utility)
    oracle_mean = test.utility.max(axis=1).mean()

    selected: list[tuple[str, pd.Series, pd.Series | None, int | None]] = [("best_single", best_single, None, None)]
    selected.append(("dataset_label_lookup", DatasetLabelRouter("dataset").fit(train.query_info, train.utility).predict(test.query_info), test.query_info["dataset"], int(test.query_info["dataset"].nunique())))
    if "predicted_topic" in train.query_info.columns and "predicted_topic" in test.query_info.columns:
        topic_selected = DatasetLabelRouter("predicted_topic").fit(train.query_info, train.utility).predict(test.query_info)
        topic_labels = test.query_info["predicted_topic"]
    else:
        topic_router = PredictedLabelLookupRouter(
            "dataset",
            random_state=seed,
            max_iter=classifier_max_iter,
        ).fit(train.query_info, train.utility, embeddings)
        topic_selected = topic_router.predict(test.query_info, embeddings)
        topic_labels = topic_router.predict_labels(embeddings.loc[test.utility.index])
    selected.append(("predicted_topic_lookup", topic_selected, topic_labels, int(topic_labels.nunique())))

    cluster_router = EmbeddingClusterRouter(cluster_k, random_state=seed).fit(train.query_info, train.utility, embeddings)
    cluster_labels = cluster_router.predict_labels(embeddings.loc[test.utility.index])
    selected.append(("embedding_cluster_lookup", cluster_router.predict(test.query_info, embeddings), cluster_labels, cluster_k))
    knn_selected = KNNRouter(knn_k).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings)
    selected.append(("kNN", knn_selected, None, None))
    logistic_selected = (
        LogisticModelRouter(random_state=seed, max_iter=classifier_max_iter)
        .fit(train.query_info, train.utility, embeddings)
        .predict(test.query_info, embeddings)
    )
    selected.append(("logistic_embedding_router", logistic_selected, None, None))

    routecode = RouteCodeCodebook(route_k, random_state=seed, max_iter=int(route_config.get("max_iter", 25))).fit(train.query_info, train.utility, embeddings)
    route_classifier = RouteCodeLabelClassifier(random_state=seed, max_iter=classifier_max_iter).fit(routecode, embeddings)
    route_labels = route_classifier.predict_labels(embeddings.loc[test.utility.index])
    selected.append(("routecode_predicted_labels", route_classifier.predict(test.query_info, embeddings), route_labels, route_k))
    selected.append(("query_oracle", oracle_selected, oracle_selected, int(test.utility.shape[1])))

    learned_reference_mean = max(
        selected_values(test.utility, models).mean()
        for name, models, _, _ in selected
        if name in {"kNN", "logistic_embedding_router", "routecode_predicted_labels"}
    )
    rows = []
    for method, models, labels, k in selected:
        row = evaluate_selection(
            method=method,
            selected_models=models,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            k=k,
            labels=labels,
        )
        row.update(
            {
                "scenario": scenario,
                "scenario_type": scenario_type,
                "n_train_queries": int(train.utility.shape[0]),
                "n_test_queries": int(test.utility.shape[0]),
                "n_models": int(test.utility.shape[1]),
            }
        )
        rows.append(row)

    rate_rows = []
    for k in split_rate_k_values(config):
        codebook = RouteCodeCodebook(
            k,
            random_state=seed,
            max_iter=rate_fit["codebook_max_iter"],
            n_init=rate_fit["codebook_n_init"],
        ).fit(train.query_info, train.utility, embeddings)
        classifier = RouteCodeLabelClassifier(random_state=seed, max_iter=rate_fit["classifier_max_iter"]).fit(codebook, embeddings)
        labels = classifier.predict_labels(embeddings.loc[test.utility.index])
        models = classifier.predict(test.query_info, embeddings)
        rate_rows.append(
            evaluate_selection(
                method="routecode_predicted_labels",
                selected_models=models,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=1,
                ci=ci,
                seed=seed,
                k=k,
                labels=labels,
            )
        )
    rate_table = pd.DataFrame(rate_rows)
    rate = {
        "scenario": scenario,
        "scenario_type": scenario_type,
        "rate_log2K_to_80pct_learned_gain": compression_rate_to_reach(rate_table, threshold=0.80),
        "best_routecode_predicted_recovered_gap_vs_learned": float(rate_table["recovered_gap_vs_learned"].max()),
        "best_routecode_predicted_recovered_gap_vs_oracle": float(rate_table["recovered_gap_vs_oracle"].max()),
    }
    return rows, rate


def evaluate_scenario_rate(
    scenario: str,
    scenario_type: str,
    split_outcomes: pd.DataFrame,
    embeddings: pd.DataFrame,
    config: dict,
    lambda_cost: float,
    references: dict[str, float],
    model_subset: list[str] | None = None,
) -> dict:
    if model_subset is not None:
        split_outcomes = split_outcomes[split_outcomes["model_id"].isin(model_subset)].copy()
    train_outcomes = split_outcomes[split_outcomes["split"] == "train"]
    test_outcomes = split_outcomes[split_outcomes["split"] == "test"]
    if train_outcomes["query_id"].nunique() < 2 or test_outcomes["query_id"].nunique() < 1:
        raise ValueError("Scenario lacks enough train/test queries")
    train = build_matrices(train_outcomes, lambda_cost=lambda_cost)
    test = build_matrices(test_outcomes, lambda_cost=lambda_cost)
    seed = int(config.get("run", {}).get("random_seed", 0))
    ci = float(config.get("bootstrap", {}).get("ci", 0.95))
    rate_fit = split_rate_fit_controls(config)
    rate_rows = []
    for k in split_rate_k_values(config):
        codebook = RouteCodeCodebook(
            k,
            random_state=seed,
            max_iter=rate_fit["codebook_max_iter"],
            n_init=rate_fit["codebook_n_init"],
        ).fit(train.query_info, train.utility, embeddings)
        classifier = RouteCodeLabelClassifier(random_state=seed, max_iter=rate_fit["classifier_max_iter"]).fit(codebook, embeddings)
        labels = classifier.predict_labels(embeddings.loc[test.utility.index])
        models = classifier.predict(test.query_info, embeddings)
        rate_rows.append(
            evaluate_selection(
                method="routecode_predicted_labels",
                selected_models=models,
                matrices=test,
                baseline_mean=references["baseline_mean"],
                learned_reference_mean=references["learned_reference_mean"],
                oracle_mean=references["oracle_mean"],
                n_bootstrap=1,
                ci=ci,
                seed=seed,
                k=k,
                labels=labels,
            )
        )
    rate_table = pd.DataFrame(rate_rows)
    return {
        "scenario": scenario,
        "scenario_type": scenario_type,
        "rate_log2K_to_80pct_learned_gain": compression_rate_to_reach(rate_table, threshold=0.80),
        "best_routecode_predicted_recovered_gap_vs_learned": float(rate_table["recovered_gap_vs_learned"].max()),
        "best_routecode_predicted_recovered_gap_vs_oracle": float(rate_table["recovered_gap_vs_oracle"].max()),
    }


def reference_values_for_scenario(sensitivity: pd.DataFrame, scenario: str) -> dict[str, float]:
    group = sensitivity[sensitivity["scenario"].astype(str) == str(scenario)]
    if group.empty:
        raise ValueError(f"Missing completed split sensitivity metrics for scenario: {scenario}")

    def method_mean(method: str) -> float:
        values = group.loc[group["method"] == method, "mean_utility"]
        if values.empty:
            raise ValueError(f"Missing method `{method}` for scenario: {scenario}")
        return float(values.iloc[0])

    learned_methods = {"kNN", "logistic_embedding_router", "routecode_predicted_labels"}
    learned = group[group["method"].isin(learned_methods)]
    if learned.empty:
        raise ValueError(f"Missing learned-router reference methods for scenario: {scenario}")
    return {
        "baseline_mean": method_mean("best_single"),
        "learned_reference_mean": float(learned["mean_utility"].max()),
        "oracle_mean": method_mean("query_oracle"),
    }


def split_rate_k_values(config: dict) -> list[int]:
    sensitivity_config = config.get("split_sensitivity", {})
    route_config = config.get("routecode", {})
    values = sensitivity_config.get("k_values", route_config.get("k_values", [1, 2, 4, 8, 16, 32]))
    return [int(value) for value in values]


def split_rate_fit_controls(config: dict) -> dict[str, int]:
    sensitivity_config = config.get("split_sensitivity", {})
    route_config = config.get("routecode", {})
    classifier_max_iter = int(
        sensitivity_config.get("rate_classifier_max_iter", sensitivity_config.get("classifier_max_iter", 1000))
    )
    codebook_max_iter = int(sensitivity_config.get("rate_codebook_max_iter", route_config.get("max_iter", 25)))
    codebook_n_init = int(sensitivity_config.get("rate_codebook_n_init", 10))
    return {
        "classifier_max_iter": classifier_max_iter,
        "codebook_max_iter": codebook_max_iter,
        "codebook_n_init": codebook_n_init,
    }


def partial_table_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / "table_split_sensitivity.partial.csv", out_dir / "table_split_rate_threshold.partial.csv"


def write_partial_tables(out_dir: Path, sensitivity_rows: list[dict], rate_rows: list[dict]) -> None:
    sensitivity_path, rate_path = partial_table_paths(out_dir)
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    pd.DataFrame(rate_rows).to_csv(rate_path, index=False)


def load_partial_tables(out_dir: Path) -> tuple[list[dict], list[dict], set[str]]:
    sensitivity_path, rate_path = partial_table_paths(out_dir)
    sensitivity_rows: list[dict] = []
    rate_rows: list[dict] = []
    completed: set[str] = set()
    if sensitivity_path.exists():
        sensitivity_frame = pd.read_csv(sensitivity_path)
        sensitivity_rows = sensitivity_frame.to_dict("records")
        if not sensitivity_frame.empty and {"scenario", "method"}.issubset(sensitivity_frame.columns):
            skipped = sensitivity_frame.loc[sensitivity_frame["method"] == "SKIPPED", "scenario"].dropna().astype(str)
            completed.update(skipped.tolist())
    if rate_path.exists():
        rate_frame = pd.read_csv(rate_path)
        rate_rows = rate_frame.to_dict("records")
        if not rate_frame.empty and "scenario" in rate_frame.columns:
            completed.update(rate_frame["scenario"].dropna().astype(str).tolist())
    return sensitivity_rows, rate_rows, completed


def clear_partial_tables(out_dir: Path) -> None:
    for path in partial_table_paths(out_dir):
        path.unlink(missing_ok=True)


def split_rank_correlations(sensitivity: pd.DataFrame) -> pd.DataFrame:
    valid = sensitivity[sensitivity["method"] != "SKIPPED"].copy()
    reference = valid[valid["scenario"] == "random"]
    rows = []
    for scenario, group in valid.groupby("scenario"):
        rows.append(
            {
                "scenario": scenario,
                "scenario_type": str(group["scenario_type"].iloc[0]),
                "rank_correlation_vs_random": ranking_correlation(reference, group),
                "mean_absolute_utility_delta_vs_random": _mean_abs_delta(reference, group, "mean_utility"),
                "mean_absolute_recovered_gap_delta_vs_random": _mean_abs_delta(reference, group, "recovered_gap_vs_oracle"),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario_type", "scenario"])


def _mean_abs_delta(reference: pd.DataFrame, comparison: pd.DataFrame, column: str) -> float:
    merged = reference[["method", column]].merge(comparison[["method", column]], on="method", suffixes=("_reference", "_comparison"))
    if merged.empty:
        return float("nan")
    return float((merged[f"{column}_reference"] - merged[f"{column}_comparison"]).abs().mean())


def append_readme(out_dir: Path, config_path: str, sensitivity: pd.DataFrame, rank_corr: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Split Sensitivity"
    valid = sensitivity[sensitivity["method"] != "SKIPPED"]
    random_rows = valid[valid["scenario"] == "random"]
    compact = random_rows[["method", "mean_utility", "recovered_gap_vs_oracle"]].sort_values("mean_utility", ascending=False)
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/04_split_sensitivity.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_split_sensitivity.csv`: method metrics for each split scenario.",
        "- `table_split_rank_correlation.csv`: ranking correlation and degradation against the random split.",
        "- `table_split_rate_threshold.csv`: RouteCode predicted-label rate needed to recover 80% learned-router gain when reached.",
        "- `fig_split_sensitivity.pdf`: heatmap of recovered gap vs oracle across scenarios.",
        "",
        "Random split ranking snapshot:",
        "",
        _markdown_table(compact),
        "",
        "Lowest rank correlations vs random:",
        "",
        _markdown_table(rank_corr.sort_values("rank_correlation_vs_random").head(8)),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
