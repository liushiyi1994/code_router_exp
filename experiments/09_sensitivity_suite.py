from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.config import load_config, output_dir
from routecode.data.text_features import build_hashing_embeddings
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.sensitivity import inject_label_noise, misestimate_cost_utility, query_length_buckets
from routecode.matrix import Matrices
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_sensitivity_summary
from routecode.predictors.classifiers import LogisticModelRouter
from routecode.reporting import upsert_markdown_section
from routecode.routers.cluster_lookup import AgglomerativeClusterRouter, EmbeddingClusterRouter
from routecode.routers.knn import KNNRouter
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter


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
    route_config = config.get("routecode", {})
    d2_config = config.get("predictability_constrained", {})
    sensitivity_config = config.get("sensitivity", {})
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))
    k = int(sensitivity_config.get("k", route_config.get("selected_k_for_cards", 16)))
    alpha = float(sensitivity_config.get("d2_alpha", d2_config.get("selected_alpha", 3.0)))
    beta = float(sensitivity_config.get("d2_beta", d2_config.get("beta", 0.0)))

    rows: list[dict[str, Any]] = []
    rows.extend(_embedding_backbone_rows(config, train, test, prepared.outcomes, embeddings, seed, k, alpha, beta, n_bootstrap, ci))
    rows.extend(_clustering_algorithm_rows(train, test, embeddings, seed, k, alpha, beta, n_bootstrap, ci))
    rows.extend(_label_noise_rows(config, train, test, embeddings, seed, n_bootstrap, ci))
    rows.extend(_cost_misestimation_rows(config, train, test, embeddings, seed, k, alpha, beta, n_bootstrap, ci))
    rows.extend(_price_ratio_rows(config, train, test, embeddings, seed, k, alpha, beta, n_bootstrap, ci))
    rows.extend(_model_pool_rows(config, train, test, embeddings, seed, k, alpha, beta, n_bootstrap, ci))
    rows.extend(
        _domain_granularity_rows(
            train,
            test,
            embeddings,
            seed,
            k,
            alpha,
            beta,
            n_bootstrap,
            ci,
            columns=sensitivity_config.get("domain_granularity_columns", ["domain", "dataset"]),
            text_cluster_counts=sensitivity_config.get("domain_granularity_text_clusters", []),
            min_queries=int(sensitivity_config.get("domain_granularity_min_queries", 10)),
        )
    )
    rows.extend(_query_length_rows(train, test, embeddings, seed, k, alpha, beta, n_bootstrap, ci))
    rows.extend(
        _bootstrap_sampling_rows(
            train,
            test,
            embeddings,
            seed,
            k,
            alpha,
            beta,
            [int(count) for count in sensitivity_config.get("bootstrap_counts", [50, 100, n_bootstrap])],
            ci,
        )
    )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_sensitivity_summary.csv", index=False)
    save_sensitivity_summary(table, out_dir / "fig_sensitivity_summary.pdf")
    append_readme(out_dir, args.config, table)
    write_memo(out_dir, args.config, table)
    print(f"Wrote sensitivity outputs to {out_dir}")


def _embedding_backbone_rows(
    config: dict,
    train: Matrices,
    test: Matrices,
    outcomes: pd.DataFrame,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    rows = []
    all_query_info = outcomes.drop_duplicates("query_id").set_index("query_id")
    variants: list[tuple[str, pd.DataFrame]] = [("configured", embeddings)]
    if "query_text" in all_query_info.columns:
        for n_features in config.get("sensitivity", {}).get("hashing_features", [64, 256]):
            variants.append((f"hashing_{int(n_features)}", build_hashing_embeddings(all_query_info, int(n_features))))
    for variant, variant_embeddings in variants:
        rows.extend(
            _key_method_rows(
                "embedding_backbone",
                variant,
                train,
                test,
                variant_embeddings,
                seed,
                k,
                alpha,
                beta,
                n_bootstrap,
                ci,
            )
        )
    return rows


def _clustering_algorithm_rows(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    baseline_mean, learned_reference_mean, oracle_mean = _references(train, test, embeddings, seed)
    rows = []
    routers = [
        ("kmeans", EmbeddingClusterRouter(k, random_state=seed).fit(train.query_info, train.utility, embeddings)),
        ("agglomerative", AgglomerativeClusterRouter(k).fit(train.query_info, train.utility, embeddings)),
    ]
    for variant, router in routers:
        labels = router.predict_labels(embeddings.loc[test.utility.index])
        rows.append(
            _row(
                "clustering_algorithm",
                variant,
                "semantic_embedding_cluster",
                router.predict(test.query_info, embeddings),
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed,
                k=k,
                labels=labels,
            )
        )
    rows.extend(_key_method_rows("clustering_algorithm", "d2_reference", train, test, embeddings, seed, k, alpha, beta, n_bootstrap, ci))
    return rows


def _label_noise_rows(
    config: dict,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    baseline_mean, learned_reference_mean, oracle_mean = _references(train, test, embeddings, seed)
    rows = []
    base_labels = train.utility.idxmax(axis=1).astype(str)
    model_ids = [str(model) for model in train.utility.columns]
    for noise_rate in config.get("sensitivity", {}).get("label_noise_rates", [0.0, 0.1, 0.2]):
        noisy_labels = inject_label_noise(base_labels, model_ids, float(noise_rate), seed=seed)
        selected = _fit_direct_logistic(noisy_labels, embeddings.loc[train.utility.index], embeddings.loc[test.utility.index], seed)
        rows.append(
            _row(
                "label_noise",
                f"noise_{float(noise_rate):g}",
                "logistic_embedding_router",
                selected,
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed,
            )
        )
    return rows


def _cost_misestimation_rows(
    config: dict,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    sensitivity_config = config.get("sensitivity", {})
    true_lambda = float(sensitivity_config.get("cost_lambda", config.get("utility", {}).get("lambda_cost", 0.0)))
    true_train = Matrices(
        train.quality,
        train.cost,
        misestimate_cost_utility(train.quality, train.cost, true_lambda, 1.0),
        train.query_info,
        train.model_ids,
    )
    true_test = Matrices(
        test.quality,
        test.cost,
        misestimate_cost_utility(test.quality, test.cost, true_lambda, 1.0),
        test.query_info,
        test.model_ids,
    )
    rows = []
    baseline_mean, learned_reference_mean, oracle_mean = _references(true_train, true_test, embeddings, seed)
    for multiplier in sensitivity_config.get("cost_multipliers", [0.5, 1.0, 2.0]):
        utility = misestimate_cost_utility(train.quality, train.cost, true_lambda, float(multiplier))
        shifted_train = Matrices(train.quality, train.cost, utility, train.query_info, train.model_ids)
        rows.extend(
            _key_method_rows(
                "cost_misestimation",
                f"cost_multiplier_{float(multiplier):g}",
                shifted_train,
                true_test,
                embeddings,
                seed,
                k,
                alpha,
                beta,
                n_bootstrap,
                ci,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
            )
        )
    return rows


def _price_ratio_rows(
    config: dict,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    sensitivity_config = config.get("sensitivity", {})
    lambda_cost = float(sensitivity_config.get("cost_lambda", config.get("utility", {}).get("lambda_cost", 0.0)))
    rows = []
    for exponent in sensitivity_config.get("price_ratio_exponents", []):
        exponent = float(exponent)
        shifted_train_cost = _scale_price_ratios(train.cost, exponent)
        shifted_test_cost = _scale_price_ratios(test.cost, exponent, reference_means=train.cost.mean(axis=0))
        shifted_train = Matrices(
            train.quality,
            shifted_train_cost,
            train.quality - lambda_cost * shifted_train_cost,
            train.query_info,
            train.model_ids,
        )
        shifted_test = Matrices(
            test.quality,
            shifted_test_cost,
            test.quality - lambda_cost * shifted_test_cost,
            test.query_info,
            test.model_ids,
        )
        baseline_mean, learned_reference_mean, oracle_mean = _references(shifted_train, shifted_test, embeddings, seed)
        rows.extend(
            _key_method_rows(
                "price_ratio",
                f"price_ratio_exponent_{exponent:g}",
                shifted_train,
                shifted_test,
                embeddings,
                seed,
                k,
                alpha,
                beta,
                n_bootstrap,
                ci,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
            )
        )
    return rows


def _scale_price_ratios(
    cost: pd.DataFrame,
    exponent: float,
    reference_means: pd.Series | None = None,
) -> pd.DataFrame:
    reference = (reference_means if reference_means is not None else cost.mean(axis=0)).astype(float)
    positive = reference[reference > 0]
    if positive.empty:
        return cost.copy()
    anchor = float(np.exp(np.log(positive).mean()))
    ratios = (reference / anchor).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    scale = ratios.pow(float(exponent) - 1.0)
    scale = scale.where(reference > 0, 1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return cost.mul(scale, axis=1)


def _model_pool_rows(
    config: dict,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    rows = []
    sensitivity_config = config.get("sensitivity", {})
    scenarios = _model_pool_scenarios(
        train.utility,
        sensitivity_config.get("model_pool_sizes", [4]),
        configured_pools=sensitivity_config.get("model_pools", []),
        auto_sizes=sensitivity_config.get("auto_model_pool_sizes", []),
    )
    for sensitivity_name, scenario, models in scenarios:
        subset_train = _subset_matrices(train, models)
        subset_test = _subset_matrices(test, models)
        rows.extend(
            _key_method_rows(
                sensitivity_name,
                scenario,
                subset_train,
                subset_test,
                embeddings,
                seed,
                min(k, len(models)),
                alpha,
                beta,
                n_bootstrap,
                ci,
            )
        )
    return rows


def _domain_granularity_rows(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
    columns: list[str],
    text_cluster_counts: list[int],
    min_queries: int,
) -> list[dict[str, Any]]:
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    knn = KNNRouter(15).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings)
    d2 = PredictabilityConstrainedRouteCode(k, alpha=alpha, beta=beta, random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    d2_labels = d2.predict_labels(embeddings.loc[test.utility.index])
    d2_selected = d2.predict_from_labels(d2_labels)
    rows: list[dict[str, Any]] = []

    assignments: list[tuple[str, pd.Series]] = []
    for column in columns:
        if column in test.query_info.columns:
            assignments.append((str(column), test.query_info[column].astype(str)))
    assignments.extend(_text_cluster_assignments(train, test, embeddings, text_cluster_counts, seed))

    for granularity, labels in assignments:
        for label in sorted(labels.dropna().astype(str).unique()):
            query_ids = labels.index[labels.astype(str) == label]
            if len(query_ids) < int(min_queries):
                continue
            bucket_test = Matrices(
                test.quality.loc[query_ids],
                test.cost.loc[query_ids],
                test.utility.loc[query_ids],
                test.query_info.loc[query_ids],
                test.model_ids,
            )
            baseline_mean, learned_reference_mean, oracle_mean = _references(train, bucket_test, embeddings, seed)
            variant = f"{granularity}:{label}"
            for method, selected, route_labels in [
                ("best_single", best_single.loc[query_ids], None),
                ("kNN", knn.loc[query_ids], None),
                ("d2_embedding_centroid", d2_selected.loc[query_ids], d2_labels.loc[query_ids]),
            ]:
                rows.append(
                    _row(
                        "domain_granularity",
                        variant,
                        method,
                        selected,
                        bucket_test,
                        baseline_mean,
                        learned_reference_mean,
                        oracle_mean,
                        n_bootstrap,
                        ci,
                        seed,
                        k=k if route_labels is not None else None,
                        labels=route_labels,
                    )
                )
    return rows


def _text_cluster_assignments(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    cluster_counts: list[int],
    seed: int,
) -> list[tuple[str, pd.Series]]:
    assignments: list[tuple[str, pd.Series]] = []
    train_embeddings = embeddings.loc[train.utility.index]
    test_embeddings = embeddings.loc[test.utility.index]
    for raw_count in cluster_counts:
        count = min(max(2, int(raw_count)), len(train_embeddings))
        if count < 2:
            continue
        kmeans = KMeans(n_clusters=count, random_state=seed, n_init=10)
        kmeans.fit(train_embeddings.to_numpy(dtype=float))
        labels = pd.Series(
            [f"cluster_{int(label)}" for label in kmeans.predict(test_embeddings.to_numpy(dtype=float))],
            index=test_embeddings.index,
            name=f"text_cluster_{count}",
        )
        assignments.append((f"text_cluster_{count}", labels))
    return assignments


def _query_length_rows(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    d2 = PredictabilityConstrainedRouteCode(k, alpha=alpha, beta=beta, random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    labels = d2.predict_labels(embeddings.loc[test.utility.index])
    selected = d2.predict_from_labels(labels)
    buckets = query_length_buckets(test.query_info, n_bins=3)
    rows = []
    for bucket in sorted(buckets.unique()):
        query_ids = buckets.index[buckets == bucket]
        if len(query_ids) == 0:
            continue
        bucket_test = Matrices(
            test.quality.loc[query_ids],
            test.cost.loc[query_ids],
            test.utility.loc[query_ids],
            test.query_info.loc[query_ids],
            test.model_ids,
        )
        baseline_mean, learned_reference_mean, oracle_mean = _references(train, bucket_test, embeddings, seed)
        rows.append(
            _row(
                "query_length_bucket",
                str(bucket),
                "d2_embedding_centroid",
                selected.loc[query_ids],
                bucket_test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed,
                k=k,
                labels=labels.loc[query_ids],
            )
        )
    return rows


def _bootstrap_sampling_rows(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    bootstrap_counts: list[int],
    ci: float,
) -> list[dict[str, Any]]:
    baseline_mean, learned_reference_mean, oracle_mean = _references(train, test, embeddings, seed)
    d2 = PredictabilityConstrainedRouteCode(k, alpha=alpha, beta=beta, random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    labels = d2.predict_labels(embeddings.loc[test.utility.index])
    selected = d2.predict_from_labels(labels)
    rows = []
    seen = set()
    for count in bootstrap_counts:
        count = int(count)
        if count <= 0 or count in seen:
            continue
        seen.add(count)
        rows.append(
            _row(
                "bootstrap_sampling",
                f"n_bootstrap_{count}",
                "d2_embedding_centroid",
                selected,
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                count,
                ci,
                seed,
                k=k,
                labels=labels,
            )
        )
    return rows


def _key_method_rows(
    sensitivity: str,
    variant: str,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
    baseline_mean: float | None = None,
    learned_reference_mean: float | None = None,
    oracle_mean: float | None = None,
) -> list[dict[str, Any]]:
    if baseline_mean is None or learned_reference_mean is None or oracle_mean is None:
        baseline_mean, learned_reference_mean, oracle_mean = _references(train, test, embeddings, seed)
    rows = []
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    knn = KNNRouter(15).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings)
    d2 = PredictabilityConstrainedRouteCode(k, alpha=alpha, beta=beta, random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    d2_labels = d2.predict_labels(embeddings.loc[test.utility.index])
    for method, selected, labels in [
        ("best_single", best_single, None),
        ("kNN", knn, None),
        ("d2_embedding_centroid", d2.predict_from_labels(d2_labels), d2_labels),
    ]:
        rows.append(
            _row(
                sensitivity,
                variant,
                method,
                selected,
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed,
                k=k if labels is not None else None,
                labels=labels,
            )
        )
    return rows


def _fit_direct_logistic(
    labels: pd.Series,
    train_embeddings: pd.DataFrame,
    test_embeddings: pd.DataFrame,
    seed: int,
) -> pd.Series:
    from routecode.eval.new_model_calibration import fit_predict_budgeted_direct_router

    return fit_predict_budgeted_direct_router(
        method="logistic",
        train_labels=labels,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        random_state=seed,
    )


def _references(train: Matrices, test: Matrices, embeddings: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    oracle_mean = float(test.utility.max(axis=1).mean())
    learned = [
        KNNRouter(15).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings),
        LogisticModelRouter(random_state=seed).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings),
    ]
    learned_reference_mean = max([baseline_mean] + [float(selected_values(test.utility, selected).mean()) for selected in learned])
    return baseline_mean, learned_reference_mean, oracle_mean


def _model_pool_scenarios(
    utility: pd.DataFrame,
    sizes: list[int],
    configured_pools: list[dict[str, Any]] | None = None,
    auto_sizes: list[int] | None = None,
) -> list[tuple[str, str, list[str]]]:
    mean_utility = utility.mean(axis=0).sort_values(ascending=False)
    known_models = {str(model) for model in utility.columns}
    scenarios: list[tuple[str, str, list[str]]] = [("model_pool", "full", [str(model) for model in utility.columns])]
    for size in sizes:
        size = min(max(2, int(size)), len(mean_utility))
        scenarios.append(("model_pool", f"top_{size}", [str(model) for model in mean_utility.head(size).index]))
    if len(mean_utility) > 2:
        scenarios.append(("model_pool", "drop_dominant", [str(model) for model in mean_utility.index[1:]]))
    for pool in configured_pools or []:
        name = str(pool.get("name") or "").strip()
        models = [str(model) for model in pool.get("models", [])]
        if not name or len(models) < 2:
            continue
        if any(model not in known_models for model in models):
            continue
        scenarios.append(("model_pool_composition", name, models))
    scenarios.extend(_automatic_model_pool_scenarios(utility, auto_sizes or []))
    unique: list[tuple[str, str, list[str]]] = []
    seen = set()
    for sensitivity_name, name, models in scenarios:
        key = (sensitivity_name, name, tuple(models))
        if key not in seen:
            unique.append((sensitivity_name, name, models))
            seen.add(key)
    return unique


def _automatic_model_pool_scenarios(
    utility: pd.DataFrame,
    sizes: list[int],
) -> list[tuple[str, str, list[str]]]:
    models = [str(model) for model in utility.columns]
    scenarios: list[tuple[str, str, list[str]]] = []
    for raw_size in sizes:
        size = min(max(2, int(raw_size)), len(models))
        scored = [_pool_stats(utility, list(combo)) for combo in combinations(models, size)]
        if not scored:
            continue
        complementary = max(
            scored,
            key=lambda row: (row["oracle_gap"], row["winner_entropy"], -row["dominance_ratio"], row["mean_utility"]),
        )
        dominated = min(
            scored,
            key=lambda row: (row["oracle_gap"], -row["dominance_ratio"], -row["mean_utility"], row["winner_entropy"]),
        )
        scenarios.append(("model_pool_auto", f"complementary_size_{size}", complementary["models"]))
        scenarios.append(("model_pool_auto", f"dominated_size_{size}", dominated["models"]))
    return scenarios


def _pool_stats(utility: pd.DataFrame, models: list[str]) -> dict[str, Any]:
    subset = utility.loc[:, models]
    mean_utility = float(subset.mean(axis=0).max())
    oracle = subset.max(axis=1)
    winners = subset.idxmax(axis=1).astype(str)
    winner_share = winners.value_counts(normalize=True)
    return {
        "models": models,
        "mean_utility": mean_utility,
        "oracle_mean": float(oracle.mean()),
        "oracle_gap": float(oracle.mean() - mean_utility),
        "dominance_ratio": float(winner_share.max()),
        "winner_entropy": _winner_entropy(winner_share),
    }


def _winner_entropy(winner_share: pd.Series) -> float:
    if winner_share.empty:
        return 0.0
    return float(-(winner_share * np.log2(winner_share)).sum())


def _subset_matrices(matrices: Matrices, models: list[str]) -> Matrices:
    return Matrices(
        quality=matrices.quality.loc[:, models],
        cost=matrices.cost.loc[:, models],
        utility=matrices.utility.loc[:, models],
        query_info=matrices.query_info,
        model_ids=models,
    )


def _row(
    sensitivity: str,
    variant: str,
    method: str,
    selected_models: pd.Series,
    test: Matrices,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    k: int | None = None,
    labels: pd.Series | None = None,
) -> dict[str, Any]:
    row = evaluate_selection(
        method=method,
        selected_models=selected_models,
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
    row.update({"sensitivity": sensitivity, "variant": variant})
    return row


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Sensitivity Suite"
    summary = (
        table.groupby(["sensitivity", "method"], as_index=False)
        .agg(mean_recovered_gap=("recovered_gap_vs_oracle", "mean"))
        .sort_values(["sensitivity", "mean_recovered_gap"], ascending=[True, False])
    )
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/09_sensitivity_suite.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_sensitivity_summary.csv`: bounded embedding, clustering, label-noise, cost, price-ratio, model-pool subset/composition, automatic dominated/complementary pool, domain-granularity, query-length, and bootstrap sensitivity rows.",
        "- `fig_sensitivity_summary.pdf`: method-by-sensitivity recovered-gap heatmap.",
        "- `phase_g_sensitivity_memo.md`: Phase G checkpoint memo.",
        "",
        _markdown_table(summary),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    summary = (
        table.groupby(["sensitivity", "method"], as_index=False)
        .agg(
            mean_gap=("recovered_gap_vs_oracle", "mean"),
            min_gap=("recovered_gap_vs_oracle", "min"),
            max_gap=("recovered_gap_vs_oracle", "max"),
        )
        .sort_values(["sensitivity", "mean_gap"], ascending=[True, False])
    )
    lines = [
        "# Phase G Sensitivity Memo",
        "",
        f"Command: `python experiments/09_sensitivity_suite.py --config {config_path}`",
        "",
        "This is a bounded sensitivity layer. It is not a full robustness proof.",
        "",
        _markdown_table(summary),
        "",
        "## Current Readout",
        "",
        "- Covered here: embedding feature variant, clustering algorithm, label noise, cost mis-estimation, price-ratio objective stress, model-pool subset/composition, automatic dominated/complementary model-pool construction, domain-granularity bucket sensitivity, query-length bucket sensitivity, and bootstrap sampling sensitivity.",
        "- Configured model-pool composition now includes the pilot `qwen_pair`, `qwen_deepseek_llama`, and `compact_pair` slices when those models are available.",
        "- Automatic model-pool construction selects dominated and complementary pools from the available model columns for configured sizes.",
        "- Price-ratio rows flatten or expand model-average cost ratios before recomputing the cost-quality utility objective.",
        "- Domain-granularity rows evaluate global router selections within coarse-domain, curated task-family/task-subtype taxonomy, dataset, and train-fitted text-cluster buckets using bucket-local references.",
        "- Split sensitivity now uses the configured coarse LLMRouterBench domain map; still missing or shallow are external embedding backbones and larger benchmark-scale taxonomy coverage.",
        "",
    ]
    (out_dir / "phase_g_sensitivity_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
