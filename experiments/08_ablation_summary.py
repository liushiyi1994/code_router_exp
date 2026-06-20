from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.codes.regret import RegretOptimizedRouteCode
from routecode.codes.routecode import RouteCodeCodebook
from routecode.config import load_config, output_dir
from routecode.data.splits import split_by_query
from routecode.eval.ablation import configured_sweep_values, sample_train_query_ids
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.new_model_calibration import fit_predict_budgeted_direct_router
from routecode.matrix import Matrices, build_matrices
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_seed_stability, save_sensitivity_k_lambda
from routecode.reporting import upsert_markdown_section
from routecode.routers.cluster_lookup import EmbeddingClusterRouter
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
    seed = int(config.get("run", {}).get("random_seed", 0))
    ablation = config.get("ablation", {})
    route_config = config.get("routecode", {})
    d2_config = config.get("predictability_constrained", {})
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))

    base_lambda = float(config.get("utility", {}).get("lambda_cost", 0.0))
    lambda_values = configured_sweep_values(config, "ablation", "lambda_values", base_lambda, cast=float)
    k_values = configured_sweep_values(
        config,
        "ablation",
        "k_values",
        int(route_config.get("selected_k_for_cards", 16)),
        cast=int,
    )
    seed_values = configured_sweep_values(config, "ablation", "seeds", seed, cast=int)
    train_fractions = configured_sweep_values(config, "ablation", "train_fractions", 1.0, cast=float)
    beta_values = configured_sweep_values(config, "ablation", "beta_values", d2_config.get("beta", 0.0), cast=float)
    max_iter = int(ablation.get("max_iter", route_config.get("max_iter", 25)))
    d2_alpha = float(ablation.get("d2_alpha", d2_config.get("selected_alpha", 3.0)))
    d2_beta = float(ablation.get("d2_beta", d2_config.get("beta", 0.0)))
    fit_controls = ablation_fit_controls(config)

    rows: list[dict[str, Any]] = []
    rows.extend(
        _k_lambda_rows(
            prepared.outcomes,
            prepared.embeddings,
            lambda_values,
            k_values,
            seed,
            max_iter,
            d2_alpha,
            d2_beta,
            n_bootstrap,
            ci,
            fit_controls,
            fit_controls["kmeans_n_init"],
        )
    )
    rows.extend(
        _rate_penalty_rows(
            prepared.outcomes,
            prepared.embeddings,
            beta_values,
            int(route_config.get("selected_k_for_cards", 16)),
            base_lambda,
            seed,
            max_iter,
            d2_alpha,
            n_bootstrap,
            ci,
            fit_controls["kmeans_n_init"],
            fit_controls,
        )
    )
    rows.extend(
        _seed_rows(
            config,
            seed_values,
            int(route_config.get("selected_k_for_cards", 16)),
            max_iter,
            d2_alpha,
            d2_beta,
            n_bootstrap,
            ci,
            fit_controls,
            fit_controls["kmeans_n_init"],
        )
    )
    rows.extend(
        _train_fraction_rows(
            prepared.matrices["train"],
            prepared.matrices["test"],
            prepared.embeddings,
            train_fractions,
            seed,
            n_bootstrap,
            ci,
            int(route_config.get("selected_k_for_cards", 16)),
            max_iter,
            d2_alpha,
            d2_beta,
            fit_controls,
            fit_controls["kmeans_n_init"],
        )
    )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_ablation_summary.csv", index=False)
    save_sensitivity_k_lambda(table, out_dir / "fig_sensitivity_k_lambda.pdf")
    save_seed_stability(table, out_dir / "fig_seed_stability.pdf")
    append_readme(out_dir, args.config, table)
    write_memo(out_dir, args.config, table)
    print(f"Wrote ablation outputs to {out_dir}")


def _k_lambda_rows(
    outcomes: pd.DataFrame,
    embeddings: pd.DataFrame,
    lambda_values: list[float],
    k_values: list[int],
    seed: int,
    max_iter: int,
    d2_alpha: float,
    d2_beta: float,
    n_bootstrap: int,
    ci: float,
    fit_controls: dict[str, Any],
    kmeans_n_init: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lambda_cost in lambda_values:
        matrices = _matrices_by_split(outcomes, lambda_cost)
        train = matrices["train"]
        test = matrices["test"]
        baseline_mean, learned_reference_mean, oracle_mean = _references(
            train,
            test,
            embeddings,
            seed,
            fit_controls,
        )
        best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
        rows.append(
            _row(
                "k_lambda",
                "best_single",
                best_single,
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed,
                lambda_cost=lambda_cost,
            )
        )
        for k in k_values:
            semantic = EmbeddingClusterRouter(k, random_state=seed, n_init=kmeans_n_init).fit(
                train.query_info,
                train.utility,
                embeddings,
            )
            semantic_labels = semantic.predict_labels(embeddings.loc[test.utility.index])
            rows.append(
                _row(
                    "k_lambda",
                    "semantic_embedding_kmeans",
                    semantic.predict(test.query_info, embeddings),
                    test,
                    baseline_mean,
                    learned_reference_mean,
                    oracle_mean,
                    n_bootstrap,
                    ci,
                    seed + k,
                    k=k,
                    labels=semantic_labels,
                    lambda_cost=lambda_cost,
                )
            )
            flat = RouteCodeCodebook(k, random_state=seed, max_iter=max_iter, n_init=kmeans_n_init).fit(
                train.query_info,
                train.utility,
                embeddings,
            )
            flat_labels = flat.predict_utility_labels(test.utility)
            rows.append(
                _row(
                    "k_lambda",
                    "flat_routecode_utility_oracle",
                    flat.predict_from_labels(flat_labels),
                    test,
                    baseline_mean,
                    learned_reference_mean,
                    oracle_mean,
                    n_bootstrap,
                    ci,
                    seed + 100 + k,
                    k=k,
                    labels=flat_labels,
                    lambda_cost=lambda_cost,
                )
            )
            regret = RegretOptimizedRouteCode(k, random_state=seed, max_iter=max_iter, n_init=kmeans_n_init).fit(
                train.query_info,
                train.utility,
                embeddings,
            )
            regret_labels = regret.predict_utility_labels(test.utility)
            rows.append(
                _row(
                    "k_lambda",
                    "regret_routecode_utility_oracle",
                    regret.predict_from_labels(regret_labels),
                    test,
                    baseline_mean,
                    learned_reference_mean,
                    oracle_mean,
                    n_bootstrap,
                    ci,
                    seed + 150 + k,
                    k=k,
                    labels=regret_labels,
                    lambda_cost=lambda_cost,
                )
            )
            d2 = PredictabilityConstrainedRouteCode(
                k,
                alpha=d2_alpha,
                beta=d2_beta,
                random_state=seed,
                max_iter=max_iter,
                n_init=kmeans_n_init,
            ).fit(train.query_info, train.utility, embeddings)
            d2_labels = d2.predict_labels(embeddings.loc[test.utility.index])
            rows.append(
                _row(
                    "k_lambda",
                    "d2_embedding_centroid",
                    d2.predict_from_labels(d2_labels),
                    test,
                    baseline_mean,
                    learned_reference_mean,
                    oracle_mean,
                    n_bootstrap,
                    ci,
                    seed + 200 + k,
                    k=k,
                    labels=d2_labels,
                    lambda_cost=lambda_cost,
                )
            )
    return rows


def _rate_penalty_rows(
    outcomes: pd.DataFrame,
    embeddings: pd.DataFrame,
    beta_values: list[float],
    k: int,
    lambda_cost: float,
    seed: int,
    max_iter: int,
    d2_alpha: float,
    n_bootstrap: int,
    ci: float,
    kmeans_n_init: int = 10,
    fit_controls: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    matrices = _matrices_by_split(outcomes, lambda_cost)
    train = matrices["train"]
    test = matrices["test"]
    controls = fit_controls or ablation_fit_controls({})
    baseline_mean, learned_reference_mean, oracle_mean = _references(train, test, embeddings, seed, controls)
    rows: list[dict[str, Any]] = []
    for beta in beta_values:
        d2 = PredictabilityConstrainedRouteCode(
            k,
            alpha=d2_alpha,
            beta=float(beta),
            random_state=seed,
            max_iter=max_iter,
            n_init=kmeans_n_init,
        ).fit(train.query_info, train.utility, embeddings)
        labels = d2.predict_labels(embeddings.loc[test.utility.index])
        rows.append(
            _row(
                "rate_penalty",
                "d2_embedding_centroid",
                d2.predict_from_labels(labels),
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed + int(1000 * float(beta)),
                k=k,
                labels=labels,
                lambda_cost=lambda_cost,
                d2_beta=float(beta),
            )
        )
    return rows


def _seed_rows(
    config: dict,
    seed_values: list[int],
    k: int,
    max_iter: int,
    d2_alpha: float,
    d2_beta: float,
    n_bootstrap: int,
    ci: float,
    fit_controls: dict[str, Any],
    kmeans_n_init: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seed_values:
        local_config = _with_seed(config, seed)
        prepared = prepare_from_config(local_config)
        train = prepared.matrices["train"]
        test = prepared.matrices["test"]
        embeddings = prepared.embeddings
        baseline_mean, learned_reference_mean, oracle_mean = _references(
            train,
            test,
            embeddings,
            seed,
            fit_controls,
        )
        methods = _deployable_methods(
            train,
            test,
            embeddings,
            seed,
            k,
            max_iter,
            d2_alpha,
            d2_beta,
            fit_controls,
            kmeans_n_init,
        )
        for method, selected, labels in methods:
            rows.append(
                _row(
                    "seed_stability",
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
                    split_seed=seed,
                    lambda_cost=float(local_config.get("utility", {}).get("lambda_cost", 0.0)),
                )
            )
    return rows


def _train_fraction_rows(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    train_fractions: list[float],
    seed: int,
    n_bootstrap: int,
    ci: float,
    k: int,
    max_iter: int,
    d2_alpha: float,
    d2_beta: float,
    fit_controls: dict[str, Any] | None = None,
    kmeans_n_init: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controls = fit_controls or ablation_fit_controls({})
    baseline_mean, learned_reference_mean, oracle_mean = _references(train, test, embeddings, seed, controls)
    for fraction in train_fractions:
        query_ids = sample_train_query_ids(train.utility.index, fraction, seed=seed + int(1000 * fraction))
        subset_query_info = train.query_info.loc[query_ids]
        subset_utility = train.utility.loc[query_ids]
        best_single = BestSingleRouter().fit(subset_query_info, subset_utility).predict(test.query_info)
        knn = KNNRouter(15).fit(subset_query_info, subset_utility, embeddings).predict(test.query_info, embeddings)
        d2 = PredictabilityConstrainedRouteCode(
            k,
            alpha=d2_alpha,
            beta=d2_beta,
            random_state=seed,
            max_iter=max_iter,
            n_init=kmeans_n_init,
        ).fit(subset_query_info, subset_utility, embeddings)
        d2_labels = d2.predict_labels(embeddings.loc[test.utility.index])
        for method, selected, labels, method_k in [
            ("best_single", best_single, None, None),
            ("kNN", knn, None, None),
            (
                "logistic_embedding_router",
                _fit_direct_router(
                    "logistic",
                    subset_utility,
                    test.query_info,
                    embeddings,
                    seed,
                    controls,
                ),
                None,
                None,
            ),
            (
                "svm_embedding_router",
                _fit_direct_router(
                    "svm",
                    subset_utility,
                    test.query_info,
                    embeddings,
                    seed,
                    controls,
                ),
                None,
                None,
            ),
            ("d2_embedding_centroid", d2.predict_from_labels(d2_labels), d2_labels, d2.effective_labels),
        ]:
            rows.append(
                _row(
                    "train_fraction",
                    method,
                    selected,
                    test,
                    baseline_mean,
                    learned_reference_mean,
                    oracle_mean,
                    n_bootstrap,
                    ci,
                    seed,
                    k=method_k,
                    labels=labels,
                    train_fraction=fraction,
                    train_query_count=len(query_ids),
                )
            )
    return rows


def _deployable_methods(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    max_iter: int,
    d2_alpha: float,
    d2_beta: float,
    fit_controls: dict[str, Any],
    kmeans_n_init: int,
) -> list[tuple[str, pd.Series, pd.Series | None]]:
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    knn = KNNRouter(15).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings)
    logistic = _fit_direct_router(
        "logistic",
        train.utility,
        test.query_info,
        embeddings,
        seed,
        fit_controls,
    )
    svm = _fit_direct_router(
        "svm",
        train.utility,
        test.query_info,
        embeddings,
        seed,
        fit_controls,
    )
    d2 = PredictabilityConstrainedRouteCode(
        k,
        alpha=d2_alpha,
        beta=d2_beta,
        random_state=seed,
        max_iter=max_iter,
        n_init=kmeans_n_init,
    ).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    d2_labels = d2.predict_labels(embeddings.loc[test.utility.index])
    return [
        ("best_single", best_single, None),
        ("kNN", knn, None),
        ("logistic_embedding_router", logistic, None),
        ("svm_embedding_router", svm, None),
        ("d2_embedding_centroid", d2.predict_from_labels(d2_labels), d2_labels),
    ]


def _references(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    fit_controls: dict[str, Any],
) -> tuple[float, float, float]:
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    oracle_mean = float(test.utility.max(axis=1).mean())
    learned = [
        KNNRouter(15).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings),
        _fit_direct_router(
            "logistic",
            train.utility,
            test.query_info,
            embeddings,
            seed,
            fit_controls,
        ),
        _fit_direct_router(
            "svm",
            train.utility,
            test.query_info,
            embeddings,
            seed,
            fit_controls,
        ),
    ]
    learned_reference_mean = max([baseline_mean] + [float(selected_values(test.utility, selected).mean()) for selected in learned])
    return baseline_mean, learned_reference_mean, oracle_mean


def _fit_direct_router(
    method: str,
    train_utility: pd.DataFrame,
    test_query_info: pd.DataFrame,
    embeddings: pd.DataFrame,
    seed: int,
    fit_controls: dict[str, Any],
) -> pd.Series:
    train_labels = train_utility.idxmax(axis=1).astype(str).rename("selected_model")
    return fit_predict_budgeted_direct_router(
        method=method,
        train_labels=train_labels,
        train_embeddings=embeddings.loc[train_utility.index],
        test_embeddings=embeddings.loc[test_query_info.index],
        random_state=seed,
        max_iter=int(fit_controls["classifier_max_iter"]),
        n_neighbors=15,
        logistic_solver=str(fit_controls["logistic_solver"]),
        svm_backend=str(fit_controls["svm_backend"]),
        tol=float(fit_controls["classifier_tol"]),
    )


def ablation_fit_controls(config: dict) -> dict[str, Any]:
    ablation = config.get("ablation", {})
    return {
        "classifier_max_iter": int(ablation.get("classifier_max_iter", 1000)),
        "kmeans_n_init": int(ablation.get("kmeans_n_init", 10)),
        "logistic_solver": str(ablation.get("logistic_solver", "lbfgs")),
        "svm_backend": str(ablation.get("svm_backend", "linear_svc")),
        "classifier_tol": float(ablation.get("classifier_tol", 1e-4)),
    }


def _matrices_by_split(outcomes: pd.DataFrame, lambda_cost: float) -> dict[str, Matrices]:
    return {
        split: build_matrices(outcomes[outcomes["split"] == split], lambda_cost=lambda_cost)
        for split in ["train", "val", "test"]
    }


def _with_seed(config: dict, seed: int) -> dict:
    copied = dict(config)
    copied["run"] = dict(config.get("run", {}))
    copied["run"]["random_seed"] = int(seed)
    return copied


def _row(
    ablation: str,
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
    split_seed: int | None = None,
    lambda_cost: float | None = None,
    train_fraction: float | None = None,
    train_query_count: int | None = None,
    d2_beta: float | None = None,
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
    row.update(
        {
            "ablation": ablation,
            "split_seed": split_seed if split_seed is not None else "",
            "lambda_cost": lambda_cost if lambda_cost is not None else "",
            "train_fraction": train_fraction if train_fraction is not None else "",
            "train_query_count": train_query_count if train_query_count is not None else "",
            "d2_beta": d2_beta if d2_beta is not None else "",
        }
    )
    return row


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Ablation And Robustness"
    summary = (
        table.groupby(["ablation", "method"], as_index=False)
        .agg(
            mean_recovered_gap=("recovered_gap_vs_oracle", "mean"),
            min_recovered_gap=("recovered_gap_vs_oracle", "min"),
            max_recovered_gap=("recovered_gap_vs_oracle", "max"),
        )
        .sort_values(["ablation", "mean_recovered_gap"], ascending=[True, False])
    )
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/08_ablation_summary.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_ablation_summary.csv`: bounded seed, K/lambda, rate-penalty, and training-fraction ablation rows, including D2 RouteCode train-size rows.",
        "- `fig_sensitivity_k_lambda.pdf`: recovered-gap heatmaps over K and lambda.",
        "- `fig_seed_stability.pdf`: seed-stability bars with standard deviations.",
        "- `phase_f_g_ablation_memo.md`: robustness checkpoint memo.",
        "",
        _markdown_table(summary),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    seed = table[table["ablation"] == "seed_stability"]
    k_lambda = table[table["ablation"] == "k_lambda"]
    rate_penalty = table[table["ablation"] == "rate_penalty"]
    lines = [
        "# Phase F/G Ablation And Robustness Memo",
        "",
        f"Command: `python experiments/08_ablation_summary.py --config {config_path}`",
        "",
        "This is a bounded robustness layer, not the full ablation matrix.",
        "",
        "## Seed Stability",
        "",
        _markdown_table(
            seed.groupby("method", as_index=False)
            .agg(
                mean_gap=("recovered_gap_vs_oracle", "mean"),
                std_gap=("recovered_gap_vs_oracle", "std"),
            )
            .fillna({"std_gap": 0.0})
            .sort_values("mean_gap", ascending=False)
        ),
        "",
        "## Best K/Lambda Rows",
        "",
        _markdown_table(
            k_lambda.sort_values("recovered_gap_vs_oracle", ascending=False)
            .head(12)[["method", "K", "lambda_cost", "mean_utility", "recovered_gap_vs_oracle"]]
        ),
        "",
        "## D2 Rate Penalty",
        "",
        _markdown_table(
            rate_penalty.sort_values("d2_beta")[
                ["method", "K", "lambda_cost", "d2_beta", "mean_utility", "recovered_gap_vs_oracle", "empirical_H_Z"]
            ]
        ),
        "",
        "## Current Readout",
        "",
        "- This covers seed stability, K/lambda sensitivity through the configured K sweep, semantic vs utility-vector vs regret-objective vs predictability-constrained code-objective comparison, D2 rate-penalty sensitivity, and training-fraction sensitivity for best-single, kNN, lightweight direct routers, and D2 RouteCode.",
        "- Regret-objective RouteCode is strong as an oracle-code diagnostic, but its embedding-centroid deployable rows in `table_rate_distortion.csv` remain far below the oracle-code ceiling.",
        "- The separate Phase G sensitivity suite covers local embedding-feature variants, clustering algorithm, label noise, cost mis-estimation, bounded model-pool scenarios, query length, and bootstrap counts.",
        "- Remaining robustness work still includes external embedding backbones, broader domain granularity beyond the coarse configured domain map, broader model pools, and stronger external baselines.",
        "",
    ]
    (out_dir / "phase_f_g_ablation_memo.md").write_text("\n".join(lines), encoding="utf-8")


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
