from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from routecode.codes.code_cards import write_code_cards, write_code_cards_json
from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.codes.routecode import RouteCodeCodebook
from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.predictor_diagnostics import expected_calibration_error, label_accuracy
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_code_label_heatmap, save_predictability_constrained_tradeoff
from routecode.predictors.classifiers import LogisticModelRouter, RouteCodeLabelClassifier
from routecode.reporting import upsert_markdown_section
from routecode.routers.cluster_lookup import EmbeddingClusterRouter
from routecode.routers.dataset_lookup import DatasetLabelRouter
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
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))

    k = int(d2_config.get("k", route_config.get("selected_k_for_cards", 16)))
    alpha_values = [float(alpha) for alpha in d2_config.get("alpha_values", [0.0, 0.1, 0.3, 1.0, 3.0, 10.0])]
    beta = float(d2_config.get("beta", 0.0))
    selected_alpha = float(d2_config.get("selected_alpha", alpha_values[min(3, len(alpha_values) - 1)]))
    max_iter = int(d2_config.get("max_iter", route_config.get("max_iter", 25)))
    refinement_iter = int(d2_config.get("refinement_iter", 10))

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    query_oracle = OracleRouter().predict(test.utility)
    oracle_mean = float(test.utility.max(axis=1).mean())
    knn = KNNRouter(int(config.get("routers", {}).get("knn_k", 15))).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    knn_selected = knn.predict(test.query_info, embeddings)
    logistic_selected = LogisticModelRouter(random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    ).predict(test.query_info, embeddings)
    learned_reference_mean = max(
        float(selected_values(test.utility, knn_selected).mean()),
        float(selected_values(test.utility, logistic_selected).mean()),
    )

    rows: list[dict[str, Any]] = []
    rows.extend(
        [
            _evaluated_row(
                "best_single",
                best_single,
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed,
            ),
            _evaluated_row(
                "kNN",
                knn_selected,
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed + 1,
            ),
            _evaluated_row(
                "logistic_embedding_router",
                logistic_selected,
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed + 2,
            ),
            _evaluated_row(
                "query_oracle",
                query_oracle,
                test,
                baseline_mean,
                learned_reference_mean,
                oracle_mean,
                n_bootstrap,
                ci,
                seed + 3,
            ),
        ]
    )
    _add_dataset_lookup_row(rows, train, test, baseline_mean, learned_reference_mean, oracle_mean, n_bootstrap, ci, seed)
    _add_semantic_cluster_row(rows, config, train, test, embeddings, baseline_mean, learned_reference_mean, oracle_mean, n_bootstrap, ci, seed, k)
    _add_flat_routecode_rows(rows, train, test, embeddings, baseline_mean, learned_reference_mean, oracle_mean, n_bootstrap, ci, seed, k, route_config)

    codebooks: dict[float, PredictabilityConstrainedRouteCode] = {}
    for offset, alpha in enumerate(alpha_values):
        codebook = PredictabilityConstrainedRouteCode(
            n_labels=k,
            alpha=alpha,
            beta=beta,
            random_state=seed,
            max_iter=max_iter,
            refinement_iter=refinement_iter,
        ).fit(train.query_info, train.utility, embeddings)
        codebooks[alpha] = codebook
        joint_labels = codebook.predict_joint_labels(test.utility, embeddings.loc[test.utility.index])
        joint_selected = codebook.predict_from_labels(joint_labels)
        rows.append(
            _with_d2_metrics(
                _evaluated_row(
                    "d2_joint_oracle_labels",
                    joint_selected,
                    test,
                    baseline_mean,
                    learned_reference_mean,
                    oracle_mean,
                    n_bootstrap,
                    ci,
                    seed + 100 + offset,
                    k=k,
                    labels=joint_labels,
                ),
                alpha,
                beta,
                codebook,
                joint_labels,
                joint_labels,
                pd.Series(1.0, index=joint_labels.index, name="confidence"),
            )
        )

        centroid_labels = codebook.predict_labels(embeddings.loc[test.utility.index])
        centroid_selected = codebook.predict_from_labels(centroid_labels)
        rows.append(
            _with_d2_metrics(
                _evaluated_row(
                    "d2_embedding_centroid",
                    centroid_selected,
                    test,
                    baseline_mean,
                    learned_reference_mean,
                    oracle_mean,
                    n_bootstrap,
                    ci,
                    seed + 200 + offset,
                    k=k,
                    labels=centroid_labels,
                ),
                alpha,
                beta,
                codebook,
                centroid_labels,
                joint_labels,
                codebook.predict_label_confidence(embeddings.loc[test.utility.index]),
            )
        )

        logistic = RouteCodeLabelClassifier(random_state=seed).fit(codebook, embeddings)
        logistic_labels = logistic.predict_labels(embeddings.loc[test.utility.index])
        logistic_selected = codebook.predict_from_labels(logistic_labels)
        rows.append(
            _with_d2_metrics(
                _evaluated_row(
                    "d2_logistic_label_predictor",
                    logistic_selected,
                    test,
                    baseline_mean,
                    learned_reference_mean,
                    oracle_mean,
                    n_bootstrap,
                    ci,
                    seed + 300 + offset,
                    k=k,
                    labels=logistic_labels,
                ),
                alpha,
                beta,
                codebook,
                logistic_labels,
                joint_labels,
                logistic.predict_confidence(embeddings.loc[test.utility.index]),
            )
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_predictability_constrained.csv", index=False)
    save_predictability_constrained_tradeoff(table, out_dir / "fig_predictability_constrained_tradeoff.pdf")

    selected_codebook = _select_codebook(codebooks, selected_alpha)
    write_code_cards(
        str(out_dir / "code_cards_predictability_constrained.md"),
        selected_codebook,
        train.query_info,
        train.utility,
    )
    write_code_cards_json(
        out_dir / "code_cards_predictability_constrained.json",
        selected_codebook,
        train.query_info,
        train.utility,
    )
    if selected_codebook.label_utility_ is not None:
        save_code_label_heatmap(
            selected_codebook.label_utility_,
            out_dir / "fig_code_label_heatmap_predictability_constrained.pdf",
        )
    write_method_memo(out_dir, config, table, args.config, k, selected_alpha, beta)
    append_readme(out_dir, args.config, table)
    print(f"Wrote predictability-constrained RouteCode outputs to {out_dir}")


def _evaluated_row(
    method: str,
    selected_models: pd.Series,
    test,
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
    row.update(
        {
            "alpha": "",
            "beta": "",
            "label_accuracy": "",
            "mean_confidence": "",
            "ece": "",
            "assignment_utility_loss": "",
            "assignment_embedding_loss": "",
            "assignment_balance_penalty": "",
            "assignment_objective": "",
        }
    )
    return row


def _with_d2_metrics(
    row: dict[str, Any],
    alpha: float,
    beta: float,
    codebook: PredictabilityConstrainedRouteCode,
    predicted_labels: pd.Series,
    target_labels: pd.Series,
    confidence: pd.Series,
) -> dict[str, Any]:
    correct = (predicted_labels.astype(int) == target_labels.astype(int)).astype(int)
    row.update(
        {
            "alpha": alpha,
            "beta": beta,
            "label_accuracy": label_accuracy(target_labels, predicted_labels),
            "mean_confidence": float(confidence.mean()),
            "ece": expected_calibration_error(confidence, correct, n_bins=10),
            **codebook.objective_summary(),
        }
    )
    return row


def _add_dataset_lookup_row(
    rows: list[dict[str, Any]],
    train,
    test,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> None:
    if "dataset" not in train.query_info.columns:
        return
    router = DatasetLabelRouter(label_column="dataset").fit(train.query_info, train.utility)
    selected = router.predict(test.query_info)
    rows.append(
        _evaluated_row(
            "dataset_label_lookup",
            selected,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 10,
        )
    )


def _add_semantic_cluster_row(
    rows: list[dict[str, Any]],
    config: dict,
    train,
    test,
    embeddings: pd.DataFrame,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    k: int,
) -> None:
    router = EmbeddingClusterRouter(k, random_state=seed).fit(train.query_info, train.utility, embeddings)
    labels = router.predict_labels(embeddings.loc[test.utility.index])
    selected = router.predict(test.query_info, embeddings)
    rows.append(
        _evaluated_row(
            "semantic_embedding_kmeans",
            selected,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 20,
            k=k,
            labels=labels,
        )
    )


def _add_flat_routecode_rows(
    rows: list[dict[str, Any]],
    train,
    test,
    embeddings: pd.DataFrame,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    k: int,
    route_config: dict,
) -> None:
    codebook = RouteCodeCodebook(k, random_state=seed, max_iter=int(route_config.get("max_iter", 25))).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    oracle_labels = codebook.predict_utility_labels(test.utility)
    oracle_selected = codebook.predict_from_labels(oracle_labels)
    rows.append(
        _evaluated_row(
            "flat_routecode_utility_oracle",
            oracle_selected,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 30,
            k=k,
            labels=oracle_labels,
        )
    )
    logistic = RouteCodeLabelClassifier(random_state=seed).fit(codebook, embeddings)
    labels = logistic.predict_labels(embeddings.loc[test.utility.index])
    selected = logistic.predict(test.query_info, embeddings)
    rows.append(
        _evaluated_row(
            "flat_routecode_logistic_label_predictor",
            selected,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 31,
            k=k,
            labels=labels,
        )
    )


def _select_codebook(
    codebooks: dict[float, PredictabilityConstrainedRouteCode],
    selected_alpha: float,
) -> PredictabilityConstrainedRouteCode:
    for alpha, codebook in codebooks.items():
        if np.isclose(alpha, selected_alpha):
            return codebook
    nearest_alpha = min(codebooks, key=lambda alpha: abs(alpha - selected_alpha))
    return codebooks[nearest_alpha]


def write_method_memo(
    out_dir: Path,
    config: dict,
    table: pd.DataFrame,
    config_path: str,
    k: int,
    selected_alpha: float,
    beta: float,
) -> None:
    compact = table[
        table["method"].isin(
            [
                "flat_routecode_utility_oracle",
                "flat_routecode_logistic_label_predictor",
                "d2_joint_oracle_labels",
                "d2_embedding_centroid",
                "d2_logistic_label_predictor",
                "semantic_embedding_kmeans",
                "dataset_label_lookup",
                "kNN",
            ]
        )
    ].copy()
    d2 = compact[compact["method"].astype(str).str.startswith("d2_")]
    best_deployable = (
        d2[d2["method"].isin(["d2_embedding_centroid", "d2_logistic_label_predictor"])]
        .sort_values("mean_utility", ascending=False)
        .head(1)
    )
    source = config.get("data", {}).get("source", "synthetic")
    lines = [
        "# Phase D2 Predictability-Constrained RouteCode Memo",
        "",
        f"Command: `python experiments/06_predictability_constrained.py --config {config_path}`",
        "",
        f"Data source: `{source}`. K = {k}. Selected code-card alpha = {selected_alpha:g}. Beta = {beta:g}.",
        "",
        "This is a pilot diagnostic. It should not be used as a novelty or full-benchmark claim.",
        "",
        "## Main Comparison",
        "",
        _markdown_table(
            compact[
                [
                    "method",
                    "alpha",
                    "mean_utility",
                    "oracle_regret",
                    "recovered_gap_vs_oracle",
                    "label_accuracy",
                    "empirical_H_Z",
                ]
            ].sort_values(["method", "alpha"])
        ),
        "",
        "## Current Readout",
        "",
    ]
    if best_deployable.empty:
        lines.append("- No deployable D2 row was produced.")
    else:
        row = best_deployable.iloc[0]
        lines.append(
            "- Best deployable D2 row in this sweep: "
            f"`{row['method']}` at alpha `{float(row['alpha']):g}`, mean utility `{float(row['mean_utility']):.4f}`, "
            f"label accuracy `{float(row['label_accuracy']):.4f}`."
        )
    lines.extend(
        [
            "- Interpret gains or losses against flat RouteCode and simple baselines before changing the main claim.",
            "- If D2 improves predictability but loses substantial utility, the next step is a wider alpha/K sweep before new-model calibration.",
            "",
        ]
    )
    (out_dir / "phase_d_method_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Predictability-Constrained RouteCode"
    compact = table[
        table["method"].isin(
            [
                "flat_routecode_logistic_label_predictor",
                "d2_embedding_centroid",
                "d2_logistic_label_predictor",
                "semantic_embedding_kmeans",
                "dataset_label_lookup",
                "kNN",
            ]
        )
    ][["method", "alpha", "mean_utility", "recovered_gap_vs_oracle", "label_accuracy"]].sort_values(
        ["method", "alpha"]
    )
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/06_predictability_constrained.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_predictability_constrained.csv`: alpha sweep with D2 joint-label, embedding-centroid, and logistic-label rows plus comparison baselines.",
        "- `fig_predictability_constrained_tradeoff.pdf`: D2 utility and label-predictability tradeoff by alpha.",
        "- `code_cards_predictability_constrained.md` and `code_cards_predictability_constrained.json`: code cards for the selected D2 alpha.",
        "- `fig_code_label_heatmap_predictability_constrained.pdf`: selected D2 label utility profiles.",
        "- `phase_d_method_memo.md`: D2 checkpoint memo and recommended interpretation.",
        "",
        _markdown_table(compact),
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
        values = [_format_cell(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
