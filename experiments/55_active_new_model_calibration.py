from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.config import load_config
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.new_model_calibration import (
    budgeted_direct_oracle_labels,
    calibrate_new_model_by_active_state,
    calibrate_new_model_by_label,
    fit_predict_budgeted_direct_router,
    sample_calibration_queries_per_label,
    sample_dataset_stratified_calibration_queries,
    sample_embedding_cluster_calibration_queries,
    sample_random_calibration_queries,
)
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.single_best import BestSingleRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--source-table", default="")
    parser.add_argument("--output-dir", default="results/phase2")
    parser.add_argument("--max-holdout-models", type=int, default=1)
    parser.add_argument("--r-values", default="")
    args = parser.parse_args()
    r_values = [int(value) for value in args.r_values.split(",") if value] if args.r_values else None
    run(
        config_path=args.config or None,
        source_table_path=args.source_table or None,
        output_dir=args.output_dir,
        max_holdout_models=args.max_holdout_models,
        r_values=r_values,
    )


def run(
    *,
    output_dir: str,
    config_path: str | None = None,
    source_table_path: str | None = None,
    max_holdout_models: int = 1,
    r_values: list[int] | None = None,
) -> pd.DataFrame:
    if source_table_path:
        table = pd.read_csv(source_table_path)
        command = f"python experiments/55_active_new_model_calibration.py --source-table {source_table_path} --output-dir {output_dir}"
    elif config_path:
        table = run_active_calibration_from_config(
            config_path=config_path,
            max_holdout_models=max_holdout_models,
            r_values=r_values,
        )
        command = _command(config_path, output_dir, max_holdout_models, r_values)
    else:
        table = pd.DataFrame(
            [
                {
                    "method": "active_route_state_calibration",
                    "status": "blocked_missing_config_or_source_table",
                    "new_model_id": "",
                    "examples_per_label": "",
                    "new_model_evaluations": 0,
                    "mean_utility": pd.NA,
                    "recovered_gap_vs_oracle": pd.NA,
                }
            ]
        )
        command = f"python experiments/55_active_new_model_calibration.py --output-dir {output_dir}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, table, command)
    print(f"Wrote Phase 2 active new-model calibration table to {out_dir / 'table_active_new_model_calibration.csv'}")
    print(f"Wrote Phase 2 active new-model calibration figure to {out_dir / 'fig_new_model_calibration_curve.pdf'}")
    return table


def run_active_calibration_from_config(
    *,
    config_path: str,
    max_holdout_models: int = 1,
    r_values: list[int] | None = None,
    seed_override: int | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]
    embeddings = prepared.embeddings
    seed = int(seed_override if seed_override is not None else config.get("run", {}).get("random_seed", 0))
    route_config = config.get("routecode", {})
    d2_config = config.get("predictability_constrained", {})
    calibration_config = config.get("new_model_calibration", {})
    k = int(calibration_config.get("k", d2_config.get("k", route_config.get("selected_k_for_cards", 16))))
    alpha = float(calibration_config.get("alpha", d2_config.get("selected_alpha", 3.0)))
    beta = float(calibration_config.get("beta", d2_config.get("beta", 0.0)))
    max_iter = int(calibration_config.get("max_iter", d2_config.get("max_iter", route_config.get("max_iter", 25))))
    refinement_iter = int(calibration_config.get("refinement_iter", d2_config.get("refinement_iter", 10)))
    dataset_column = str(calibration_config.get("dataset_column", "dataset"))
    embedding_cluster_count = int(calibration_config.get("embedding_clusters", k))
    active_scout_per_state = int(calibration_config.get("active_scout_per_state", 1))
    active_prior_mean = float(calibration_config.get("active_prior_mean", 0.5))
    active_prior_strength = float(calibration_config.get("active_prior_strength", 2.0))
    active_delta = float(calibration_config.get("active_delta", 0.01))
    active_tau = float(calibration_config.get("active_tau", 0.90))
    configured_r_values = [int(value) for value in calibration_config.get("r_values", [1, 2, 4, 8, 16, 32, 64])]
    r_values = r_values or configured_r_values
    holdout_models = _holdout_models(calibration_config, train.utility, max_holdout_models=max_holdout_models)

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    oracle_mean = float(test.utility.max(axis=1).mean())
    rows: list[dict[str, Any]] = []
    for holdout_index, new_model_id in enumerate(holdout_models):
        base_models = [model for model in train.utility.columns if model != new_model_id]
        if not base_models:
            continue
        train_base_utility = train.utility.loc[:, base_models]
        codebook = PredictabilityConstrainedRouteCode(
            n_labels=k,
            alpha=alpha,
            beta=beta,
            random_state=seed + holdout_index,
            max_iter=max_iter,
            refinement_iter=refinement_iter,
        ).fit(train.query_info, train_base_utility, embeddings)
        if codebook.train_labels_ is None or codebook.label_utility_ is None or codebook.fallback_model is None:
            raise RuntimeError("D2 codebook failed to produce train labels")
        train_labels = codebook.train_labels_
        test_labels = codebook.predict_labels(embeddings.loc[test.utility.index])

        no_new_selected = codebook.predict_from_labels(test_labels)
        rows.append(
            _selection_row(
                method="routecode_no_new_model",
                selected_models=no_new_selected,
                test=test,
                baseline_mean=baseline_mean,
                oracle_mean=oracle_mean,
                new_model_id=new_model_id,
                examples_per_label=0,
                calibration_query_count=0,
                k=k,
                labels=test_labels,
            )
        )

        for r_index, examples_per_label in enumerate(r_values):
            uniform_ids = sample_calibration_queries_per_label(
                train_labels,
                examples_per_label=examples_per_label,
                seed=seed + 1000 * holdout_index + r_index,
            )
            active_calibration = calibrate_new_model_by_active_state(
                labels=train_labels,
                base_state_utility=codebook.label_utility_,
                full_utility=train.utility,
                new_model_id=new_model_id,
                total_budget=len(uniform_ids),
                scout_per_state=active_scout_per_state,
                prior_mean=active_prior_mean,
                prior_strength=active_prior_strength,
                delta=active_delta,
                tau=active_tau,
            )
            active_ids = pd.Index(active_calibration.selected_queries["query_id"], name=train_labels.index.name)
            random_ids = sample_random_calibration_queries(
                train_labels,
                total_budget=len(uniform_ids),
                seed=seed + 2500 * holdout_index + r_index,
            )
            dataset_ids = sample_dataset_stratified_calibration_queries(
                train_labels,
                train.query_info,
                total_budget=len(uniform_ids),
                seed=seed + 2750 * holdout_index + r_index,
                dataset_column=dataset_column,
            )
            embedding_ids = sample_embedding_cluster_calibration_queries(
                train_labels,
                embeddings.loc[train.utility.index],
                total_budget=len(uniform_ids),
                seed=seed + 2800 * holdout_index + r_index,
                n_clusters=embedding_cluster_count,
            )
            for method, calibration_ids in [
                ("random_route_state_calibration", random_ids),
                ("dataset_stratified_calibration", dataset_ids),
                ("embedding_cluster_calibration", embedding_ids),
                ("uniform_route_state_calibration", uniform_ids),
            ]:
                label_calibration = calibrate_new_model_by_label(
                    labels=train_labels,
                    base_label_utility=codebook.label_utility_,
                    full_utility=train.utility,
                    new_model_id=new_model_id,
                    calibration_query_ids=calibration_ids,
                )
                selected = pd.Series(
                    [
                        label_calibration.label_to_model.get(int(label), codebook.fallback_model)
                        for label in test_labels
                    ],
                    index=test_labels.index,
                    name="selected_model",
                )
                row = _selection_row(
                    method=method,
                    selected_models=selected,
                    test=test,
                    baseline_mean=baseline_mean,
                    oracle_mean=oracle_mean,
                    new_model_id=new_model_id,
                    examples_per_label=examples_per_label,
                    calibration_query_count=label_calibration.calibration_query_count,
                    k=k,
                    labels=test_labels,
                )
                row["labels_switching_to_new_model"] = sum(
                    1 for model in label_calibration.label_to_model.values() if model == new_model_id
                )
                rows.append(row)

            active_selected = pd.Series(
                [
                    active_calibration.label_to_model.get(
                        label,
                        active_calibration.label_to_model.get(int(label), codebook.fallback_model),
                    )
                    for label in test_labels
                ],
                index=test_labels.index,
                name="selected_model",
            )
            active_row = _selection_row(
                method="active_route_state_calibration",
                selected_models=active_selected,
                test=test,
                baseline_mean=baseline_mean,
                oracle_mean=oracle_mean,
                new_model_id=new_model_id,
                examples_per_label=examples_per_label,
                calibration_query_count=active_calibration.calibration_query_count,
                k=k,
                labels=test_labels,
            )
            active_row["labels_switching_to_new_model"] = sum(
                1 for model in active_calibration.label_to_model.values() if model == new_model_id
            )
            active_row["active_scout_per_state"] = active_scout_per_state
            active_row["active_delta"] = active_delta
            active_row["active_tau"] = active_tau
            rows.append(active_row)

            direct_labels = budgeted_direct_oracle_labels(
                base_utility=train_base_utility,
                full_utility=train.utility,
                new_model_id=new_model_id,
                calibration_query_ids=active_ids,
            )
            direct_selected = fit_predict_budgeted_direct_router(
                method="logistic",
                train_labels=direct_labels,
                train_embeddings=embeddings.loc[train.utility.index],
                test_embeddings=embeddings.loc[test.utility.index],
                random_state=seed + 3000 * holdout_index + r_index,
                max_iter=200,
                logistic_solver="lbfgs",
            )
            rows.append(
                _selection_row(
                    method="direct_retraining_budgeted_logistic_active_budget",
                    selected_models=direct_selected,
                    test=test,
                    baseline_mean=baseline_mean,
                    oracle_mean=oracle_mean,
                    new_model_id=new_model_id,
                    examples_per_label=examples_per_label,
                    calibration_query_count=len(active_ids),
                )
            )
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, table: pd.DataFrame, command: str) -> None:
    table_path = out_dir / "table_active_new_model_calibration.csv"
    figure_path = out_dir / "fig_new_model_calibration_curve.pdf"
    table.to_csv(table_path, index=False)
    write_calibration_figure(table, figure_path)
    write_memo(out_dir, table, command)
    append_readme(out_dir, table, command)


def write_calibration_figure(table: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if table.empty or "mean_utility" not in table.columns:
        ax.text(0.5, 0.5, "No active calibration rows", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
    else:
        plotted = False
        for method, group in table.dropna(subset=["mean_utility"]).groupby("method", sort=True):
            if "new_model_evaluations" not in group.columns:
                continue
            group = group.sort_values("new_model_evaluations")
            ax.plot(group["new_model_evaluations"], group["mean_utility"], marker="o", label=str(method))
            plotted = True
        if plotted:
            ax.set_xlabel("New-model evaluations")
            ax.set_ylabel("Mean utility")
            ax.set_title("Active New-Model Calibration")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No executable active calibration rows", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def write_memo(out_dir: Path, table: pd.DataFrame, command: str) -> None:
    lines = [
        "# Phase 2 Active New-Model Calibration",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "This compares route-state calibration strategies under matched new-model evaluation budgets.",
        "",
        "Outputs:",
        "",
        "- `table_active_new_model_calibration.csv`",
        "- `fig_new_model_calibration_curve.pdf`",
        "- `m6_active_new_model_calibration_memo.md`",
        "",
        "Best Rows:",
        "",
        _markdown_table(_best_rows(table)),
        "",
    ]
    (out_dir / "m6_active_new_model_calibration_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, table: pd.DataFrame, command: str) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Active New-Model Calibration"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "Outputs:",
        "",
        "- `table_active_new_model_calibration.csv`",
        "- `fig_new_model_calibration_curve.pdf`",
        "- `m6_active_new_model_calibration_memo.md`",
        "",
        "Best rows:",
        "",
        _markdown_table(_best_rows(table)),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _holdout_models(config: dict, utility: pd.DataFrame, *, max_holdout_models: int) -> list[str]:
    configured = config.get("holdout_models")
    if configured:
        return [str(model) for model in configured[:max_holdout_models]]
    winners = utility.idxmax(axis=1).value_counts()
    if winners.empty:
        return [str(model) for model in utility.columns[:max_holdout_models]]
    return [str(model) for model in winners.head(max_holdout_models).index]


def _selection_row(
    *,
    method: str,
    selected_models: pd.Series,
    test,
    baseline_mean: float,
    oracle_mean: float,
    new_model_id: str,
    examples_per_label: int | str,
    calibration_query_count: int,
    k: int | None = None,
    labels: pd.Series | None = None,
) -> dict[str, Any]:
    row = evaluate_selection(
        method=method,
        selected_models=selected_models,
        matrices=test,
        baseline_mean=baseline_mean,
        learned_reference_mean=baseline_mean,
        oracle_mean=oracle_mean,
        n_bootstrap=100,
        ci=0.95,
        seed=0,
        k=k,
        labels=labels,
    )
    row.update(
        {
            "new_model_id": new_model_id,
            "examples_per_label": examples_per_label,
            "calibration_query_count": int(calibration_query_count),
            "new_model_evaluations": int(calibration_query_count),
            "labels_switching_to_new_model": "",
        }
    )
    return row


def _best_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "mean_utility" not in table.columns:
        return pd.DataFrame()
    columns = [
        column
        for column in [
            "method",
            "new_model_id",
            "examples_per_label",
            "new_model_evaluations",
            "mean_utility",
            "utility_ci_low",
            "utility_ci_high",
            "recovered_gap_vs_oracle",
        ]
        if column in table.columns
    ]
    return (
        table.dropna(subset=["mean_utility"])
        .sort_values("mean_utility", ascending=False)
        .groupby("method", as_index=False)
        .head(1)[columns]
        .sort_values(["mean_utility"], ascending=False)
    )


def _command(config_path: str, output_dir: str, max_holdout_models: int, r_values: list[int] | None) -> str:
    parts = [
        "python experiments/55_active_new_model_calibration.py",
        f"--config {config_path}",
        f"--output-dir {output_dir}",
        f"--max-holdout-models {max_holdout_models}",
    ]
    if r_values:
        parts.append("--r-values " + ",".join(str(value) for value in r_values))
    return " ".join(parts)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
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
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
