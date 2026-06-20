from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.config import load_config, output_dir
from routecode.eval.cost_quality import (
    cost_quality_frontier,
    default_cost_budgets,
    default_quality_targets,
    summarize_method_cost_quality,
)
from routecode.matrix import Matrices, build_matrices
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.cluster_lookup import EmbeddingClusterRouter
from routecode.routers.dataset_lookup import DatasetLabelRouter
from routecode.routers.knn import KNNRouter
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter, CheapestRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    cq_config = config.get("cost_quality", {})
    base_lambda = float(config.get("utility", {}).get("lambda_cost", 0.0))
    lambda_values = [float(value) for value in cq_config.get("lambda_values", [base_lambda])]
    quality_fractions = [float(value) for value in cq_config.get("quality_target_fractions", [0.8, 0.9, 0.95])]
    cost_fractions = [float(value) for value in cq_config.get("cost_budget_fractions", [0.25, 0.5, 0.75, 1.0])]
    quality_targets_config = [float(value) for value in cq_config.get("quality_targets", [])]
    cost_budgets_config = [float(value) for value in cq_config.get("cost_budgets", [])]

    summary_tables: list[pd.DataFrame] = []
    frontier_tables: list[pd.DataFrame] = []
    for lambda_cost in lambda_values:
        matrices = _matrices_by_split(prepared.outcomes, lambda_cost)
        train = matrices["train"]
        test = matrices["test"]
        selections = _method_selections(config, train, test, prepared.embeddings, lambda_cost)
        summary = summarize_method_cost_quality(test, selections, lambda_cost=lambda_cost)
        summary_tables.append(summary)
        quality_targets = quality_targets_config or default_quality_targets(summary, quality_fractions)
        cost_budgets = cost_budgets_config or default_cost_budgets(summary, cost_fractions)
        frontier_tables.extend(_frontier_tables(summary, quality_targets, cost_budgets, lambda_cost))

    summary_table = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    frontier_table = pd.concat(frontier_tables, ignore_index=True) if frontier_tables else pd.DataFrame()
    summary_table.to_csv(out_dir / "table_cost_quality_summary.csv", index=False)
    frontier_table.to_csv(out_dir / "table_cost_quality_frontier.csv", index=False)
    save_cost_quality_plot(summary_table, out_dir / "fig_cost_quality_frontier.pdf")
    write_memo(out_dir, config_path, summary_table, frontier_table)
    append_readme(out_dir, config_path, summary_table, frontier_table)
    print(f"Wrote cost-quality operating point outputs to {out_dir}")


def _matrices_by_split(outcomes: pd.DataFrame, lambda_cost: float) -> dict[str, Matrices]:
    return {
        split: build_matrices(outcomes[outcomes["split"] == split], lambda_cost=lambda_cost)
        for split in ["train", "val", "test"]
    }


def _method_selections(
    config: dict,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    lambda_cost: float,
) -> dict[str, pd.Series]:
    seed = int(config.get("run", {}).get("random_seed", 0))
    router_config = config.get("routers", {})
    d2_config = config.get("predictability_constrained", {})
    route_config = config.get("routecode", {})
    k = int(d2_config.get("k", route_config.get("selected_k_for_cards", 16)))
    alpha = float(d2_config.get("selected_alpha", d2_config.get("alpha", 1.0)))
    beta = float(d2_config.get("beta", 0.0))
    max_iter = int(d2_config.get("max_iter", route_config.get("max_iter", 25)))
    refinement_iter = int(d2_config.get("refinement_iter", 10))

    selections: dict[str, pd.Series] = {
        "best_single": BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info),
        "cheapest": CheapestRouter().fit(train.query_info, train.cost).predict(test.query_info),
        "utility_oracle": OracleRouter().predict(test.utility),
        "quality_oracle": OracleRouter().predict(test.quality),
    }

    label_column = _label_column(train.query_info)
    if label_column is not None:
        selections[f"{label_column}_label_lookup"] = DatasetLabelRouter(label_column).fit(
            train.query_info,
            train.utility,
        ).predict(test.query_info)

    if not embeddings.empty:
        cluster_k = int(router_config.get("embedding_clusters", k))
        selections["embedding_cluster_lookup"] = EmbeddingClusterRouter(cluster_k, random_state=seed).fit(
            train.query_info,
            train.utility,
            embeddings,
        ).predict(test.query_info, embeddings)
        selections["kNN"] = KNNRouter(int(router_config.get("knn_k", 15))).fit(
            train.query_info,
            train.utility,
            embeddings,
        ).predict(test.query_info, embeddings)
        d2 = PredictabilityConstrainedRouteCode(
            k,
            alpha=alpha,
            beta=beta,
            random_state=seed,
            max_iter=max_iter,
            refinement_iter=refinement_iter,
        ).fit(train.query_info, train.utility, embeddings)
        d2_labels = d2.predict_labels(embeddings.loc[test.utility.index])
        selections["d2_embedding_centroid"] = d2.predict_from_labels(d2_labels)

    return selections


def _frontier_tables(
    summary: pd.DataFrame,
    quality_targets: list[float],
    cost_budgets: list[float],
    lambda_cost: float,
) -> list[pd.DataFrame]:
    all_methods = cost_quality_frontier(
        summary,
        quality_targets=quality_targets,
        cost_budgets=cost_budgets,
        lambda_cost=lambda_cost,
    )
    all_methods.insert(1, "frontier_family", "all_methods")
    deployable_summary = summary[~summary["method"].astype(str).str.contains("oracle")].copy()
    deployable = cost_quality_frontier(
        deployable_summary,
        quality_targets=quality_targets,
        cost_budgets=cost_budgets,
        lambda_cost=lambda_cost,
    )
    deployable.insert(1, "frontier_family", "deployable_methods")
    return [all_methods, deployable]


def _label_column(query_info: pd.DataFrame) -> str | None:
    for candidate in ["dataset", "domain", "task_family"]:
        if candidate in query_info.columns:
            return candidate
    return None


def save_cost_quality_plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if not table.empty:
        for lambda_cost, group in table.groupby("lambda_cost"):
            ax.scatter(group["mean_cost"], group["mean_quality"], label=f"lambda={lambda_cost:g}", alpha=0.8)
        highlight = table[table["method"].isin(["best_single", "cheapest", "d2_embedding_centroid", "quality_oracle"])]
        for _, row in highlight.iterrows():
            ax.annotate(str(row["method"]), (row["mean_cost"], row["mean_quality"]), fontsize=7, alpha=0.8)
    ax.set_xlabel("Mean cost")
    ax.set_ylabel("Mean quality")
    ax.set_title("Cost-quality operating points")
    if not table.empty:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_memo(out_dir: Path, config_path: str, summary: pd.DataFrame, frontier: pd.DataFrame) -> None:
    lines = [
        "# Phase E Cost-Quality Operating Point Memo",
        "",
        f"Command: `python experiments/22_cost_quality_frontier.py --config {config_path}`",
        "",
        "This is a local fixed-quality and fixed-cost operating-point diagnostic. It uses existing outcome quality/cost fields and makes no external API calls.",
        "",
        "## Method Summary",
        "",
        _markdown_table(
            summary.sort_values(["lambda_cost", "mean_quality", "mean_cost"], ascending=[True, False, True]).head(20)
            if not summary.empty
            else summary
        ),
        "",
        "## Frontier Rows",
        "",
        _markdown_table(frontier.head(24) if not frontier.empty else frontier),
        "",
        "## Interpretation",
        "",
        "- `cost_at_fixed_quality` rows identify the lowest-cost method whose mean quality reaches the target.",
        "- `quality_at_fixed_cost` rows identify the highest-quality method whose mean cost stays within the budget.",
        "- Released benchmark costs include zero-cost local/open model rows for some models, so these are benchmark-metadata operating points rather than provider-price claims.",
        "",
    ]
    (out_dir / "phase_e_cost_quality_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, summary: pd.DataFrame, frontier: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Cost-Quality Operating Points"
    best = (
        summary.sort_values(["lambda_cost", "mean_quality", "mean_cost"], ascending=[True, False, True]).head(12)
        if not summary.empty
        else summary
    )
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/22_cost_quality_frontier.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_cost_quality_summary.csv`: method-level mean quality, cost, and utility for each lambda.",
        "- `table_cost_quality_frontier.csv`: fixed-quality and fixed-cost operating-point winners.",
        "- `fig_cost_quality_frontier.pdf`: cost-quality scatter plot.",
        "- `phase_e_cost_quality_memo.md`: operating-point interpretation memo.",
        "",
        _markdown_table(best),
        "",
        "Frontier preview:",
        "",
        _markdown_table(frontier.head(12) if not frontier.empty else frontier),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
