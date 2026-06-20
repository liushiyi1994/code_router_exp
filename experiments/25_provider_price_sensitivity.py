from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

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
from routecode.eval.provider_pricing import apply_provider_price_schedule, provider_price_coverage
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
    pricing_config = config.get("provider_pricing", {})
    schedules = list(pricing_config.get("schedules", []))
    if not schedules:
        raise ValueError("provider_pricing.schedules must contain at least one schedule")

    lambda_values = [
        float(value)
        for value in pricing_config.get(
            "lambda_values",
            config.get("cost_quality", {}).get("lambda_values", [config.get("utility", {}).get("lambda_cost", 0.0)]),
        )
    ]
    quality_fractions = [float(value) for value in pricing_config.get("quality_target_fractions", [0.8, 0.9, 0.95])]
    cost_fractions = [float(value) for value in pricing_config.get("cost_budget_fractions", [0.25, 0.5, 0.75, 1.0])]

    coverage_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    frontier_tables: list[pd.DataFrame] = []
    for schedule in schedules:
        coverage = provider_price_coverage(prepared.outcomes, schedule)
        coverage_tables.append(coverage)
        priced_outcomes = apply_provider_price_schedule(prepared.outcomes, schedule)
        if priced_outcomes["model_id"].nunique() < 2:
            continue
        for lambda_cost in lambda_values:
            matrices = _matrices_by_split(priced_outcomes, lambda_cost)
            train = matrices["train"]
            test = matrices["test"]
            if train.utility.shape[1] < 2 or test.utility.shape[1] < 2:
                continue
            selections = _method_selections(config, train, test, prepared.embeddings, lambda_cost)
            summary = summarize_method_cost_quality(test, selections, lambda_cost=lambda_cost)
            summary.insert(0, "schedule", str(schedule.get("name", "provider_schedule")))
            summary.insert(1, "provider", str(schedule.get("provider", "")))
            summary["source_checked_date"] = str(schedule.get("source_checked_date", ""))
            summary["model_coverage_count"] = int(coverage["mapped"].sum())
            summary["model_coverage_fraction"] = float(coverage["mapped"].mean()) if not coverage.empty else 0.0
            summary["mapped_model_ids"] = ",".join(sorted(priced_outcomes["model_id"].astype(str).unique()))
            summary_tables.append(summary)

            quality_targets = default_quality_targets(summary, quality_fractions)
            cost_budgets = default_cost_budgets(summary, cost_fractions)
            frontier_tables.extend(_frontier_tables(summary, quality_targets, cost_budgets, lambda_cost))

    coverage_table = pd.concat(coverage_tables, ignore_index=True) if coverage_tables else pd.DataFrame()
    summary_table = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    frontier_table = pd.concat(frontier_tables, ignore_index=True) if frontier_tables else pd.DataFrame()
    coverage_table.to_csv(out_dir / "table_provider_price_schedule.csv", index=False)
    summary_table.to_csv(out_dir / "table_provider_cost_quality_summary.csv", index=False)
    frontier_table.to_csv(out_dir / "table_provider_cost_quality_frontier.csv", index=False)
    save_provider_price_plot(summary_table, out_dir / "fig_provider_price_sensitivity.pdf")
    write_memo(out_dir, config_path, coverage_table, summary_table, frontier_table)
    append_readme(out_dir, config_path, coverage_table, summary_table, frontier_table)
    print(f"Wrote provider-price sensitivity outputs to {out_dir}")


def _matrices_by_split(outcomes: pd.DataFrame, lambda_cost: float) -> dict[str, Matrices]:
    return {
        split: build_matrices(outcomes[outcomes["split"] == split], lambda_cost=lambda_cost)
        for split in ["train", "val", "test"]
    }


def _method_selections(
    config: dict[str, Any],
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    lambda_cost: float,
) -> dict[str, pd.Series]:
    del lambda_cost
    seed = int(config.get("run", {}).get("random_seed", 0))
    router_config = config.get("routers", {})
    d2_config = config.get("predictability_constrained", {})
    route_config = config.get("routecode", {})
    k = min(int(d2_config.get("k", route_config.get("selected_k_for_cards", 16))), len(train.utility))
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
        cluster_k = max(1, min(int(router_config.get("embedding_clusters", k)), len(train.utility)))
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
            max(1, k),
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
    schedule = str(summary["schedule"].iloc[0]) if "schedule" in summary.columns and not summary.empty else ""
    provider = str(summary["provider"].iloc[0]) if "provider" in summary.columns and not summary.empty else ""
    all_methods = cost_quality_frontier(summary, quality_targets, cost_budgets, lambda_cost)
    all_methods.insert(0, "schedule", schedule)
    all_methods.insert(1, "provider", provider)
    all_methods.insert(2, "frontier_family", "all_methods")
    deployable_summary = summary[~summary["method"].astype(str).str.contains("oracle")].copy()
    deployable = cost_quality_frontier(deployable_summary, quality_targets, cost_budgets, lambda_cost)
    deployable.insert(0, "schedule", schedule)
    deployable.insert(1, "provider", provider)
    deployable.insert(2, "frontier_family", "deployable_methods")
    return [all_methods, deployable]


def _label_column(query_info: pd.DataFrame) -> str | None:
    for candidate in ["dataset", "domain", "task_family"]:
        if candidate in query_info.columns:
            return candidate
    return None


def save_provider_price_plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if not table.empty:
        for schedule, group in table.groupby("schedule"):
            ax.scatter(group["mean_cost"], group["mean_quality"], label=str(schedule), alpha=0.8)
    ax.set_xlabel("Mean provider-priced cost")
    ax.set_ylabel("Mean quality")
    ax.set_title("Provider-price sensitivity")
    if not table.empty:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_memo(
    out_dir: Path,
    config_path: str,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    frontier: pd.DataFrame,
) -> None:
    mapped = int(coverage["mapped"].sum()) if not coverage.empty else 0
    total = int(len(coverage)) if not coverage.empty else 0
    per_schedule = _coverage_summary_lines(coverage)
    lines = [
        "# Phase G Provider-Price Sensitivity Memo",
        "",
        f"Command: `python experiments/25_provider_price_sensitivity.py --config {config_path}`",
        "",
        "This is a partial provider-price schedule diagnostic. It recomputes token costs from audited provider list prices for mapped models only and makes no external API calls.",
        "",
        f"Mapped schedule-model rows: `{mapped}` of `{total}`.",
        "",
        *per_schedule,
        "",
        "## Coverage",
        "",
        _markdown_table(_coverage_preview(coverage)),
        "",
        "## Cost-Quality Summary",
        "",
        _markdown_table(_summary_preview(summary)),
        "",
        "## Frontier Rows",
        "",
        _markdown_table(frontier.head(24) if not frontier.empty else frontier),
        "",
        "## Interpretation",
        "",
        "- These rows are provider-price sensitivity diagnostics, not a full-model-pool claim.",
        "- Unmapped models are excluded under the configured `drop` policy so prices are not mixed with released benchmark metadata.",
        "- Price values are snapshots from the configured source URLs and should be refreshed before paper claims or budget decisions.",
        "",
    ]
    (out_dir / "phase_g_provider_pricing_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _coverage_summary_lines(coverage: pd.DataFrame) -> list[str]:
    if coverage.empty:
        return ["Per-schedule coverage: none."]
    rows = []
    for schedule, group in coverage.groupby("schedule", sort=True):
        mapped = int(group["mapped"].sum())
        total = int(len(group))
        rows.append(f"- `{schedule}` maps `{mapped}` of `{total}` models.")
    return rows


def append_readme(
    out_dir: Path,
    config_path: str,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    frontier: pd.DataFrame,
) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Provider-Price Sensitivity"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/25_provider_price_sensitivity.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_provider_price_schedule.csv`: provider-price coverage by model.",
        "- `table_provider_cost_quality_summary.csv`: method utilities after provider token-cost recomputation on mapped models.",
        "- `table_provider_cost_quality_frontier.csv`: fixed-quality and fixed-cost frontier rows for the provider-priced subset.",
        "- `fig_provider_price_sensitivity.pdf`: provider-priced cost-quality scatter plot.",
        "- `phase_g_provider_pricing_memo.md`: memo with scope and coverage notes.",
        "",
        _markdown_table(_coverage_preview(coverage)),
        "",
        _markdown_table(_summary_preview(summary)),
        "",
        _markdown_table(frontier.head(12) if not frontier.empty else frontier),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _coverage_preview(coverage: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "schedule",
        "provider",
        "model_id",
        "mapped",
        "provider_model_id",
        "input_price_per_million_tokens",
        "output_price_per_million_tokens",
        "source_url",
        "coverage_note",
    ]
    return coverage[[column for column in columns if column in coverage.columns]] if not coverage.empty else coverage


def _summary_preview(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "schedule",
        "lambda_cost",
        "method",
        "mean_utility",
        "mean_quality",
        "mean_cost",
        "model_coverage_count",
        "model_coverage_fraction",
    ]
    if summary.empty:
        return summary
    preview = summary.sort_values(["schedule", "lambda_cost", "mean_quality", "mean_cost"], ascending=[True, True, False, True])
    return preview[[column for column in columns if column in preview.columns]].head(20)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_markdown_cell(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    main()
