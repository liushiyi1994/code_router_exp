from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_model_win_distribution(winners: pd.Series, path: str | Path) -> None:
    counts = winners.value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_ylabel("Query count")
    ax.set_xlabel("Oracle winning model")
    ax.set_title("Synthetic Oracle Model-Win Distribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_oracle_gap_by_dataset(table: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table["dataset"], table["oracle_gap"], color="#F58518")
    ax.set_ylabel("Oracle gap vs best single")
    ax.set_xlabel("Dataset")
    ax.set_title("Synthetic Oracle Gap by Dataset")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_compression_ladder(table: pd.DataFrame, path: str | Path) -> None:
    plot_table = table[table["method"] != "query_oracle"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    errors = [
        plot_table["mean_utility"] - plot_table["utility_ci_low"],
        plot_table["utility_ci_high"] - plot_table["mean_utility"],
    ]
    ax.bar(plot_table["method"], plot_table["mean_utility"], yerr=errors, color="#54A24B", capsize=3)
    ax.set_ylabel("Mean utility")
    ax.set_xlabel("Router")
    ax.set_title("Compression Ladder, Synthetic Test Split")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_rate_distortion(table: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, group in table.groupby("method"):
        if method in {"best_single", "query_oracle", "kNN"}:
            continue
        group = group.sort_values("rate_log2K")
        axes[0].plot(group["rate_log2K"], group["oracle_regret"], marker="o", label=method)
        axes[1].plot(group["rate_log2K"], group["recovered_gap_vs_oracle"], marker="o", label=method)
    axes[0].set_xlabel("Rate log2(K)")
    axes[0].set_ylabel("Oracle regret")
    axes[0].set_title("Rate-Distortion")
    axes[1].set_xlabel("Rate log2(K)")
    axes[1].set_ylabel("Recovered gap vs oracle")
    axes[1].set_title("Recovered Gap")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_residual_concentration(table: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["top_fraction"] * 100, table["regret_mass_fraction"] * 100, marker="o", color="#E45756")
    ax.set_xlabel("Top-regret queries (%)")
    ax.set_ylabel("Total regret mass (%)")
    ax.set_title("Residual Regret Concentration")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_risk_coverage(table: pd.DataFrame, path: str | Path) -> None:
    plot_table = table.copy()
    if plot_table.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for score, group in plot_table.groupby("score"):
        group = group.sort_values("top_fraction")
        ax.plot(group["top_fraction"] * 100, group["regret_mass_fraction"] * 100, marker="o", label=score)
    ax.set_xlabel("Top-risk queries flagged (%)")
    ax.set_ylabel("Total regret mass captured (%)")
    ax.set_title("Residual Risk Coverage")
    ax.set_ylim(0, 105)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_split_sensitivity(table: pd.DataFrame, path: str | Path) -> None:
    subset = table[~table["method"].isin(["query_oracle"])].copy()
    if subset.empty:
        return
    pivot = subset.pivot_table(
        index="scenario",
        columns="method",
        values="recovered_gap_vs_oracle",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Split Sensitivity: Recovered Gap vs Oracle")
    fig.colorbar(image, ax=ax, label="Recovered gap vs oracle")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_utility_weighted_confusion(table: pd.DataFrame, path: str | Path, predictor: str | None = None) -> None:
    plot_table = table.copy()
    if predictor is not None and "predictor" in plot_table.columns:
        plot_table = plot_table[plot_table["predictor"] == predictor]
    if plot_table.empty:
        return
    pivot = plot_table.pivot_table(
        index="true_label",
        columns="predicted_label",
        values="mean_regret",
        aggfunc="mean",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Predicted route label")
    ax.set_ylabel("Utility-oracle route label")
    title = "Utility-Weighted Label Confusion"
    if predictor:
        title += f": {predictor}"
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Mean utility regret")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_calibration_curve(table: pd.DataFrame, path: str | Path) -> None:
    plot_table = table.dropna(subset=["mean_confidence", "accuracy"]).copy()
    if plot_table.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for predictor, group in plot_table.groupby("predictor"):
        group = group.sort_values("bin_low")
        ax.plot(group["mean_confidence"], group["accuracy"], marker="o", label=predictor)
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical label accuracy")
    ax.set_title("Route Label Predictor Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_code_label_heatmap(label_utility: pd.DataFrame, path: str | Path) -> None:
    if label_utility.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(label_utility))))
    image = ax.imshow(label_utility.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(label_utility.columns)))
    ax.set_xticklabels(label_utility.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(label_utility.index)))
    ax.set_yticklabels(label_utility.index)
    ax.set_xlabel("Model")
    ax.set_ylabel("Route label")
    ax.set_title("RouteCode Label Utility Profiles")
    fig.colorbar(image, ax=ax, label="Mean train utility")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_predictability_constrained_tradeoff(table: pd.DataFrame, path: str | Path) -> None:
    plot_table = table[table["method"].isin(["d2_embedding_centroid", "d2_logistic_label_predictor"])].copy()
    if plot_table.empty:
        return
    plot_table = plot_table.sort_values(["method", "alpha"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, group in plot_table.groupby("method"):
        axes[0].plot(group["alpha"], group["mean_utility"], marker="o", label=method)
        axes[1].plot(group["alpha"], group["label_accuracy"], marker="o", label=method)
    axes[0].set_xscale("symlog", linthresh=0.05)
    axes[1].set_xscale("symlog", linthresh=0.05)
    axes[0].set_xlabel("Predictability weight alpha")
    axes[0].set_ylabel("Mean utility")
    axes[0].set_title("Utility Under D2 Labels")
    axes[1].set_xlabel("Predictability weight alpha")
    axes[1].set_ylabel("Label accuracy vs joint labels")
    axes[1].set_title("Route Label Predictability")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_transfer_calibration_curve(table: pd.DataFrame, path: str | Path) -> None:
    plot_table = table[
        (table["method"] == "routecode_label_calibration")
        | table["method"].astype(str).str.startswith("direct_retraining_budgeted_")
    ].copy()
    if plot_table.empty:
        return
    grouped = (
        plot_table.groupby(["method", "examples_per_label"], as_index=False)
        .agg(
            mean_utility=("mean_utility", "mean"),
            calibration_query_count=("calibration_query_count", "mean"),
        )
        .sort_values(["method", "examples_per_label"])
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, group in grouped.groupby("method"):
        ax.plot(
            group["calibration_query_count"],
            group["mean_utility"],
            marker="o",
            label=method,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("New-model calibration evaluations")
    ax.set_ylabel("Mean utility")
    ax.set_title("New-Model Calibration")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_sensitivity_k_lambda(table: pd.DataFrame, path: str | Path) -> None:
    plot_table = table[table["ablation"] == "k_lambda"].copy()
    if plot_table.empty:
        return
    plot_table["K"] = pd.to_numeric(plot_table["K"], errors="coerce")
    plot_table["lambda_cost"] = pd.to_numeric(plot_table["lambda_cost"], errors="coerce")
    plot_table = plot_table.dropna(subset=["K", "lambda_cost", "recovered_gap_vs_oracle"])
    if plot_table.empty:
        return
    methods = list(plot_table["method"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(max(5, 4 * len(methods)), 4),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, method in zip(axes[0], methods):
        group = plot_table[plot_table["method"] == method]
        pivot = group.pivot_table(
            index="lambda_cost",
            columns="K",
            values="recovered_gap_vs_oracle",
            aggfunc="mean",
        ).sort_index()
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(column)) for column in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{value:g}" for value in pivot.index])
        ax.set_xlabel("K")
        ax.set_ylabel("lambda cost")
        ax.set_title(method)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Recovered gap vs oracle")
    fig.savefig(path)
    plt.close(fig)


def save_seed_stability(table: pd.DataFrame, path: str | Path) -> None:
    plot_table = table[table["ablation"] == "seed_stability"].copy()
    if plot_table.empty:
        return
    summary = (
        plot_table.groupby("method", as_index=False)
        .agg(
            mean_gap=("recovered_gap_vs_oracle", "mean"),
            std_gap=("recovered_gap_vs_oracle", "std"),
        )
        .fillna({"std_gap": 0.0})
        .sort_values("mean_gap", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["method"], summary["mean_gap"], yerr=summary["std_gap"], capsize=3, color="#4C78A8")
    ax.set_ylabel("Recovered gap vs oracle")
    ax.set_xlabel("Method")
    ax.set_title("Seed Stability")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_sensitivity_summary(table: pd.DataFrame, path: str | Path) -> None:
    if table.empty or "sensitivity" not in table.columns:
        return
    summary = (
        table.groupby(["sensitivity", "method"], as_index=False)
        .agg(mean_gap=("recovered_gap_vs_oracle", "mean"))
        .sort_values(["sensitivity", "mean_gap"], ascending=[True, False])
    )
    if summary.empty:
        return
    pivot = summary.pivot_table(
        index="sensitivity",
        columns="method",
        values="mean_gap",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Sensitivity Summary")
    fig.colorbar(image, ax=ax, label="Mean recovered gap vs oracle")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
