from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frozen state calibration against direct router retraining.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = Path(config["outputs"]["root"])
    source = root / "new_model_onboarding/table_new_model_onboarding.csv"
    out_dir = root / "frozen_state_vs_retrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Run experiments/233_phase3_new_model_onboarding.py first: {source}")

    table = pd.read_csv(source)
    comparison = build_comparison(table)
    budget_summary = build_budget_summary(table)
    comparison.to_csv(out_dir / "table_frozen_state_vs_retrain.csv", index=False)
    budget_summary.to_csv(out_dir / "table_budget_to_match_direct.csv", index=False)
    write_figure(out_dir / "fig_budget_vs_utility.pdf", comparison)
    write_memo(out_dir / "FROZEN_STATE_VS_RETRAIN_MEMO.md", comparison, budget_summary, source)
    print(f"Wrote frozen-state vs retrain comparison to {out_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def build_comparison(table: pd.DataFrame) -> pd.DataFrame:
    keep_methods = {
        "uniform_route_state_calibration": "frozen_state_uniform",
        "active_route_state_calibration": "frozen_state_active",
        "calibration_aware_route_state": "frozen_state_calibration_aware",
        "direct_probe_regressor_retrain": "direct_router_retrain_proxy",
        "random_query_route_state_calibration": "random_query_state_calibration",
        "dataset_stratified_calibration": "dataset_stratified_calibration",
        "embedding_cluster_calibration": "embedding_cluster_calibration",
    }
    work = table[table["method"].isin(keep_methods) & table["budget"].ge(0)].copy()
    work["comparison_family"] = work["method"].map(keep_methods)
    grouped = (
        work.groupby(["comparison_family", "method", "budget"], as_index=False)
        .agg(
            n_rows=("heldout_model", "size"),
            mean_utility=("mean_utility", "mean"),
            mean_quality=("mean_quality", "mean"),
            regret_to_full_calibration=("regret_to_full_calibration", "mean"),
            n_new_model_evals=("n_new_model_evals", "mean"),
            frontier_call_rate=("frontier_call_rate", "mean"),
            remote_cost_per_1k_queries=("remote_cost_per_1k_queries", "mean"),
            training_time_s=("training_time_s", "mean"),
            new_model_selection_rate=("new_model_selection_rate", "mean"),
        )
        .sort_values(["budget", "mean_utility"], ascending=[True, False])
    )
    direct = grouped[grouped["comparison_family"].eq("direct_router_retrain_proxy")][["budget", "mean_utility"]].rename(
        columns={"mean_utility": "direct_router_mean_utility"}
    )
    grouped = grouped.merge(direct, on="budget", how="left")
    grouped["utility_delta_vs_direct_retrain"] = grouped["mean_utility"] - grouped["direct_router_mean_utility"]
    return grouped


def build_budget_summary(table: pd.DataFrame) -> pd.DataFrame:
    work = build_comparison(table)
    rows = []
    direct_by_budget = work[work["comparison_family"].eq("direct_router_retrain_proxy")].set_index("budget")["mean_utility"]
    max_budget = int(work["budget"].max())
    direct_at_max = float(direct_by_budget.loc[max_budget]) if max_budget in direct_by_budget.index else float(direct_by_budget.max())
    for family, frame in work.groupby("comparison_family"):
        eligible = frame[frame["mean_utility"].ge(direct_at_max)]
        rows.append(
            {
                "comparison_family": family,
                "target_direct_budget": max_budget,
                "target_direct_mean_utility": direct_at_max,
                "min_budget_matching_direct_at_max": int(eligible["budget"].min()) if not eligible.empty else -1,
                "best_mean_utility": float(frame["mean_utility"].max()),
                "best_budget": int(frame.loc[frame["mean_utility"].idxmax(), "budget"]),
            }
        )
    return pd.DataFrame(rows).sort_values("best_mean_utility", ascending=False)


def write_figure(path: Path, comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for family, frame in comparison.groupby("comparison_family"):
        if family in {
            "frozen_state_uniform",
            "frozen_state_active",
            "frozen_state_calibration_aware",
            "direct_router_retrain_proxy",
            "random_query_state_calibration",
        }:
            ax.plot(frame["budget"], frame["mean_utility"], marker="o", label=family)
    ax.set_xlabel("New-model calibration evaluations")
    ax.set_ylabel("Mean utility")
    ax.set_title("Frozen State Calibration vs Direct Router Retraining Proxy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_memo(path: Path, comparison: pd.DataFrame, budget_summary: pd.DataFrame, source: Path) -> None:
    lines = [
        "# Frozen State Router vs Direct Router Retraining",
        "",
        f"Source onboarding table: `{source}`",
        "",
        "This is a cache-backed proxy comparison. The direct-router row is a probe-feature utility regressor retrained with the same new-model budget.",
        "",
        "## Best Mean Utility By Family",
        "",
    ]
    best = comparison.sort_values("mean_utility", ascending=False).groupby("comparison_family").head(1)
    for row in best.to_dict("records"):
        lines.append(
            f"- `{row['comparison_family']}`: utility `{float(row['mean_utility']):.4f}` at budget `{int(row['budget'])}`, "
            f"delta vs direct same budget `{float(row['utility_delta_vs_direct_retrain']):.4f}`"
        )
    lines.extend(["", "## Budget To Match Direct At Max Budget", ""])
    for row in budget_summary.to_dict("records"):
        budget = int(row["min_budget_matching_direct_at_max"])
        budget_text = str(budget) if budget >= 0 else "not matched"
        lines.append(
            f"- `{row['comparison_family']}`: `{budget_text}`; best utility `{float(row['best_mean_utility']):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This does not replace a full learned-router baseline on live data. It is the first no-call evidence for whether frozen states can avoid full retraining.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

