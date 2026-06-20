from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")
FRONTIER_MODELS = {"gpt-5.5", "gemini-3.5-flash", "gemini-3.5-flash-strong-solve"}
GPT_MODELS = {"gpt-5.5"}
GEMINI_MODELS = {"gemini-3.5-flash", "gemini-3.5-flash-strong-solve"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-state cost and price sensitivity.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lambda-costs", type=float, nargs="*", default=[0.0, 0.1, 0.35, 0.7, 1.0])
    parser.add_argument("--frontier-price-multipliers", type=float, nargs="*", default=[0.5, 1.0, 2.0, 5.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = Path(config["outputs"]["root"]) / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_outputs = load_outputs(Path(config["inputs"]["broad100_outputs"]))
    groups = routecode_groups(config)
    rows = []
    action_rows = []
    for lambda_cost in args.lambda_costs:
        for multiplier in args.frontier_price_multipliers:
            outputs = adjust_utility(base_outputs, lambda_cost=lambda_cost, frontier_price_multiplier=multiplier)
            rows.extend(reference_rows(outputs, lambda_cost=lambda_cost, multiplier=multiplier))
            state_row, action_table = evaluate_state_policy(outputs, groups, lambda_cost=lambda_cost, multiplier=multiplier)
            rows.append(state_row)
            action_rows.append(action_table)

    table = pd.DataFrame(rows).sort_values(["lambda_cost", "frontier_price_multiplier", "mean_utility"], ascending=[True, True, False])
    actions = pd.concat(action_rows, ignore_index=True) if action_rows else pd.DataFrame()
    table.to_csv(out_dir / "table_price_sensitivity.csv", index=False)
    actions.to_csv(out_dir / "table_price_sensitivity_action_table.csv", index=False)
    write_figure(out_dir / "fig_price_sensitivity.pdf", table)
    write_memo(out_dir / "SENSITIVITY_MEMO.md", table, actions, config)
    print(f"Wrote cost/price sensitivity experiment to {out_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_outputs(path: Path) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs = outputs[outputs["status"].astype(str).eq("success")].copy()
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["split"] = outputs["split"].astype(str)
    outputs["benchmark"] = outputs["benchmark"].astype(str)
    outputs["quality_score"] = outputs["quality_score"].astype(float)
    outputs["normalized_remote_cost"] = outputs["normalized_remote_cost"].astype(float)
    outputs["cost_total_usd"] = outputs["cost_total_usd"].astype(float)
    outputs["latency_s"] = outputs["latency_s"].astype(float)
    return outputs


def adjust_utility(base: pd.DataFrame, *, lambda_cost: float, frontier_price_multiplier: float) -> pd.DataFrame:
    work = base.copy()
    frontier = work["model_id"].isin(FRONTIER_MODELS)
    work["adjusted_normalized_remote_cost"] = work["normalized_remote_cost"].astype(float)
    work.loc[frontier, "adjusted_normalized_remote_cost"] *= float(frontier_price_multiplier)
    work["adjusted_cost_total_usd"] = work["cost_total_usd"].astype(float)
    work.loc[frontier, "adjusted_cost_total_usd"] *= float(frontier_price_multiplier)
    work["utility"] = work["quality_score"] - float(lambda_cost) * work["adjusted_normalized_remote_cost"]
    return work


def routecode_groups(config: dict[str, Any]) -> pd.DataFrame:
    assignments = pd.read_csv(config["inputs"]["broad100_learned_verifiability_assignments"])
    assignments["query_id"] = assignments["query_id"].astype(str)
    method = str(config["method"]["compact_state_method"])
    assignments = assignments[assignments["method"].astype(str).eq(method)].drop_duplicates("query_id").copy()
    assignments["group_id"] = "z" + assignments["probe_state"].astype(int).astype(str).str.zfill(2)
    return assignments[["query_id", "split", "benchmark", "group_id"]].copy()


def reference_rows(outputs: pd.DataFrame, *, lambda_cost: float, multiplier: float) -> list[dict[str, Any]]:
    test = outputs[outputs["split"].eq("test")].copy()
    rows = []
    oracle = test.sort_values(["query_id", "utility"], ascending=[True, False]).groupby("query_id").head(1)
    rows.append(summarize(oracle, method="cost_aware_oracle", role="oracle", lambda_cost=lambda_cost, multiplier=multiplier, oracle=oracle))
    for model_id, frame in test.groupby("model_id"):
        if model_id in GPT_MODELS:
            rows.append(summarize(frame, method=f"all_{model_id}", role="all_gpt", lambda_cost=lambda_cost, multiplier=multiplier, oracle=oracle))
        elif model_id in GEMINI_MODELS:
            rows.append(summarize(frame, method=f"all_{model_id}", role="all_gemini", lambda_cost=lambda_cost, multiplier=multiplier, oracle=oracle))
    local_rows = []
    for model_id, frame in test[~test["model_id"].isin(FRONTIER_MODELS)].groupby("model_id"):
        local_rows.append(summarize(frame, method=f"all_{model_id}", role="local_model", lambda_cost=lambda_cost, multiplier=multiplier, oracle=oracle))
    if local_rows:
        rows.extend(local_rows)
        best = max(local_rows, key=lambda row: row["mean_utility"])
        best_frame = test[test["model_id"].eq(best["method"].removeprefix("all_"))].copy()
        rows.append(summarize(best_frame, method="best_local_single_model", role="local_reference", lambda_cost=lambda_cost, multiplier=multiplier, oracle=oracle))
    return rows


def evaluate_state_policy(
    outputs: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    lambda_cost: float,
    multiplier: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_groups = groups[groups["split"].eq("train")].copy()
    test_groups = groups[groups["split"].eq("test")].copy()
    train = outputs[outputs["split"].eq("train")].merge(train_groups[["query_id", "group_id"]], on="query_id", how="inner")
    test = outputs[outputs["split"].eq("test")].merge(test_groups[["query_id", "group_id"]], on="query_id", how="inner")
    means = train.groupby(["group_id", "model_id"], as_index=False).agg(mean_utility=("utility", "mean"), mean_quality=("quality_score", "mean"))
    idx = means.groupby("group_id")["mean_utility"].idxmax()
    action_table = means.loc[idx].rename(columns={"model_id": "selected_model"}).copy()
    action_table["lambda_cost"] = lambda_cost
    action_table["frontier_price_multiplier"] = multiplier
    action_table["selected_is_frontier"] = action_table["selected_model"].isin(FRONTIER_MODELS)

    selected_models = test_groups[["query_id", "group_id"]].merge(action_table[["group_id", "selected_model"]], on="group_id", how="left")
    lookup = outputs[outputs["split"].eq("test")].set_index(["query_id", "model_id"], drop=False)
    rows = []
    for item in selected_models.to_dict("records"):
        key = (str(item["query_id"]), str(item["selected_model"]))
        if key in lookup.index:
            rows.append(lookup.loc[key])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    oracle = outputs[outputs["split"].eq("test")].sort_values(["query_id", "utility"], ascending=[True, False]).groupby("query_id").head(1)
    return (
        summarize(
            selected,
            method="frozen_routecode_state_action_table",
            role="frozen_state_policy",
            lambda_cost=lambda_cost,
            multiplier=multiplier,
            oracle=oracle,
        ),
        action_table,
    )


def summarize(
    selected: pd.DataFrame,
    *,
    method: str,
    role: str,
    lambda_cost: float,
    multiplier: float,
    oracle: pd.DataFrame,
) -> dict[str, Any]:
    frontier = selected["model_id"].isin(FRONTIER_MODELS)
    oracle_utility = float(oracle["utility"].mean())
    mean_utility = float(selected["utility"].mean())
    return {
        "method": method,
        "method_role": role,
        "lambda_cost": float(lambda_cost),
        "frontier_price_multiplier": float(multiplier),
        "n_queries": int(selected["query_id"].nunique()),
        "mean_quality": float(selected["quality_score"].mean()),
        "mean_utility": mean_utility,
        "oracle_mean_utility": oracle_utility,
        "utility_gap_to_oracle": oracle_utility - mean_utility,
        "oracle_utility_ratio": mean_utility / max(oracle_utility, 1e-12),
        "remote_cost_per_1k_queries": float(selected["adjusted_cost_total_usd"].sum(skipna=True) / max(len(selected), 1) * 1000.0),
        "mean_normalized_cost": float(selected["adjusted_normalized_remote_cost"].mean()),
        "frontier_call_rate": float(frontier.mean()),
        "selected_actions_json": json.dumps(selected["model_id"].astype(str).value_counts().sort_index().to_dict(), sort_keys=True),
    }


def write_figure(path: Path, table: pd.DataFrame) -> None:
    plot = table[table["method"].eq("frozen_routecode_state_action_table")].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lambda_cost, frame in plot.groupby("lambda_cost"):
        ax.plot(frame["frontier_price_multiplier"], frame["mean_utility"], marker="o", label=f"lambda={lambda_cost:g}")
    ax.set_xlabel("Frontier price multiplier")
    ax.set_ylabel("Mean utility")
    ax.set_title("Frozen RouteCode State Price Sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_memo(path: Path, table: pd.DataFrame, actions: pd.DataFrame, config: dict[str, Any]) -> None:
    state_rows = table[table["method"].eq("frozen_routecode_state_action_table")].sort_values(
        ["lambda_cost", "frontier_price_multiplier"]
    )
    lines = [
        "# Cost And Price Sensitivity",
        "",
        "This no-call experiment freezes the query-to-state assignments and recomputes only the state-to-action table.",
        "",
        "## Inputs",
        "",
        f"- Outcome matrix: `{config['inputs']['broad100_outputs']}`",
        f"- State method: `{config['method']['compact_state_method']}`",
        "",
        "## Frozen State Policy Rows",
        "",
    ]
    for row in state_rows.to_dict("records"):
        lines.append(
            f"- lambda `{float(row['lambda_cost']):.2f}`, price x`{float(row['frontier_price_multiplier']):.1f}`: "
            f"utility `{float(row['mean_utility']):.4f}`, quality `{float(row['mean_quality']):.4f}`, "
            f"frontier rate `{float(row['frontier_call_rate']):.4f}`"
        )
    if not actions.empty:
        lines.extend(["", "## Action Table Change Summary", ""])
        summary = (
            actions.groupby(["lambda_cost", "frontier_price_multiplier"], as_index=False)
            .agg(n_states=("group_id", "nunique"), frontier_states=("selected_is_frontier", "sum"))
            .sort_values(["lambda_cost", "frontier_price_multiplier"])
        )
        for row in summary.to_dict("records"):
            lines.append(
                f"- lambda `{float(row['lambda_cost']):.2f}`, price x`{float(row['frontier_price_multiplier']):.1f}`: "
                f"`{int(row['frontier_states'])}/{int(row['n_states'])}` states select frontier actions"
            )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This tests cost-table adaptability under cached outcomes. It does not update provider pricing documents or make live calls.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

