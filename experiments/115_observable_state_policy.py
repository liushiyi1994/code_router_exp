from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ACTIONS = [
    "qwen3-0.6b-probe",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
    "strong-gpt-5.5",
]
LOCAL_ACTIONS = {"qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"}
TOOL = "deterministic_math_tool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train observable-state RouteCode policy from cached exact-math outcomes.")
    parser.add_argument("--output-dir", default="results/controlled/observable_state_policy")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--utility-ratio-target", type=float, default=0.95)
    parser.add_argument("--fresh-seeds", type=int, default=10)
    parser.add_argument("--first-fresh-seed", type=int, default=1000)
    return parser.parse_args()


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_base_table() -> tuple[pd.DataFrame, object, object]:
    tool = load_module("experiments/112_tool_augmented_aime_policy.py", "tool_policy")
    conf = load_module("experiments/113_locked_fresh_split_confirmation.py", "fresh_confirm")
    router = tool.load_router_module()
    table = tool.add_tool_outputs(tool.load_table(router))
    route_path = Path("results/controlled/exact_math_gpt_route_judge/table_exact_math_gpt_route_judge_routes.csv")
    if route_path.exists():
        routes = pd.read_csv(route_path)[["query_id", "route_action", "route_confidence", "route_cost"]].rename(
            columns={
                "route_action": "exact_route_action",
                "route_confidence": "exact_route_confidence",
                "route_cost": "exact_route_cost",
            }
        )
        table = table.merge(routes, on="query_id", how="left")
    else:
        table["exact_route_action"] = ""
        table["exact_route_confidence"] = 0.0
        table["exact_route_cost"] = 0.0
    table["exact_route_action"] = table["exact_route_action"].fillna("")
    table["exact_route_confidence"] = pd.to_numeric(table["exact_route_confidence"], errors="coerce").fillna(0.0)
    table["exact_route_cost"] = pd.to_numeric(table["exact_route_cost"], errors="coerce").fillna(0.0)
    return table, tool, conf


def vote_bin(value: object) -> str:
    try:
        votes = int(float(value))
    except Exception:
        votes = 0
    if votes >= 4:
        return "4"
    if votes >= 3:
        return "3"
    if votes >= 2:
        return "2"
    return "1"


def len_bin(value: object) -> str:
    try:
        length = int(float(value))
    except Exception:
        length = 0
    if length < 300:
        return "short"
    if length < 900:
        return "medium"
    return "long"


def num_bin(value: object) -> str:
    try:
        count = int(float(value))
    except Exception:
        count = 0
    if count <= 2:
        return "few"
    if count <= 6:
        return "some"
    return "many"


def route_conf_bin(value: object) -> str:
    try:
        confidence = float(value)
    except Exception:
        confidence = 0.0
    if confidence >= 0.9:
        return "hi"
    if confidence >= 0.7:
        return "mid"
    return "lo"


def state_key(row: pd.Series, variant: str) -> tuple[object, ...]:
    parts: list[object] = []
    if "nodataset" not in variant:
        parts.append(str(row["dataset"]))
    parts.extend(
        [
            bool(row.get("gemini_gpt_agree", False)),
            bool(row.get("gemini_strong_agree_base", False)),
            bool(row.get("gemini_strong_agree_gemini", False)),
            vote_bin(row.get("local_ensemble_votes", 0)),
        ]
    )
    if "shape" in variant:
        parts.extend([len_bin(row.get("query_len", 0)), num_bin(row.get("number_count", 0))])
    if "route" in variant:
        parts.extend([str(row.get("exact_route_action", "")), route_conf_bin(row.get("exact_route_confidence", 0.0))])
    return tuple(parts)


def row_quality_cost(row: pd.Series, action: str, tool) -> tuple[float, float]:
    if action == TOOL:
        return float(row["tool_quality"]), 0.0
    return tool.row_quality_cost(row, action)


def fit_policy(train: pd.DataFrame, *, variant: str, alpha: float, lambda_cost: float, tool) -> dict[str, object]:
    train = train.copy()
    train["state_key"] = [state_key(row, variant) for _, row in train.iterrows()]
    strong_norm = max(float(train["strong_cost"].sum()), 1e-12)
    global_util: dict[str, float] = {}
    for action in ACTIONS:
        utilities = []
        for _, row in train.iterrows():
            quality, cost = row_quality_cost(row, action, tool)
            utilities.append(quality - float(lambda_cost) * cost * len(train) / strong_norm)
        global_util[action] = float(np.mean(utilities))
    fallback = max(global_util, key=global_util.get)
    state_to_action: dict[tuple[object, ...], str] = {}
    for key, frame in train.groupby("state_key", sort=False):
        scores: dict[str, float] = {}
        for action in ACTIONS:
            utilities = []
            for _, row in frame.iterrows():
                quality, cost = row_quality_cost(row, action, tool)
                utilities.append(quality - float(lambda_cost) * cost * len(train) / strong_norm)
            state_sum = float(np.sum(utilities))
            scores[action] = (state_sum + float(alpha) * global_util[action]) / (len(frame) + float(alpha))
        state_to_action[tuple(key)] = max(scores, key=scores.get)
    return {"variant": variant, "alpha": alpha, "fallback": fallback, "state_to_action": state_to_action}


def predict_actions(frame: pd.DataFrame, policy: dict[str, object]) -> pd.Series:
    variant = str(policy["variant"])
    state_to_action = policy["state_to_action"]
    fallback = str(policy["fallback"])
    actions = []
    for _, row in frame.iterrows():
        if bool(row.get("tool_available", False)):
            actions.append(TOOL)
        else:
            actions.append(state_to_action.get(state_key(row, variant), fallback))
    return pd.Series(actions, index=frame.index)


def evaluate_actions(
    frame: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    variant: str,
    alpha: float,
    lambda_cost: float,
    include_route_cost: bool,
    tool,
) -> dict[str, object]:
    qualities = []
    solver_costs = []
    total_costs = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        quality, cost = row_quality_cost(row, action, tool)
        route_cost = float(row.get("exact_route_cost", 0.0) or 0.0) if include_route_cost and action != TOOL else 0.0
        qualities.append(quality)
        solver_costs.append(cost)
        total_costs.append(cost + route_cost)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    normalized_cost = float(np.sum(total_costs) / strong_norm)
    mean_utility = mean_quality - float(lambda_cost) * normalized_cost
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "variant": variant,
        "alpha": float(alpha),
        "include_route_cost": bool(include_route_cost),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_strong_inclusive_oracle": float(oracle_quality - mean_quality),
        "normalized_cost_vs_all_strong": normalized_cost,
        "normalized_solver_cost_vs_all_strong": float(np.sum(solver_costs) / strong_norm),
        "utility_ratio_to_strong_inclusive_oracle": float(mean_utility / oracle_utility),
        "action_counts": json.dumps(actions.astype(str).value_counts().to_dict(), sort_keys=True),
    }


def evaluate_grid(table: pd.DataFrame, args: argparse.Namespace, tool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = table[table["split"].eq("train")].copy()
    rows = []
    variants = [
        "agreement",
        "agreement_shape",
        "agreement_nodataset",
        "agreement_shape_nodataset",
        "agreement_route",
        "agreement_shape_route",
    ]
    alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    for variant in variants:
        include_route_cost = "route" in variant
        for alpha in alphas:
            policy = fit_policy(train, variant=variant, alpha=alpha, lambda_cost=args.lambda_cost, tool=tool)
            method = f"observable_state_{variant}_alpha{alpha:g}"
            for _, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
                actions = predict_actions(frame, policy)
                rows.append(
                    evaluate_actions(
                        frame,
                        actions,
                        method=method,
                        variant=variant,
                        alpha=alpha,
                        lambda_cost=args.lambda_cost,
                        include_route_cost=include_route_cost,
                        tool=tool,
                    )
                )
    grid = pd.DataFrame(rows)
    selected = select_rows(grid, args)
    return grid, selected


def select_rows(grid: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    val = grid[grid["split"].eq("val")].copy()
    feasible = val[
        val["quality_gap_to_strong_inclusive_oracle"].le(args.quality_gap_target)
        & val["normalized_cost_vs_all_strong"].le(args.cost_target)
        & val["utility_ratio_to_strong_inclusive_oracle"].ge(args.utility_ratio_target)
    ]
    rows = []
    if len(feasible):
        picks = [
            (
                "validation_feasible_min_cost",
                feasible.sort_values(["normalized_cost_vs_all_strong", "quality_gap_to_strong_inclusive_oracle"]).head(1),
            ),
            (
                "validation_feasible_quality_conservative",
                feasible.sort_values(
                    ["mean_quality", "quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle", "normalized_cost_vs_all_strong"],
                    ascending=[False, True, False, True],
                ).head(1),
            ),
        ]
    else:
        under_cost = val[val["normalized_cost_vs_all_strong"].le(args.cost_target)]
        picks = [
            (
                "no_validation_feasible_best_gap_under_cost",
                under_cost.sort_values(
                    ["quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle"],
                    ascending=[True, False],
                ).head(1),
            )
        ]
    seen = set()
    for rule, picked in picks:
        if not len(picked):
            continue
        method = str(picked.iloc[0]["method"])
        if method in seen:
            continue
        seen.add(method)
        rows.append(picked.assign(selection_rule=rule))
        rows.append(grid[grid["method"].eq(method) & grid["split"].eq("test")].assign(selection_rule=f"{rule}_test"))
    diag = grid[grid["split"].eq("test") & grid["normalized_cost_vs_all_strong"].le(args.cost_target)].sort_values(
        ["quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle"],
        ascending=[True, False],
    ).head(1)
    if len(diag):
        rows.append(diag.assign(selection_rule="best_heldout_diagnostic_under_cost"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pass_gate(row: pd.Series, args: argparse.Namespace) -> bool:
    return bool(
        row["quality_gap_to_strong_inclusive_oracle"] <= args.quality_gap_target
        and row["normalized_cost_vs_all_strong"] <= args.cost_target
        and row["utility_ratio_to_strong_inclusive_oracle"] >= args.utility_ratio_target
    )


def summarize_fresh(selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    tests = selected[selected["split"].eq("test")].copy()
    tests["pass_gate"] = tests.apply(lambda row: pass_gate(row, args), axis=1)
    rows = []
    for rule, frame in tests.groupby("selection_rule", sort=False):
        rows.append(
            {
                "selection_rule": rule,
                "n_seeds": int(frame["fresh_seed"].nunique()),
                "pass_rate": float(frame["pass_gate"].mean()),
                "mean_quality": float(frame["mean_quality"].mean()),
                "min_quality": float(frame["mean_quality"].min()),
                "mean_quality_gap": float(frame["quality_gap_to_strong_inclusive_oracle"].mean()),
                "max_quality_gap": float(frame["quality_gap_to_strong_inclusive_oracle"].max()),
                "mean_normalized_cost": float(frame["normalized_cost_vs_all_strong"].mean()),
                "max_normalized_cost": float(frame["normalized_cost_vs_all_strong"].max()),
                "mean_utility_ratio": float(frame["utility_ratio_to_strong_inclusive_oracle"].mean()),
                "min_utility_ratio": float(frame["utility_ratio_to_strong_inclusive_oracle"].min()),
            }
        )
    return pd.DataFrame(rows)


def write_memo(output_dir: Path, selected: pd.DataFrame, fresh_summary: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "method",
        "split",
        "mean_quality",
        "quality_gap_to_strong_inclusive_oracle",
        "normalized_cost_vs_all_strong",
        "utility_ratio_to_strong_inclusive_oracle",
        "action_counts",
    ]
    lines = [
        "# Observable-State Policy",
        "",
        "This policy learns observable agreement-state to action tables from train only, with shrinkage selected on validation.",
        "",
        "Current split selected rows:",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in selected[cols].iterrows():
        values = [f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]).replace("|", "\\|") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Fresh split summary:", "", "| " + " | ".join(fresh_summary.columns) + " |", "| " + " | ".join(["---"] * len(fresh_summary.columns)) + " |"])
    for _, row in fresh_summary.iterrows():
        values = [
            f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col])
            for col in fresh_summary.columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    output_dir.joinpath("OBSERVABLE_STATE_POLICY_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table, tool, conf = load_base_table()
    grid, selected = evaluate_grid(table, args, tool)
    grid.to_csv(output_dir / "table_observable_state_policy.csv", index=False)
    selected.to_csv(output_dir / "table_observable_state_policy_selected.csv", index=False)
    fresh_grids = []
    fresh_selected = []
    for offset in range(args.fresh_seeds):
        seed = int(args.first_fresh_seed + offset)
        fresh_table = conf.recompute_strong_inclusive_oracle(conf.stratified_resplit(table, seed), tool, args.lambda_cost)
        fresh_grid, fresh_sel = evaluate_grid(fresh_table, args, tool)
        fresh_grid.insert(0, "fresh_seed", seed)
        fresh_sel.insert(0, "fresh_seed", seed)
        fresh_grids.append(fresh_grid)
        fresh_selected.append(fresh_sel)
    fresh_grid_df = pd.concat(fresh_grids, ignore_index=True)
    fresh_selected_df = pd.concat(fresh_selected, ignore_index=True)
    fresh_summary = summarize_fresh(fresh_selected_df, args)
    fresh_grid_df.to_csv(output_dir / "table_observable_state_policy_fresh_grid.csv", index=False)
    fresh_selected_df.to_csv(output_dir / "table_observable_state_policy_fresh_selected.csv", index=False)
    fresh_summary.to_csv(output_dir / "table_observable_state_policy_fresh_summary.csv", index=False)
    write_memo(output_dir, selected, fresh_summary)
    print(f"Wrote observable-state policy results to {output_dir}")
    print(selected.to_string(index=False))
    print(fresh_summary.to_string(index=False))


if __name__ == "__main__":
    main()
