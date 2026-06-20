from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from routecode.controlled.exact_math_tools import deterministic_exact_math_answer
from routecode.controlled.live_stage0 import normalize_answer, score_output


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
STRONG_GPT = "strong-gpt-5.5"
GEMINI_STRONG = "gemini-3.5-flash-strong-solve"
TOOL = "deterministic_math_tool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic-tool + AIME judge composite policy.")
    parser.add_argument("--output-dir", default="results/controlled/tool_augmented_aime_policy")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--utility-ratio-target", type=float, default=0.95)
    parser.add_argument("--frontier-rate-target", type=float, default=0.40)
    return parser.parse_args()


def load_router_module():
    spec = importlib.util.spec_from_file_location("gemini_router", "experiments/110_gemini_strong_router.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_table(router) -> pd.DataFrame:
    base = pd.read_csv("results/controlled/strong_inclusive_oracle_audit/query_table_with_strong_inclusive_oracle.csv")
    table = router.merge_gemini_strong(
        base,
        [
            "results/controlled/gemini_strong_solver_probe_train/table_gemini_strong_solver_outputs.csv",
            "results/controlled/gemini_strong_solver_probe_val/table_gemini_strong_solver_outputs.csv",
            "results/controlled/gemini_strong_solver_probe_test/table_gemini_strong_solver_outputs.csv",
        ],
    )
    table = router.add_features(table)
    table = router.add_oracle_labels(table, 0.35)
    return table


def baseline_actions(router, table: pd.DataFrame) -> pd.Series:
    fs = router.feature_sets(table)["local"]
    train = router.prepare_features(table[table["split"].eq("train")].copy(), list(fs["cat_cols"]), list(fs["num_cols"]))
    eval_frame = router.prepare_features(
        table[table["split"].isin(["val", "test"])].copy(), list(fs["cat_cols"]), list(fs["num_cols"])
    )
    cost_scale = max(float(train["strong_cost"].sum()) / max(len(train), 1), 1e-12)
    pred = router.fit_expected_quality_router(
        train,
        eval_frame,
        feature_spec=fs,
        regressor=router.regressor_specs()["extra_trees_reg"],
        cost_scale=cost_scale,
        lambda_cost=0.35,
    )
    out = pd.Series(index=table.index, dtype=object)
    out.loc[eval_frame.index] = pred
    return out


def deterministic_tool_answer(query_text: str) -> str | None:
    return deterministic_exact_math_answer(query_text)


def add_tool_outputs(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    answers = []
    qualities = []
    for _, row in table.iterrows():
        answer = deterministic_tool_answer(str(row["query_text"]))
        answers.append(answer or "")
        if answer is None:
            qualities.append(0.0)
        else:
            _, quality = score_output(normalize_answer(answer), str(row["gold_answer"]), str(row["metric"]))
            qualities.append(float(quality))
    table["tool_answer"] = answers
    table["tool_available"] = [bool(answer) for answer in answers]
    table["tool_quality"] = qualities
    return table


def row_quality_cost(row: pd.Series, action: str) -> tuple[float, float]:
    if action == TOOL:
        return float(row["tool_quality"]), 0.0
    if action == GEMINI_STRONG:
        return float(row["gemini_strong_quality"]), float(row["gemini_strong_cost"])
    if action == STRONG_GPT:
        return float(row["strong_quality"]), float(row["strong_cost"])
    if action in LOCAL_MODELS:
        return float(row[f"{action}_quality"]), 0.0
    if action == GEMINI:
        return float(row[f"{GEMINI}_quality"]), float(row[f"{GEMINI}_cost"])
    if action == BASE_GPT:
        return float(row[f"{BASE_GPT}_quality"]), float(row[f"{BASE_GPT}_cost"])
    raise ValueError(action)


def map_route(route_action: str, baseline_action: str, strong_overflow_action: str, row: pd.Series, strong_cost_cap: float) -> str:
    overflow_action = baseline_action if strong_overflow_action == "baseline" else strong_overflow_action
    if route_action == "USE_QWEN14":
        return "qwen3-14b-awq-local"
    if route_action == "USE_GEMINI":
        return GEMINI
    if route_action == "USE_GEMINI_STRONG":
        return GEMINI_STRONG
    if route_action == "USE_BASE_GPT":
        return BASE_GPT
    if route_action == "USE_STRONG_GPT":
        if float(row["strong_cost"]) > strong_cost_cap:
            return overflow_action
        return STRONG_GPT
    return baseline_action


def evaluate(table: pd.DataFrame, baseline: pd.Series, routes: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_by_idx = routes.set_index("row_index")
    rows = []
    for threshold in [0.0, 0.4, 0.6, 0.72, 0.8, 0.95]:
        for strong_cost_cap in [0.025, 0.035, 0.05, 0.065]:
            for overflow in ["qwen3-14b-awq-local", GEMINI_STRONG, "baseline"]:
                for route_scope in ["all_aime", "tool_abstain_aime"]:
                    for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
                        actions: list[str] = []
                        qualities: list[float] = []
                        solver_costs: list[float] = []
                        total_costs: list[float] = []
                        for idx, row in frame.iterrows():
                            action = str(baseline.loc[idx])
                            route_cost = 0.0
                            if bool(row["tool_available"]):
                                action = TOOL
                            elif row["dataset"] == "aime" and idx in route_by_idx.index:
                                use_route = route_scope == "all_aime" or route_scope == "tool_abstain_aime"
                                if use_route:
                                    route = route_by_idx.loc[idx]
                                    route_cost = float(route.get("route_cost", 0.0) or 0.0)
                                    if float(route.get("route_confidence", 0.0) or 0.0) >= threshold:
                                        action = map_route(
                                            str(route["route_action"]),
                                            str(baseline.loc[idx]),
                                            overflow,
                                            row,
                                            strong_cost_cap,
                                        )
                            quality, solver_cost = row_quality_cost(row, action)
                            actions.append(action)
                            qualities.append(quality)
                            solver_costs.append(solver_cost)
                            total_costs.append(solver_cost + route_cost)
                        rows.append(
                            split_metrics(
                                frame,
                                actions,
                                qualities,
                                solver_costs,
                                total_costs,
                                threshold,
                                strong_cost_cap,
                                overflow,
                                route_scope,
                                args.lambda_cost,
                            )
                        )
    grid = pd.DataFrame(rows)
    selected = select_rows(grid, args)
    return grid, selected


def split_metrics(
    frame: pd.DataFrame,
    actions: list[str],
    qualities: list[float],
    solver_costs: list[float],
    total_costs: list[float],
    threshold: float,
    strong_cost_cap: float,
    overflow: str,
    route_scope: str,
    lambda_cost: float,
) -> dict[str, object]:
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    normalized_cost = float(np.sum(total_costs) / strong_norm)
    mean_utility = mean_quality - float(lambda_cost) * normalized_cost
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    action_counts = pd.Series(actions).value_counts().to_dict()
    frontier_actions = {GEMINI, GEMINI_STRONG, BASE_GPT, STRONG_GPT}
    return {
        "method": f"tool_aime_judge_t{threshold:g}_cap{strong_cost_cap:g}_overflow{overflow}_scope{route_scope}",
        "split": str(frame["split"].iloc[0]),
        "threshold": threshold,
        "strong_cost_cap": strong_cost_cap,
        "overflow": overflow,
        "route_scope": route_scope,
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_strong_inclusive_oracle": float(oracle_quality - mean_quality),
        "normalized_cost_vs_all_strong": normalized_cost,
        "normalized_solver_cost_vs_all_strong": float(np.sum(solver_costs) / strong_norm),
        "utility_ratio_to_strong_inclusive_oracle": float(mean_utility / oracle_utility),
        "frontier_call_rate": float(sum(action_counts.get(action, 0) for action in frontier_actions) / max(len(actions), 1)),
        "action_counts": json.dumps(action_counts, sort_keys=True),
    }


def select_rows(grid: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    val = grid[grid["split"].eq("val")].copy()
    if "frontier_call_rate" not in val.columns:
        val["frontier_call_rate"] = 0.0
        grid = grid.copy()
        grid["frontier_call_rate"] = 0.0
    rows = []
    frontier_target = float(getattr(args, "frontier_rate_target", 1.0))
    feasible = val[
        val["quality_gap_to_strong_inclusive_oracle"].le(args.quality_gap_target)
        & val["normalized_cost_vs_all_strong"].le(args.cost_target)
        & val["utility_ratio_to_strong_inclusive_oracle"].ge(args.utility_ratio_target)
        & val["frontier_call_rate"].le(frontier_target)
    ]
    if len(feasible):
        picked = feasible.sort_values(["normalized_cost_vs_all_strong", "quality_gap_to_strong_inclusive_oracle"]).head(1)
        picked_quality = feasible.sort_values(
            [
                "mean_quality",
                "quality_gap_to_strong_inclusive_oracle",
                "strong_cost_cap",
                "utility_ratio_to_strong_inclusive_oracle",
                "normalized_cost_vs_all_strong",
                "method",
            ],
            ascending=[False, True, False, False, True, True],
        ).head(1)
        selections = [
            ("validation_feasible_min_cost", picked),
            ("validation_feasible_quality_conservative", picked_quality),
        ]
    else:
        under_limits = val[
            val["normalized_cost_vs_all_strong"].le(args.cost_target)
            & val["frontier_call_rate"].le(frontier_target)
        ]
        if under_limits.empty:
            under_limits = val[val["normalized_cost_vs_all_strong"].le(args.cost_target)]
        picked = under_limits.sort_values(
            ["quality_gap_to_strong_inclusive_oracle", "frontier_call_rate", "utility_ratio_to_strong_inclusive_oracle"],
            ascending=[True, True, False],
        ).head(1)
        selections = [("no_validation_feasible_best_gap_under_cost", picked)]
    seen_methods: set[str] = set()
    for rule, picked in selections:
        if not len(picked):
            continue
        method = str(picked.iloc[0]["method"])
        if method in seen_methods:
            continue
        seen_methods.add(method)
        rows.append(picked.assign(selection_rule=rule))
        rows.append(grid[grid["method"].eq(method) & grid["split"].eq("test")].assign(selection_rule=f"{rule}_test"))
    diag = grid[grid["split"].eq("test") & grid["normalized_cost_vs_all_strong"].le(args.cost_target)].sort_values(
        ["quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle"],
        ascending=[True, False],
    ).head(1)
    if len(diag):
        rows.append(diag.assign(selection_rule="best_heldout_diagnostic_under_cost"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_memo(output_dir: Path, table: pd.DataFrame, routes: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "method",
        "split",
        "mean_quality",
        "quality_gap_to_strong_inclusive_oracle",
        "normalized_cost_vs_all_strong",
        "utility_ratio_to_strong_inclusive_oracle",
        "frontier_call_rate",
        "action_counts",
    ]
    tool_rows = table[table["tool_available"] & table["split"].isin(["val", "test"])]
    lines = [
        "# Tool-Augmented AIME Policy",
        "",
        f"Deterministic tool rows on val/test: `{len(tool_rows)}`.",
        f"AIME route rows available: `{len(routes)}`. Route cost: `${routes['route_cost'].sum():.4f}`.",
        "",
        "Selected rows:",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in selected[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    output_dir.joinpath("TOOL_AUGMENTED_AIME_POLICY_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    router = load_router_module()
    table = add_tool_outputs(load_table(router))
    baseline = baseline_actions(router, table)
    routes = pd.read_csv("results/controlled/aime_gpt_judge_with_gemini_strong/table_aime_gpt_judge_routes.csv")
    grid, selected = evaluate(table, baseline, routes, args)
    table.to_csv(output_dir / "query_table_with_tool_outputs.csv", index=False)
    grid.to_csv(output_dir / "table_tool_augmented_aime_policy.csv", index=False)
    selected.to_csv(output_dir / "table_tool_augmented_aime_policy_selected.csv", index=False)
    write_memo(output_dir, table, routes, selected)
    print(f"Wrote tool-augmented AIME policy results to {output_dir}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
