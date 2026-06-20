from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAMBDA_COST = 0.35
FRONTIER_ACTIONS = {
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
    "strong-gpt-5.5",
}
TOOL = "deterministic_math_tool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build top-level Phase 3 exact-math summary artifacts.")
    parser.add_argument("--output-dir", default="results/controlled")
    parser.add_argument("--lambda-cost", type=float, default=LAMBDA_COST)
    return parser.parse_args()


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_current_table():
    tool = load_module("experiments/112_tool_augmented_aime_policy.py", "tool_policy")
    router = tool.load_router_module()
    table = tool.add_tool_outputs(tool.load_table(router))
    routes = pd.read_csv("results/controlled/aime_gpt_judge_with_gemini_strong/table_aime_gpt_judge_routes.csv")
    return table, routes, tool, router


def latency_for_action(row: pd.Series, action: str) -> float:
    if action == TOOL:
        return 0.0
    candidates = {
        "gemini-3.5-flash": ["gemini-3.5-flash_latency", "gemini_meta_latency_s"],
        "gpt-5.5": ["gpt-5.5_latency"],
        "gemini-3.5-flash-strong-solve": ["gemini_strong_latency"],
        "strong-gpt-5.5": ["strong_latency_s", "strong_latency"],
    }.get(action, [f"{action}_latency"])
    for column in candidates:
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])
    return 0.0


def action_quality_cost(row: pd.Series, action: str, tool) -> tuple[float, float]:
    if action == TOOL:
        return float(row["tool_quality"]), 0.0
    quality, cost = tool.row_quality_cost(row, action)
    return float(quality), float(cost)


def route_actions_for_selected_method(selected_row: pd.Series, table: pd.DataFrame, routes: pd.DataFrame, tool, router) -> pd.DataFrame:
    baseline = tool.baseline_actions(router, table)
    route_by_idx = routes.set_index("row_index")
    threshold = float(selected_row["threshold"])
    strong_cost_cap = float(selected_row["strong_cost_cap"])
    overflow = str(selected_row["overflow"])
    route_scope = str(selected_row["route_scope"])
    frame = table[table["split"].eq(str(selected_row["split"]))].copy()
    rows = []
    for idx, row in frame.iterrows():
        action = str(baseline.loc[idx])
        route_cost = 0.0
        route_latency = 0.0
        used_route = False
        if bool(row["tool_available"]):
            action = TOOL
        elif row["dataset"] == "aime" and idx in route_by_idx.index:
            if route_scope in {"all_aime", "tool_abstain_aime"}:
                route = route_by_idx.loc[idx]
                route_cost = float(route.get("route_cost", 0.0) or 0.0)
                route_latency = float(route.get("route_latency_s", 0.0) or 0.0)
                used_route = True
                if float(route.get("route_confidence", 0.0) or 0.0) >= threshold:
                    action = tool.map_route(
                        str(route["route_action"]),
                        str(baseline.loc[idx]),
                        overflow,
                        row,
                        strong_cost_cap,
                    )
        quality, solver_cost = action_quality_cost(row, action, tool)
        solver_latency = latency_for_action(row, action)
        rows.append(
            {
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "action": action,
                "quality": quality,
                "solver_cost": solver_cost,
                "route_cost": route_cost,
                "total_cost": solver_cost + route_cost,
                "solver_latency_s": solver_latency,
                "route_latency_s": route_latency if used_route else 0.0,
                "latency_s": solver_latency + (route_latency if used_route else 0.0),
                "is_frontier": action in FRONTIER_ACTIONS,
                "is_local": action not in FRONTIER_ACTIONS,
                "is_probe": bool(used_route),
            }
        )
    return pd.DataFrame(rows)


def aggregate_policy(name: str, frame: pd.DataFrame, strong_norm: float, oracle_utility: float, notes: str) -> dict[str, object]:
    quality = float(frame["quality"].mean())
    normalized_cost = float(frame["total_cost"].sum() / strong_norm)
    utility = quality - LAMBDA_COST * normalized_cost
    latencies = frame["latency_s"].astype(float)
    return {
        "method": name,
        "n_queries": int(len(frame)),
        "quality_mean": quality,
        "utility_quality_only": quality,
        "utility_cost_aware": utility,
        "utility_cost_latency_aware": utility,
        "oracle_regret": float(oracle_utility - utility),
        "remote_cost_per_query": float(frame["total_cost"].sum() / max(len(frame), 1)),
        "remote_cost_per_1k_queries": float(frame["total_cost"].sum() / max(len(frame), 1) * 1000),
        "normalized_remote_cost_vs_all_gpt": normalized_cost,
        "frontier_call_rate": float(frame["is_frontier"].mean()),
        "local_call_rate": float(frame["is_local"].mean()),
        "probe_call_rate": float(frame["is_probe"].mean()),
        "latency_mean": float(latencies.mean()),
        "latency_p50": float(latencies.quantile(0.50)),
        "latency_p95": float(latencies.quantile(0.95)),
        "latency_p99": float(latencies.quantile(0.99)),
        "notes": notes,
    }


def constant_policy_frame(table: pd.DataFrame, action: str, tool) -> pd.DataFrame:
    frame = table[table["split"].eq("test")].copy()
    rows = []
    for _, row in frame.iterrows():
        quality, cost = action_quality_cost(row, action, tool)
        rows.append(
            {
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "action": action,
                "quality": quality,
                "solver_cost": cost,
                "route_cost": 0.0,
                "total_cost": cost,
                "solver_latency_s": latency_for_action(row, action),
                "route_latency_s": 0.0,
                "latency_s": latency_for_action(row, action),
                "is_frontier": action in FRONTIER_ACTIONS,
                "is_local": action not in FRONTIER_ACTIONS,
                "is_probe": False,
            }
        )
    return pd.DataFrame(rows)


def oracle_frame(table: pd.DataFrame, tool) -> pd.DataFrame:
    frame = table[table["split"].eq("test")].copy()
    actions = [
        TOOL,
        *tool.LOCAL_MODELS,
        tool.GEMINI,
        tool.BASE_GPT,
        tool.GEMINI_STRONG,
        tool.STRONG_GPT,
    ]
    rows = []
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    scale = len(frame) / strong_norm
    for _, row in frame.iterrows():
        best = None
        for action in actions:
            if action == TOOL and not bool(row["tool_available"]):
                continue
            quality, cost = action_quality_cost(row, action, tool)
            utility = quality - LAMBDA_COST * cost * scale
            candidate = (utility, quality, -cost, action, cost)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        _, quality, _, action, cost = best
        rows.append(
            {
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "action": action,
                "quality": float(quality),
                "solver_cost": float(cost),
                "route_cost": 0.0,
                "total_cost": float(cost),
                "solver_latency_s": latency_for_action(row, str(action)),
                "route_latency_s": 0.0,
                "latency_s": latency_for_action(row, str(action)),
                "is_frontier": str(action) in FRONTIER_ACTIONS,
                "is_local": str(action) not in FRONTIER_ACTIONS,
                "is_probe": False,
            }
        )
    return pd.DataFrame(rows)


def build_main_eval(table: pd.DataFrame, routes: pd.DataFrame, tool, router, selected: pd.DataFrame) -> pd.DataFrame:
    test = table[table["split"].eq("test")].copy()
    strong_norm = max(float(test["strong_cost"].sum()), 1e-12)
    oracle = oracle_frame(table, tool)
    oracle_quality = float(oracle["quality"].mean())
    oracle_cost = float(oracle["total_cost"].sum() / strong_norm)
    oracle_utility = oracle_quality - LAMBDA_COST * oracle_cost
    rows = [
        aggregate_policy(
            "exact_math_cost_aware_oracle",
            oracle,
            strong_norm,
            oracle_utility,
            "Diagnostic outcome oracle over deterministic tools, local models, Gemini, GPT-5.5, Gemini-strong, and strong GPT.",
        )
    ]
    selected_specs = [
        ("exact_math_tool_augmented_min_cost", "validation_feasible_min_cost_test"),
        ("exact_math_tool_augmented_quality_conservative", "validation_feasible_quality_conservative_test"),
        ("exact_math_tool_augmented_best_heldout_diagnostic", "best_heldout_diagnostic_under_cost"),
    ]
    for name, rule in selected_specs:
        row = selected[selected["selection_rule"].eq(rule)].iloc[0]
        frame = route_actions_for_selected_method(row, table, routes, tool, router)
        rows.append(
            aggregate_policy(
                name,
                frame,
                strong_norm,
                oracle_utility,
                "Validation-selected deployable row." if "diagnostic" not in name else "Diagnostic row selected on held-out test.",
            )
        )
    for action, name in [
        ("qwen3-8b-local", "exact_math_all_qwen8_local"),
        ("qwen3-14b-awq-local", "exact_math_all_qwen14_local"),
        ("gemini-3.5-flash", "exact_math_all_gemini_flash"),
        ("gpt-5.5", "exact_math_all_gpt_5_5"),
        ("strong-gpt-5.5", "exact_math_all_strong_gpt_5_5"),
    ]:
        rows.append(aggregate_policy(name, constant_policy_frame(table, action, tool), strong_norm, oracle_utility, f"Route every held-out exact-math query to {action}."))
    out = pd.DataFrame(rows)
    out["benchmark_scope"] = "mixed_exact_math_test"
    out["source_artifact"] = "experiments/119_phase3_exact_math_summary.py"
    return out.sort_values(["oracle_regret", "normalized_remote_cost_vs_all_gpt"], ascending=[True, True])


def merge_top_level_main_eval(output_dir: Path, exact_eval: pd.DataFrame) -> None:
    path = output_dir / "table_main_eval.csv"
    if path.exists():
        current = pd.read_csv(path)
        current = current[~current["method"].astype(str).str.startswith("exact_math_")]
    else:
        current = pd.DataFrame(columns=exact_eval.columns)
    common_cols = list(current.columns)
    for column in exact_eval.columns:
        if column not in common_cols:
            common_cols.append(column)
    merged = pd.concat([current.reindex(columns=common_cols), exact_eval.reindex(columns=common_cols)], ignore_index=True)
    merged.to_csv(path, index=False)


def write_figures(output_dir: Path, exact_eval: pd.DataFrame, fresh: pd.DataFrame, frontier: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(
        exact_eval["normalized_remote_cost_vs_all_gpt"],
        exact_eval["quality_mean"],
        s=60,
        c=np.where(exact_eval["method"].str.contains("tool_augmented"), "#1f77b4", "#777777"),
    )
    for _, row in exact_eval.iterrows():
        if "tool_augmented" in row["method"] or "oracle" in row["method"]:
            ax.annotate(row["method"].replace("exact_math_", ""), (row["normalized_remote_cost_vs_all_gpt"], row["quality_mean"]), fontsize=7)
    ax.set_xlabel("normalized cost vs all strong GPT")
    ax.set_ylabel("held-out exact-match quality")
    ax.set_title("Phase 3 exact-math quality-cost frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_phase3_exact_math_quality_cost_frontier.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = fresh[fresh["selection_rule"].str.contains("validation_feasible", na=False)].copy()
    x = np.arange(len(rows))
    ax.bar(x - 0.15, rows["pass_rate"], width=0.3, label="pass rate")
    ax.bar(x + 0.15, rows["mean_frontier_call_rate"], width=0.3, label="mean frontier rate")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["selection_rule"].str.replace("_test", "", regex=False), rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Locked fresh-split confirmation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_phase3_exact_math_fresh_confirmation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    util = frontier[frontier["objective"].eq("utility")]
    ax.plot(util["frontier_rate_cap"], util["pass_all_rate"], marker="o", label="pass rate")
    ax.plot(util["frontier_rate_cap"], util["mean_frontier_call_rate"], marker="o", label="actual frontier rate")
    ax.set_xlabel("frontier cap")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exact-math oracle-bound frontier cap")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_phase3_exact_math_frontier_bound.pdf")
    plt.close(fig)


def write_summary(output_dir: Path, exact_eval: pd.DataFrame, fresh: pd.DataFrame, frontier: pd.DataFrame) -> None:
    min_cost = exact_eval[exact_eval["method"].eq("exact_math_tool_augmented_min_cost")].iloc[0]
    conservative = exact_eval[exact_eval["method"].eq("exact_math_tool_augmented_quality_conservative")].iloc[0]
    fresh_min = fresh[fresh["selection_rule"].eq("validation_feasible_min_cost_test")].iloc[0]
    frontier_025 = frontier[(frontier["objective"].eq("utility")) & (frontier["frontier_rate_cap"].eq(0.25))].iloc[0]
    lines = [
        "# Phase 3 Exact-Math Summary",
        "",
        "This summary is generated from cached exact-math artifacts. It makes no new API calls.",
        "",
        "Key current held-out result:",
        "",
        f"- Min-cost selected policy: quality `{min_cost.quality_mean:.4f}`, normalized cost `{min_cost.normalized_remote_cost_vs_all_gpt:.4f}`, utility `{min_cost.utility_cost_aware:.4f}`, frontier rate `{min_cost.frontier_call_rate:.4f}`.",
        f"- Quality-conservative selected policy: quality `{conservative.quality_mean:.4f}`, normalized cost `{conservative.normalized_remote_cost_vs_all_gpt:.4f}`, utility `{conservative.utility_cost_aware:.4f}`, frontier rate `{conservative.frontier_call_rate:.4f}`.",
        f"- Locked fresh split min-cost pass rate: `{fresh_min.pass_rate:.4f}` over `{int(fresh_min.n_seeds)}` seeds; max frontier rate `{fresh_min.max_frontier_call_rate:.4f}`.",
        f"- Oracle-bound utility objective at 0.25 frontier cap: pass rate `{frontier_025.pass_all_rate:.4f}`, mean quality `{frontier_025.mean_quality:.4f}`, mean frontier rate `{frontier_025.mean_frontier_call_rate:.4f}`.",
        "",
        "Scope note: this is the controlled mixed exact-math slice, not the full 8--9 benchmark Phase 3 plan.",
    ]
    (output_dir / "PHASE3_EXACT_MATH_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table, routes, tool, router = load_current_table()
    selected = pd.read_csv(output_dir / "tool_augmented_aime_policy" / "table_tool_augmented_aime_policy_selected.csv")
    fresh = pd.read_csv(output_dir / "tool_augmented_fresh_split_confirmation" / "table_locked_fresh_split_summary.csv")
    frontier = pd.read_csv(output_dir / "frontier_rate_feasibility" / "table_frontier_rate_feasibility_summary.csv")
    exact_eval = build_main_eval(table, routes, tool, router, selected)
    exact_eval.to_csv(output_dir / "table_phase3_exact_math_main_eval.csv", index=False)
    fresh.to_csv(output_dir / "table_phase3_exact_math_fresh_split_confirmation.csv", index=False)
    frontier.to_csv(output_dir / "table_phase3_exact_math_frontier_bound.csv", index=False)
    merge_top_level_main_eval(output_dir, exact_eval)
    write_figures(output_dir, exact_eval, fresh, frontier)
    write_summary(output_dir, exact_eval, fresh, frontier)
    print(f"Wrote Phase 3 exact-math summary artifacts to {output_dir}")
    print(exact_eval[["method", "quality_mean", "utility_cost_aware", "normalized_remote_cost_vs_all_gpt", "frontier_call_rate", "latency_p95"]].to_string(index=False))


if __name__ == "__main__":
    main()
