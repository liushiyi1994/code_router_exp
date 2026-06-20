from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LAMBDA_COST = 0.35
TOOL = "deterministic_math_tool"
FRONTIER_ACTIONS = {
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
    "strong-gpt-5.5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-math ablation and sensitivity rows for Phase 3.")
    parser.add_argument("--output-dir", default="results/controlled")
    parser.add_argument("--lambda-cost", type=float, default=DEFAULT_LAMBDA_COST)
    parser.add_argument("--lambda-values", default="0.0,0.2,0.35,0.6")
    parser.add_argument("--frontier-price-multipliers", default="0.5,1.0,2.0,5.0")
    return parser.parse_args()


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_inputs(output_dir: Path):
    summary = load_module("experiments/119_phase3_exact_math_summary.py", "exact_summary")
    table, routes, tool, router = summary.load_current_table()
    selected = pd.read_csv(output_dir / "tool_augmented_aime_policy" / "table_tool_augmented_aime_policy_selected.csv")
    return table, routes, tool, router, selected, summary


def action_quality_cost(row: pd.Series, action: str, tool) -> tuple[float, float]:
    if action == TOOL:
        return float(row.get("tool_quality", 0.0)), 0.0
    quality, cost = tool.row_quality_cost(row, action)
    return float(0.0 if pd.isna(quality) else quality), float(0.0 if pd.isna(cost) else cost)


def latency_for_action(row: pd.Series, action: str) -> float:
    candidates = {
        "gemini-3.5-flash": ["gemini-3.5-flash_latency", "gemini_meta_latency_s"],
        "gpt-5.5": ["gpt-5.5_latency"],
        "gemini-3.5-flash-strong-solve": ["gemini_strong_latency"],
        "strong-gpt-5.5": ["strong_latency_s", "strong_latency"],
        TOOL: [],
    }.get(action, [f"{action}_latency"])
    for column in candidates:
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])
    return 0.0


def build_baseline_frame(table: pd.DataFrame, tool, router, *, use_tools: bool) -> pd.DataFrame:
    baseline = tool.baseline_actions(router, table)
    frame = table[table["split"].eq("test")].copy()
    rows = []
    for idx, row in frame.iterrows():
        action = str(baseline.loc[idx])
        if use_tools and bool(row.get("tool_available", False)):
            action = TOOL
        quality, cost = action_quality_cost(row, action, tool)
        latency = latency_for_action(row, action)
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
                "solver_latency_s": latency,
                "route_latency_s": 0.0,
                "latency_s": latency,
                "is_frontier": action in FRONTIER_ACTIONS,
                "is_local": action not in FRONTIER_ACTIONS,
                "is_probe": False,
            }
        )
    return pd.DataFrame(rows)


def build_route_frame(
    *,
    selected_row: pd.Series,
    table: pd.DataFrame,
    routes: pd.DataFrame,
    tool,
    router,
    use_tools: bool = True,
    allow_route_judge: bool = True,
    allow_gemini_strong: bool = True,
) -> pd.DataFrame:
    baseline = tool.baseline_actions(router, table)
    route_by_idx = routes.set_index("row_index")
    threshold = float(selected_row["threshold"])
    strong_cost_cap = float(selected_row["strong_cost_cap"])
    overflow = str(selected_row["overflow"])
    route_scope = str(selected_row["route_scope"])
    frame = table[table["split"].eq("test")].copy()
    rows = []
    for idx, row in frame.iterrows():
        action = str(baseline.loc[idx])
        route_cost = 0.0
        route_latency = 0.0
        used_route = False
        if use_tools and bool(row.get("tool_available", False)):
            action = TOOL
        elif allow_route_judge and row["dataset"] == "aime" and idx in route_by_idx.index:
            if route_scope in {"all_aime", "tool_abstain_aime"}:
                route = route_by_idx.loc[idx]
                route_cost = float(route.get("route_cost", 0.0) or 0.0)
                route_latency = float(route.get("route_latency_s", 0.0) or 0.0)
                used_route = True
                if float(route.get("route_confidence", 0.0) or 0.0) >= threshold:
                    route_action = str(route["route_action"])
                    if route_action == "USE_GEMINI_STRONG" and not allow_gemini_strong:
                        route_action = "USE_BASE_GPT"
                    action = tool.map_route(route_action, str(baseline.loc[idx]), overflow, row, strong_cost_cap)
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
                "route_cost": route_cost if used_route else 0.0,
                "total_cost": solver_cost + (route_cost if used_route else 0.0),
                "solver_latency_s": solver_latency,
                "route_latency_s": route_latency if used_route else 0.0,
                "latency_s": solver_latency + (route_latency if used_route else 0.0),
                "is_frontier": action in FRONTIER_ACTIONS,
                "is_local": action not in FRONTIER_ACTIONS,
                "is_probe": bool(used_route),
            }
        )
    return pd.DataFrame(rows)


def aggregate(
    method: str,
    frame: pd.DataFrame,
    *,
    strong_norm: float,
    oracle_utility: float,
    lambda_cost: float,
    notes: str,
) -> dict[str, object]:
    quality = float(frame["quality"].mean())
    normalized_cost = float(frame["total_cost"].sum() / strong_norm)
    utility = quality - float(lambda_cost) * normalized_cost
    latencies = frame["latency_s"].astype(float)
    return {
        "method": method,
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
        "benchmark_scope": "mixed_exact_math_test",
        "source_artifact": "experiments/121_phase3_exact_math_ablation_sensitivity.py",
        "action_counts": json.dumps(frame["action"].astype(str).value_counts().to_dict(), sort_keys=True),
    }


def build_ablation(args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    table, routes, tool, router, selected, summary = load_inputs(output_dir)
    test = table[table["split"].eq("test")].copy()
    strong_norm = max(float(test["strong_cost"].sum()), 1e-12)
    oracle_eval = pd.read_csv(output_dir / "table_phase3_exact_math_main_eval.csv")
    oracle_utility = float(oracle_eval.loc[oracle_eval["method"].eq("exact_math_cost_aware_oracle"), "utility_cost_aware"].iloc[0])
    min_cost_row = selected[selected["selection_rule"].eq("validation_feasible_min_cost_test")].iloc[0]
    quality_row = selected[selected["selection_rule"].eq("validation_feasible_quality_conservative_test")].iloc[0]
    frames = {
        "exact_math_full_min_cost": build_route_frame(
            selected_row=min_cost_row,
            table=table,
            routes=routes,
            tool=tool,
            router=router,
        ),
        "exact_math_full_quality_conservative": build_route_frame(
            selected_row=quality_row,
            table=table,
            routes=routes,
            tool=tool,
            router=router,
        ),
        "exact_math_ablate_no_deterministic_tools": build_route_frame(
            selected_row=min_cost_row,
            table=table,
            routes=routes,
            tool=tool,
            router=router,
            use_tools=False,
        ),
        "exact_math_ablate_no_aime_route_judge": build_baseline_frame(table, tool, router, use_tools=True),
        "exact_math_ablate_no_tools_no_route_judge": build_baseline_frame(table, tool, router, use_tools=False),
        "exact_math_ablate_no_gemini_strong_route": build_route_frame(
            selected_row=min_cost_row,
            table=table,
            routes=routes,
            tool=tool,
            router=router,
            allow_gemini_strong=False,
        ),
    }
    notes = {
        "exact_math_full_min_cost": "Validation-selected full tool-augmented policy.",
        "exact_math_full_quality_conservative": "Validation-selected quality-conservative full policy.",
        "exact_math_ablate_no_deterministic_tools": "Same selected policy with deterministic tools disabled.",
        "exact_math_ablate_no_aime_route_judge": "Deterministic tools plus train-fit baseline action table; no AIME route judge.",
        "exact_math_ablate_no_tools_no_route_judge": "Train-fit baseline action table only.",
        "exact_math_ablate_no_gemini_strong_route": "Same selected policy, but AIME route suggestions to Gemini-strong are mapped to base GPT.",
    }
    rows = [
        aggregate(
            method,
            frame,
            strong_norm=strong_norm,
            oracle_utility=oracle_utility,
            lambda_cost=args.lambda_cost,
            notes=notes[method],
        )
        for method, frame in frames.items()
    ]
    return pd.DataFrame(rows), frames


def build_sensitivity(args: argparse.Namespace, output_dir: Path, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    test = pd.read_csv(output_dir / "table_phase3_exact_math_main_eval.csv")
    oracle_utility = float(test.loc[test["method"].eq("exact_math_cost_aware_oracle"), "utility_cost_aware"].iloc[0])
    table, _, _, _, _, _ = load_inputs(output_dir)
    test_rows = table[table["split"].eq("test")].copy()
    strong_norm = max(float(test_rows["strong_cost"].sum()), 1e-12)
    lambda_values = [float(value) for value in str(args.lambda_values).split(",") if value.strip()]
    multipliers = [float(value) for value in str(args.frontier_price_multipliers).split(",") if value.strip()]
    rows = []
    selected_methods = ["exact_math_full_min_cost", "exact_math_full_quality_conservative"]
    for method in selected_methods:
        frame = frames[method]
        quality = float(frame["quality"].mean())
        base_cost = float(frame["total_cost"].sum() / strong_norm)
        for value in lambda_values:
            utility = quality - value * base_cost
            rows.append(
                {
                    "sensitivity": "lambda_cost",
                    "value": value,
                    "method": method,
                    "utility_cost_latency_aware": utility,
                    "quality_mean": quality,
                    "normalized_remote_cost_vs_all_strong": base_cost,
                    "frontier_call_rate": float(frame["is_frontier"].mean()),
                    "notes": "Exact-math cached lambda sensitivity.",
                    "benchmark_scope": "mixed_exact_math_test",
                    "source_artifact": "experiments/121_phase3_exact_math_ablation_sensitivity.py",
                }
            )
        for value in multipliers:
            adjusted_cost = base_cost * value
            utility = quality - float(args.lambda_cost) * adjusted_cost
            rows.append(
                {
                    "sensitivity": "frontier_price_multiplier",
                    "value": value,
                    "method": method,
                    "utility_cost_latency_aware": utility,
                    "quality_mean": quality,
                    "normalized_remote_cost_vs_all_strong": adjusted_cost,
                    "frontier_call_rate": float(frame["is_frontier"].mean()),
                    "notes": "Exact-math cached provider-price multiplier sensitivity.",
                    "benchmark_scope": "mixed_exact_math_test",
                    "source_artifact": "experiments/121_phase3_exact_math_ablation_sensitivity.py",
                }
            )
    del oracle_utility
    return pd.DataFrame(rows)


def merge_table(path: Path, exact: pd.DataFrame, prefix: str = "exact_math_") -> None:
    if path.exists():
        current = pd.read_csv(path)
        if "method" in current.columns:
            current = current[~current["method"].astype(str).str.startswith(prefix)]
    else:
        current = pd.DataFrame()
    columns = list(current.columns)
    for column in exact.columns:
        if column not in columns:
            columns.append(column)
    merged = pd.concat([current.reindex(columns=columns), exact.reindex(columns=columns)], ignore_index=True)
    merged.to_csv(path, index=False)


def write_figures(output_dir: Path, ablation: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    rows = ablation.sort_values("quality_mean", ascending=True)
    ax.barh(rows["method"].str.replace("exact_math_", "", regex=False), rows["quality_mean"], color="#4477aa")
    ax.set_xlabel("held-out exact-match quality")
    ax.set_title("Exact-math ablation")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_phase3_exact_math_ablation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    rows = sensitivity[sensitivity["sensitivity"].eq("frontier_price_multiplier")].copy()
    for method, group in rows.groupby("method"):
        group = group.sort_values("value")
        ax.plot(group["value"], group["utility_cost_latency_aware"], marker="o", label=method.replace("exact_math_full_", ""))
    ax.set_xlabel("frontier price multiplier")
    ax.set_ylabel("cost-aware utility")
    ax.set_title("Exact-math price sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_phase3_exact_math_sensitivity.pdf")
    plt.close(fig)


def write_memo(output_dir: Path, ablation: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    full = ablation[ablation["method"].eq("exact_math_full_min_cost")].iloc[0]
    no_tools = ablation[ablation["method"].eq("exact_math_ablate_no_deterministic_tools")].iloc[0]
    no_route = ablation[ablation["method"].eq("exact_math_ablate_no_aime_route_judge")].iloc[0]
    price_5x = sensitivity[
        sensitivity["method"].eq("exact_math_full_min_cost") & sensitivity["sensitivity"].eq("frontier_price_multiplier") & sensitivity["value"].eq(5.0)
    ].iloc[0]
    lines = [
        "# Phase 3 Exact-Math Ablation And Sensitivity Memo",
        "",
        "This memo is generated from cached exact-math artifacts and makes no new API calls.",
        "",
        "Main ablation rows:",
        "",
        f"- Full min-cost policy: quality `{full.quality_mean:.4f}`, utility `{full.utility_cost_aware:.4f}`, normalized cost `{full.normalized_remote_cost_vs_all_gpt:.4f}`, frontier rate `{full.frontier_call_rate:.4f}`.",
        f"- No deterministic tools: quality `{no_tools.quality_mean:.4f}`, utility `{no_tools.utility_cost_aware:.4f}`.",
        f"- No AIME route judge: quality `{no_route.quality_mean:.4f}`, utility `{no_route.utility_cost_aware:.4f}`.",
        "",
        "Sensitivity:",
        "",
        f"- At a 5x frontier price multiplier, the full min-cost policy utility is `{price_5x.utility_cost_latency_aware:.4f}` because deterministic tools keep normalized remote cost low.",
        "",
        "Interpretation: deterministic exact-math tools are the main component that makes the current held-out exact-math system pass the frontier-rate and utility gates. The AIME route judge and Gemini-strong route action help recover the remaining hard rows.",
    ]
    (output_dir / "PHASE3_EXACT_MATH_ABLATION_SENSITIVITY_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation, frames = build_ablation(args, output_dir)
    sensitivity = build_sensitivity(args, output_dir, frames)
    ablation.to_csv(output_dir / "table_phase3_exact_math_ablation.csv", index=False)
    sensitivity.to_csv(output_dir / "table_phase3_exact_math_sensitivity.csv", index=False)
    merge_table(output_dir / "table_ablation.csv", ablation)
    merge_table(output_dir / "table_sensitivity.csv", sensitivity)
    write_figures(output_dir, ablation, sensitivity)
    write_memo(output_dir, ablation, sensitivity)
    print(f"Wrote exact-math ablation/sensitivity artifacts to {output_dir}")
    print(ablation[["method", "quality_mean", "utility_cost_aware", "normalized_remote_cost_vs_all_gpt", "frontier_call_rate", "probe_call_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
