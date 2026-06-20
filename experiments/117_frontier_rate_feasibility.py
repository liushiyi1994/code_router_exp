from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_ACTIONS = [
    "deterministic_math_tool",
    "qwen3-0.6b-probe",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
]
FRONTIER_ACTIONS = [
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
    "strong-gpt-5.5",
]
ALL_ACTIONS = LOCAL_ACTIONS + FRONTIER_ACTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute frontier-call-rate constrained oracle feasibility on cached exact-math rows."
    )
    parser.add_argument("--output-dir", default="results/controlled/frontier_rate_feasibility")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--utility-ratio-target", type=float, default=0.95)
    parser.add_argument("--fresh-seeds", type=int, default=10)
    parser.add_argument("--first-fresh-seed", type=int, default=1000)
    parser.add_argument("--frontier-rate-caps", default="0.25,0.40,0.60,0.80,1.0")
    return parser.parse_args()


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_table() -> tuple[pd.DataFrame, object, object]:
    tool = load_module("experiments/112_tool_augmented_aime_policy.py", "tool_policy")
    conf = load_module("experiments/113_locked_fresh_split_confirmation.py", "fresh_confirm")
    router = tool.load_router_module()
    table = tool.add_tool_outputs(tool.load_table(router))
    return table, tool, conf


def parse_rates(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def row_quality_cost(row: pd.Series, action: str, tool) -> tuple[float, float]:
    quality, cost = tool.row_quality_cost(row, action)
    quality = 0.0 if pd.isna(quality) else float(quality)
    cost = 0.0 if pd.isna(cost) else float(cost)
    return quality, cost


def action_table(frame: pd.DataFrame, tool, lambda_cost: float) -> dict[str, pd.DataFrame]:
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    n_queries = max(int(len(frame)), 1)
    out: dict[str, pd.DataFrame] = {}
    for action in ALL_ACTIONS:
        qualities = []
        costs = []
        for _, row in frame.iterrows():
            quality, cost = row_quality_cost(row, action, tool)
            if action == "deterministic_math_tool" and not bool(row.get("tool_available", False)):
                quality = -1e9
            qualities.append(quality)
            costs.append(cost)
        stats = pd.DataFrame(index=frame.index)
        stats["action"] = action
        stats["quality"] = np.asarray(qualities, dtype=float)
        stats["cost"] = np.asarray(costs, dtype=float)
        stats["row_utility"] = stats["quality"] - float(lambda_cost) * stats["cost"] * n_queries / strong_norm
        stats["is_frontier"] = action in FRONTIER_ACTIONS
        out[action] = stats
    return out


def best_by_objective(stats: dict[str, pd.DataFrame], actions: list[str], objective: str) -> pd.DataFrame:
    frames = [stats[action] for action in actions]
    stack = pd.concat(frames, axis=0)
    if objective == "utility":
        ordered = stack.sort_values(["row_utility", "quality", "cost", "action"], ascending=[False, False, True, True])
    elif objective == "quality":
        ordered = stack.sort_values(["quality", "row_utility", "cost", "action"], ascending=[False, False, True, True])
    else:
        raise ValueError(objective)
    return ordered.groupby(level=0, sort=False).head(1).sort_index()


def aggregate(
    frame: pd.DataFrame,
    chosen: pd.DataFrame,
    target: pd.DataFrame,
    *,
    split_context: str,
    fresh_seed: int | None,
    objective: str,
    frontier_rate_cap: float,
    lambda_cost: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    quality = float(chosen["quality"].mean())
    target_quality = float(target["quality"].mean())
    normalized_cost = float(chosen["cost"].sum() / strong_norm)
    target_normalized_cost = float(target["cost"].sum() / strong_norm)
    utility = quality - float(lambda_cost) * normalized_cost
    target_utility = target_quality - float(lambda_cost) * target_normalized_cost
    frontier_rate = float(chosen["is_frontier"].mean())
    row = {
        "split_context": split_context,
        "fresh_seed": fresh_seed,
        "split": str(frame["split"].iloc[0]),
        "objective": objective,
        "frontier_rate_cap": float(frontier_rate_cap),
        "frontier_call_cap": int(np.floor(float(frontier_rate_cap) * len(frame) + 1e-12)),
        "n_queries": int(len(frame)),
        "mean_quality": quality,
        "target_oracle_quality": target_quality,
        "quality_gap_to_target_oracle": float(target_quality - quality),
        "normalized_cost_vs_all_strong": normalized_cost,
        "target_oracle_normalized_cost_vs_all_strong": target_normalized_cost,
        "mean_utility": utility,
        "target_oracle_utility": target_utility,
        "utility_ratio_to_target_oracle": float(utility / target_utility) if abs(target_utility) > 1e-12 else np.nan,
        "frontier_call_rate": frontier_rate,
        "action_counts": json.dumps(chosen["action"].value_counts().to_dict(), sort_keys=True),
    }
    row["pass_quality_gate"] = bool(row["quality_gap_to_target_oracle"] <= float(args.quality_gap_target))
    row["pass_cost_gate"] = bool(row["normalized_cost_vs_all_strong"] <= float(args.cost_target))
    row["pass_utility_gate"] = bool(row["utility_ratio_to_target_oracle"] >= float(args.utility_ratio_target))
    row["pass_frontier_gate"] = bool(row["frontier_call_rate"] <= float(frontier_rate_cap) + 1e-12)
    row["pass_all_phase3_gates"] = bool(
        row["pass_quality_gate"] and row["pass_cost_gate"] and row["pass_utility_gate"] and row["pass_frontier_gate"]
    )
    return row


def constrained_oracle(
    frame: pd.DataFrame,
    *,
    tool,
    objective: str,
    frontier_rate_cap: float,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = action_table(frame, tool, lambda_cost)
    local = best_by_objective(stats, LOCAL_ACTIONS, objective)
    frontier = best_by_objective(stats, FRONTIER_ACTIONS, objective)
    if objective == "utility":
        gain = frontier["row_utility"] - local["row_utility"]
    else:
        gain = frontier["quality"] - local["quality"]
    chosen = local.copy()
    call_cap = int(np.floor(float(frontier_rate_cap) * len(frame) + 1e-12))
    if call_cap > 0:
        candidates = gain[gain > 1e-12].sort_values(ascending=False).head(call_cap)
        if len(candidates):
            chosen.loc[candidates.index, :] = frontier.loc[candidates.index, :]
    target = best_by_objective(stats, ALL_ACTIONS, "utility")
    return chosen.sort_index(), target.sort_index()


def evaluate_split(
    table: pd.DataFrame,
    *,
    tool,
    rates: list[float],
    split_context: str,
    fresh_seed: int | None,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    frame = table[table["split"].eq("test")].copy()
    rows = []
    for objective in ["utility", "quality"]:
        for rate in rates:
            chosen, target = constrained_oracle(
                frame,
                tool=tool,
                objective=objective,
                frontier_rate_cap=rate,
                lambda_cost=args.lambda_cost,
            )
            rows.append(
                aggregate(
                    frame,
                    chosen,
                    target,
                    split_context=split_context,
                    fresh_seed=fresh_seed,
                    objective=objective,
                    frontier_rate_cap=rate,
                    lambda_cost=args.lambda_cost,
                    args=args,
                )
            )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    fresh = results[results["split_context"].eq("fresh")].copy()
    rows = []
    for (objective, cap), frame in fresh.groupby(["objective", "frontier_rate_cap"], sort=False):
        rows.append(
            {
                "objective": objective,
                "frontier_rate_cap": float(cap),
                "n_splits": int(len(frame)),
                "pass_all_rate": float(frame["pass_all_phase3_gates"].mean()),
                "pass_quality_rate": float(frame["pass_quality_gate"].mean()),
                "mean_quality": float(frame["mean_quality"].mean()),
                "min_quality": float(frame["mean_quality"].min()),
                "mean_quality_gap": float(frame["quality_gap_to_target_oracle"].mean()),
                "max_quality_gap": float(frame["quality_gap_to_target_oracle"].max()),
                "mean_normalized_cost": float(frame["normalized_cost_vs_all_strong"].mean()),
                "max_normalized_cost": float(frame["normalized_cost_vs_all_strong"].max()),
                "mean_utility_ratio": float(frame["utility_ratio_to_target_oracle"].mean()),
                "min_utility_ratio": float(frame["utility_ratio_to_target_oracle"].min()),
                "mean_frontier_call_rate": float(frame["frontier_call_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def min_needed(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (context, seed, objective), frame in results.groupby(["split_context", "fresh_seed", "objective"], dropna=False):
        passing = frame[frame["pass_all_phase3_gates"]].sort_values("frontier_rate_cap")
        quality_passing = frame[frame["pass_quality_gate"]].sort_values("frontier_rate_cap")
        rows.append(
            {
                "split_context": context,
                "fresh_seed": None if pd.isna(seed) else int(seed),
                "objective": objective,
                "min_cap_for_all_gates": float(passing["frontier_rate_cap"].iloc[0]) if len(passing) else np.nan,
                "min_cap_for_quality_gate": float(quality_passing["frontier_rate_cap"].iloc[0])
                if len(quality_passing)
                else np.nan,
                "target_oracle_quality": float(frame["target_oracle_quality"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_memo(output_dir: Path, results: pd.DataFrame, summary: pd.DataFrame, needed: pd.DataFrame) -> None:
    current = results[results["split_context"].eq("original")].copy()
    current_040 = current[current["frontier_rate_cap"].eq(0.40)].sort_values("objective")
    current_060 = current[current["frontier_rate_cap"].eq(0.60)].sort_values("objective")
    fresh_040 = summary[summary["frontier_rate_cap"].eq(0.40)].sort_values("objective")
    lines = [
        "# Frontier-Rate Feasibility Bound",
        "",
        "This is an oracle-bound diagnostic over cached exact-math outcomes. It is not a deployable router.",
        "For each test split, the bound first picks the best local/tool action, then spends a limited number",
        "of frontier calls on rows with the largest oracle-visible gain.",
        "",
        "Candidate actions: deterministic tool, Qwen local/probe models, Gemini 3.5 Flash, GPT-5.5,",
        "Gemini 3.5 Flash strong solve, and GPT-5.5 medium strong solve. Claude was not used.",
        "",
        "Current split at 0.40 frontier cap:",
        "",
        "| objective | quality | gap | cost | utility_ratio | frontier_rate | pass_all |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in current_040.iterrows():
        lines.append(
            f"| {row['objective']} | {row['mean_quality']:.4f} | {row['quality_gap_to_target_oracle']:.4f} | "
            f"{row['normalized_cost_vs_all_strong']:.4f} | {row['utility_ratio_to_target_oracle']:.4f} | "
            f"{row['frontier_call_rate']:.4f} | {bool(row['pass_all_phase3_gates'])} |"
        )
    lines.extend(
        [
            "",
            "Current split at 0.60 frontier cap:",
            "",
            "| objective | quality | gap | cost | utility_ratio | frontier_rate | pass_all |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in current_060.iterrows():
        lines.append(
            f"| {row['objective']} | {row['mean_quality']:.4f} | {row['quality_gap_to_target_oracle']:.4f} | "
            f"{row['normalized_cost_vs_all_strong']:.4f} | {row['utility_ratio_to_target_oracle']:.4f} | "
            f"{row['frontier_call_rate']:.4f} | {bool(row['pass_all_phase3_gates'])} |"
        )
    lines.extend(
        [
            "",
            "Fresh split summary at 0.40 frontier cap:",
            "",
            "| objective | pass_all_rate | pass_quality_rate | mean_quality | mean_gap | mean_cost | mean_utility_ratio |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in fresh_040.iterrows():
        lines.append(
            f"| {row['objective']} | {row['pass_all_rate']:.4f} | {row['pass_quality_rate']:.4f} | "
            f"{row['mean_quality']:.4f} | {row['mean_quality_gap']:.4f} | "
            f"{row['mean_normalized_cost']:.4f} | {row['mean_utility_ratio']:.4f} |"
        )
    original_needed = needed[needed["split_context"].eq("original")]
    lines.extend(["", "Minimum cap needed on the current split:", ""])
    for _, row in original_needed.iterrows():
        lines.append(
            f"- `{row['objective']}` objective: all gates at cap `{row['min_cap_for_all_gates']}`, "
            f"quality gate at cap `{row['min_cap_for_quality_gate']}`."
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "The expanded deterministic tools make the current held-out split feasible at the 0.40 frontier cap.",
            "Across fresh stratified splits, the 0.40 cap is improved but still not guaranteed. This means the",
            "frontier-rate claim is plausible on the current split, but split-stable deployment still needs",
            "stronger local/tool coverage or a better cheap verifier.",
        ]
    )
    output_dir.joinpath("FRONTIER_RATE_FEASIBILITY_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rates = parse_rates(args.frontier_rate_caps)
    table, tool, conf = load_table()
    rows = evaluate_split(table, tool=tool, rates=rates, split_context="original", fresh_seed=None, args=args)
    for offset in range(int(args.fresh_seeds)):
        seed = int(args.first_fresh_seed + offset)
        fresh = conf.stratified_resplit(table, seed)
        rows.extend(evaluate_split(fresh, tool=tool, rates=rates, split_context="fresh", fresh_seed=seed, args=args))
    results = pd.DataFrame(rows)
    summary = summarize(results)
    needed = min_needed(results)
    results.to_csv(output_dir / "table_frontier_rate_feasibility.csv", index=False)
    summary.to_csv(output_dir / "table_frontier_rate_feasibility_summary.csv", index=False)
    needed.to_csv(output_dir / "table_min_frontier_rate_needed.csv", index=False)
    write_memo(output_dir, results, summary, needed)
    print(f"Wrote frontier-rate feasibility artifacts to {output_dir}")
    print(results[results["split_context"].eq("original")].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
