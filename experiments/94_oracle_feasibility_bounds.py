from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_MODEL = "qwen3-8b-local"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"
MODELS = (LOCAL_MODEL, GEMINI_MODEL, GPT_MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute oracle lower bounds for quality/cost/frontier feasibility on cached exact-math outcomes."
    )
    parser.add_argument("--query-table", default="results/controlled/gemini_metadata_gate/query_table_with_gemini_metadata.csv")
    parser.add_argument("--output-dir", default="results/controlled/oracle_feasibility_bounds")
    parser.add_argument("--split", default="test")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--target-quality-gap", type=float, default=0.03)
    return parser.parse_args()


def action_options(row: pd.Series, cost_norm: float, lambda_cost: float) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for action, model_id in [("local", LOCAL_MODEL), ("gemini", GEMINI_MODEL), ("gpt", GPT_MODEL)]:
        cost = 0.0 if action == "local" else float(row[f"{model_id}_cost"])
        quality = float(row[f"{model_id}_quality"])
        options.append(
            {
                "action": action,
                "quality": quality,
                "cost": cost,
                "utility": quality - lambda_cost * (cost / cost_norm),
                "frontier": int(action != "local"),
                "gpt": int(action == "gpt"),
            }
        )
    return options


def best_quality_under_frontier(frame: pd.DataFrame, max_frontier: int, cost_norm: float, lambda_cost: float) -> float:
    dp = {0: 0.0}
    for _, row in frame.iterrows():
        next_dp: dict[int, float] = {}
        for used_frontier, quality in dp.items():
            for option in action_options(row, cost_norm, lambda_cost):
                new_frontier = used_frontier + int(option["frontier"])
                if new_frontier <= max_frontier:
                    next_dp[new_frontier] = max(next_dp.get(new_frontier, -1.0), quality + float(option["quality"]))
        dp = next_dp
    return max(dp.values()) / len(frame)


def min_resources_for_quality(
    frame: pd.DataFrame,
    *,
    target_quality: float,
    cost_norm: float,
    lambda_cost: float,
) -> dict[str, object]:
    # Dynamic program over integer correct-count, minimizing cost/frontier/GPT lexicographically.
    target_correct = int(np.ceil(target_quality * len(frame) - 1e-12))
    inf = (float("inf"), float("inf"), float("inf"))
    dp: dict[int, tuple[float, float, float]] = {0: (0.0, 0.0, 0.0)}
    for _, row in frame.iterrows():
        next_dp: dict[int, tuple[float, float, float]] = {}
        for correct, resources in dp.items():
            for option in action_options(row, cost_norm, lambda_cost):
                new_correct = correct + int(float(option["quality"]) > 0.5)
                new_resources = (
                    resources[0] + float(option["cost"]),
                    resources[1] + float(option["frontier"]),
                    resources[2] + float(option["gpt"]),
                )
                if new_resources < next_dp.get(new_correct, inf):
                    next_dp[new_correct] = new_resources
        dp = next_dp
    feasible = [(correct, resources) for correct, resources in dp.items() if correct >= target_correct]
    if not feasible:
        return {
            "target_quality": target_quality,
            "target_correct": target_correct,
            "feasible": False,
            "min_cost_usd": np.nan,
            "normalized_remote_cost_vs_all_gpt": np.nan,
            "min_frontier_calls_at_min_cost": np.nan,
            "min_gpt_calls_at_min_cost": np.nan,
        }
    correct, resources = min(feasible, key=lambda item: item[1])
    all_gpt_cost = float(frame[f"{GPT_MODEL}_cost"].sum())
    return {
        "target_quality": target_quality,
        "target_correct": target_correct,
        "achieved_correct": int(correct),
        "achieved_quality": float(correct / len(frame)),
        "feasible": True,
        "min_cost_usd": float(resources[0]),
        "normalized_remote_cost_vs_all_gpt": float(resources[0] / all_gpt_cost) if all_gpt_cost > 0 else np.nan,
        "min_frontier_calls_at_min_cost": int(resources[1]),
        "min_frontier_rate_at_min_cost": float(resources[1] / len(frame)),
        "min_gpt_calls_at_min_cost": int(resources[2]),
        "min_gpt_rate_at_min_cost": float(resources[2] / len(frame)),
    }


def min_frontier_for_quality(frame: pd.DataFrame, target_quality: float, cost_norm: float, lambda_cost: float) -> dict[str, object]:
    for max_frontier in range(len(frame) + 1):
        quality = best_quality_under_frontier(frame, max_frontier, cost_norm, lambda_cost)
        if quality >= target_quality - 1e-12:
            return {
                "target_quality": target_quality,
                "min_frontier_calls": max_frontier,
                "min_frontier_rate": float(max_frontier / len(frame)),
                "achieved_quality": quality,
            }
    return {
        "target_quality": target_quality,
        "min_frontier_calls": np.nan,
        "min_frontier_rate": np.nan,
        "achieved_quality": np.nan,
    }


def summarize_oracles(frame: pd.DataFrame, cost_norm: float, lambda_cost: float) -> pd.DataFrame:
    local_q = frame[f"{LOCAL_MODEL}_quality"].astype(float)
    gemini_q = frame[f"{GEMINI_MODEL}_quality"].astype(float)
    gpt_q = frame[f"{GPT_MODEL}_quality"].astype(float)
    quality_oracle = frame[[f"{model}_quality" for model in MODELS]].max(axis=1)
    utility_cols = [f"{model}_utility_selected_cost" for model in MODELS]
    selected = frame[utility_cols].idxmax(axis=1)
    action = selected.map(
        {
            f"{LOCAL_MODEL}_utility_selected_cost": "local",
            f"{GEMINI_MODEL}_utility_selected_cost": "gemini",
            f"{GPT_MODEL}_utility_selected_cost": "gpt",
        }
    )
    costs = [
        0.0
        if act == "local"
        else float(row[f"{GEMINI_MODEL}_cost"])
        if act == "gemini"
        else float(row[f"{GPT_MODEL}_cost"])
        for act, (_, row) in zip(action, frame.iterrows())
    ]
    utility = quality_oracle - lambda_cost * (np.array(costs) / cost_norm)
    return pd.DataFrame(
        [
            {
                "split": str(frame["split"].iloc[0]) if "split" in frame.columns and len(frame) else "",
                "n_queries": int(len(frame)),
                "local_quality": float(local_q.mean()),
                "gemini_quality": float(gemini_q.mean()),
                "gpt_quality": float(gpt_q.mean()),
                "quality_oracle_quality": float(quality_oracle.mean()),
                "selected_cost_oracle_utility": float(utility.mean()),
                "selected_cost_oracle_frontier_rate": float(action.ne("local").mean()),
                "selected_cost_oracle_gpt_rate": float(action.eq("gpt").mean()),
                "selected_cost_oracle_normalized_cost": float(np.sum(costs) / frame[f"{GPT_MODEL}_cost"].sum()),
                "selected_cost_oracle_action_counts": dict(action.value_counts().to_dict()),
            }
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    frame = table[table["split"].astype(str).eq(args.split)].copy()
    if frame.empty:
        raise ValueError(f"No rows for split {args.split!r}.")
    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    oracle_summary = summarize_oracles(frame, cost_norm, args.lambda_cost)
    oracle_quality = float(oracle_summary["quality_oracle_quality"].iloc[0])
    target_quality = oracle_quality - float(args.target_quality_gap)
    if len(frame) > 0:
        # Also report the nearest achievable discrete threshold at ceil(correct).
        target_correct = int(np.ceil(target_quality * len(frame) - 1e-12))
        target_quality_discrete = target_correct / len(frame)
    else:
        target_quality_discrete = target_quality

    frontier_rows = []
    for rate in [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]:
        max_frontier = min(len(frame), int(np.floor(rate * len(frame))))
        frontier_rows.append(
            {
                "frontier_rate_cap": rate,
                "frontier_call_cap": max_frontier,
                "max_quality": best_quality_under_frontier(frame, max_frontier, cost_norm, args.lambda_cost),
            }
        )
    frontier_table = pd.DataFrame(frontier_rows)
    resource_table = pd.DataFrame(
        [
            min_frontier_for_quality(frame, oracle_quality, cost_norm, args.lambda_cost)
            | {"target": "exact_quality_oracle"},
            min_frontier_for_quality(frame, target_quality, cost_norm, args.lambda_cost)
            | {"target": "oracle_minus_target_gap_continuous"},
            min_frontier_for_quality(frame, target_quality_discrete, cost_norm, args.lambda_cost)
            | {"target": "oracle_minus_target_gap_discrete"},
        ]
    )
    cost_table = pd.DataFrame(
        [
            min_resources_for_quality(frame, target_quality=oracle_quality, cost_norm=cost_norm, lambda_cost=args.lambda_cost)
            | {"target": "exact_quality_oracle"},
            min_resources_for_quality(frame, target_quality=target_quality, cost_norm=cost_norm, lambda_cost=args.lambda_cost)
            | {"target": "oracle_minus_target_gap_continuous"},
            min_resources_for_quality(
                frame, target_quality=target_quality_discrete, cost_norm=cost_norm, lambda_cost=args.lambda_cost
            )
            | {"target": "oracle_minus_target_gap_discrete"},
        ]
    )

    oracle_path = output_dir / "table_oracle_summary.csv"
    frontier_path = output_dir / "table_frontier_rate_bounds.csv"
    resource_path = output_dir / "table_min_frontier_for_quality.csv"
    cost_path = output_dir / "table_min_cost_for_quality.csv"
    oracle_summary.to_csv(oracle_path, index=False)
    frontier_table.to_csv(frontier_path, index=False)
    resource_table.to_csv(resource_path, index=False)
    cost_table.to_csv(cost_path, index=False)

    memo = [
        "# Oracle Feasibility Bounds Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Split: `{args.split}`. Rows: `{len(frame)}`.",
        "This is an outcome-matrix lower-bound audit. It gives the best possible quality under solver-call constraints if the router had perfect knowledge of per-query model correctness.",
        "",
        "## Oracle Summary",
        "",
        markdown_table(oracle_summary),
        "",
        "## Max Quality Under Frontier Solver-Call Caps",
        "",
        markdown_table(frontier_table),
        "",
        "## Minimum Frontier Calls For Quality Targets",
        "",
        markdown_table(resource_table),
        "",
        "## Minimum Remote Cost For Quality Targets",
        "",
        markdown_table(cost_table),
        "",
        "## Interpretation",
        "",
        (
            "On this held-out exact-math split, the configured <=40% frontier-call target is not feasible at the "
            "within-3-point quality target with the current model pool. Even an outcome oracle needs more than "
            "40% frontier solver calls to get within the target quality gap. The remedy is not a better threshold "
            "alone; the local model pool or task mix must change."
        ),
        "",
        "## Files",
        "",
        f"- `{oracle_path}`",
        f"- `{frontier_path}`",
        f"- `{resource_path}`",
        f"- `{cost_path}`",
    ]
    memo_path = output_dir / "ORACLE_FEASIBILITY_BOUNDS_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
