from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked fresh-split confirmation for the tool-augmented exact-math policy."
    )
    parser.add_argument("--output-dir", default="results/controlled/tool_augmented_fresh_split_confirmation")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--utility-ratio-target", type=float, default=0.95)
    parser.add_argument("--frontier-rate-target", type=float, default=0.40)
    return parser.parse_args()


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def stratified_resplit(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = table.groupby(["dataset", "split"], sort=False).size().unstack(fill_value=0)
    out = table.copy()
    out["original_split"] = out["split"]
    new_split = pd.Series(index=out.index, dtype=object)
    for dataset, frame in out.groupby("dataset", sort=False):
        idx = frame.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n_train = int(counts.loc[dataset].get("train", 0))
        n_val = int(counts.loc[dataset].get("val", 0))
        n_test = int(counts.loc[dataset].get("test", 0))
        expected = n_train + n_val + n_test
        if expected != len(idx):
            raise ValueError(f"Split counts for {dataset} sum to {expected}, got {len(idx)} rows.")
        new_split.loc[idx[:n_train]] = "train"
        new_split.loc[idx[n_train : n_train + n_val]] = "val"
        new_split.loc[idx[n_train + n_val :]] = "test"
    out["split"] = new_split.astype(str)
    return out


def recompute_strong_inclusive_oracle(table: pd.DataFrame, tool_module, lambda_cost: float) -> pd.DataFrame:
    frames = []
    oracle_actions = list(tool_module.LOCAL_MODELS) + [tool_module.GEMINI, tool_module.BASE_GPT, tool_module.STRONG_GPT]
    for _, frame in table.groupby("split", sort=False):
        frame = frame.copy()
        strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
        quality_cols = []
        utility_cols = []
        for action in oracle_actions:
            qualities = []
            costs = []
            for _, row in frame.iterrows():
                quality, cost = tool_module.row_quality_cost(row, action)
                qualities.append(quality)
                costs.append(cost)
            q = np.asarray(qualities, dtype=float)
            c = np.asarray(costs, dtype=float)
            quality_cols.append(q)
            utility_cols.append(q - float(lambda_cost) * c * len(frame) / strong_norm)
        qmat = np.vstack(quality_cols).T
        umat = np.vstack(utility_cols).T
        best = umat.argmax(axis=1)
        frame["strong_inclusive_cost_oracle_model"] = [oracle_actions[i] for i in best]
        frame["strong_inclusive_cost_oracle_quality"] = qmat[np.arange(len(frame)), best]
        frame["strong_inclusive_cost_oracle_utility"] = umat[np.arange(len(frame)), best]
        frames.append(frame)
    return pd.concat(frames).sort_index()


def row_passes(row: pd.Series, args: argparse.Namespace) -> bool:
    return bool(
        float(row["quality_gap_to_strong_inclusive_oracle"]) <= float(args.quality_gap_target)
        and float(row["normalized_cost_vs_all_strong"]) <= float(args.cost_target)
        and float(row["utility_ratio_to_strong_inclusive_oracle"]) >= float(args.utility_ratio_target)
        and float(row.get("frontier_call_rate", 1.0)) <= float(args.frontier_rate_target)
    )


def summarize(selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for rule, frame in selected[selected["split"].eq("test")].groupby("selection_rule", sort=False):
        passed = frame.apply(lambda row: row_passes(row, args), axis=1)
        rows.append(
            {
                "selection_rule": rule,
                "n_seeds": int(len(frame)),
                "pass_rate": float(passed.mean()) if len(frame) else 0.0,
                "mean_quality": float(frame["mean_quality"].mean()),
                "min_quality": float(frame["mean_quality"].min()),
                "mean_quality_gap": float(frame["quality_gap_to_strong_inclusive_oracle"].mean()),
                "max_quality_gap": float(frame["quality_gap_to_strong_inclusive_oracle"].max()),
                "mean_normalized_cost": float(frame["normalized_cost_vs_all_strong"].mean()),
                "max_normalized_cost": float(frame["normalized_cost_vs_all_strong"].max()),
                "mean_utility_ratio": float(frame["utility_ratio_to_strong_inclusive_oracle"].mean()),
                "min_utility_ratio": float(frame["utility_ratio_to_strong_inclusive_oracle"].min()),
                "mean_frontier_call_rate": float(frame["frontier_call_rate"].mean()),
                "max_frontier_call_rate": float(frame["frontier_call_rate"].max()),
            }
        )
    return pd.DataFrame(rows)


def write_memo(output_dir: Path, summary: pd.DataFrame, selected: pd.DataFrame, route_cost: float) -> None:
    lines = [
        "# Locked Fresh-Split Confirmation",
        "",
        "This confirmation uses cached model outputs and cached AIME GPT-5.5 route labels only.",
        "No new solver outputs are generated by this script.",
        "",
        f"Cached AIME route-label cost available: `${route_cost:.4f}`.",
        f"Fresh split seeds evaluated: `{selected['fresh_seed'].nunique()}`.",
        "",
        "Summary by selected test row:",
        "",
        "| " + " | ".join(summary.columns) + " |",
        "| " + " | ".join(["---"] * len(summary.columns)) + " |",
    ]
    for _, row in summary.iterrows():
        values = []
        for col in summary.columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    qc = selected[
        selected["selection_rule"].eq("validation_feasible_quality_conservative_test")
        & selected["split"].eq("test")
    ].copy()
    if len(qc):
        passes = int(qc.apply(lambda row: row_passes(row, SimpleNamespace(
            quality_gap_target=0.03,
            cost_target=0.35,
            utility_ratio_target=0.95,
            frontier_rate_target=0.40,
        )), axis=1).sum())
        lines.extend(
            [
                "",
                "Locked quality-conservative interpretation:",
                "",
                f"- Passed Phase 3 gates on `{passes}/{len(qc)}` fresh stratified splits.",
                f"- Mean quality `{qc['mean_quality'].mean():.4f}`; worst quality `{qc['mean_quality'].min():.4f}`.",
                f"- Mean gap `{qc['quality_gap_to_strong_inclusive_oracle'].mean():.4f}`; worst gap `{qc['quality_gap_to_strong_inclusive_oracle'].max():.4f}`.",
                f"- Mean normalized cost `{qc['normalized_cost_vs_all_strong'].mean():.4f}`.",
                f"- Mean utility ratio `{qc['utility_ratio_to_strong_inclusive_oracle'].mean():.4f}`.",
                f"- Mean frontier-call rate `{qc['frontier_call_rate'].mean():.4f}`.",
            ]
        )
    output_dir.joinpath("LOCKED_FRESH_SPLIT_CONFIRMATION_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tool = load_module("experiments/112_tool_augmented_aime_policy.py", "tool_policy")
    router = tool.load_router_module()
    full_table = tool.add_tool_outputs(tool.load_table(router))
    routes = pd.read_csv("results/controlled/aime_gpt_judge_with_gemini_strong/table_aime_gpt_judge_routes.csv")
    route_counts = routes.merge(full_table[["query_id", "dataset"]], on="query_id", how="left")
    expected_aime = int(full_table["dataset"].eq("aime").sum())
    if int(route_counts["dataset"].eq("aime").sum()) != expected_aime:
        raise RuntimeError("Fresh split confirmation requires cached route labels for every AIME row.")

    all_grid = []
    all_selected = []
    eval_args = SimpleNamespace(
        lambda_cost=args.lambda_cost,
        quality_gap_target=args.quality_gap_target,
        cost_target=args.cost_target,
        utility_ratio_target=args.utility_ratio_target,
        frontier_rate_target=args.frontier_rate_target,
    )
    for offset in range(int(args.seeds)):
        seed = int(args.first_seed + offset)
        table = stratified_resplit(full_table, seed)
        table = recompute_strong_inclusive_oracle(table, tool, args.lambda_cost)
        baseline = tool.baseline_actions(router, table)
        grid, selected = tool.evaluate(table, baseline, routes, eval_args)
        grid.insert(0, "fresh_seed", seed)
        selected.insert(0, "fresh_seed", seed)
        all_grid.append(grid)
        all_selected.append(selected)

    grid_df = pd.concat(all_grid, ignore_index=True)
    selected_df = pd.concat(all_selected, ignore_index=True)
    summary_df = summarize(selected_df, args)
    grid_df.to_csv(output_dir / "table_locked_fresh_split_grid.csv", index=False)
    selected_df.to_csv(output_dir / "table_locked_fresh_split_selected.csv", index=False)
    summary_df.to_csv(output_dir / "table_locked_fresh_split_summary.csv", index=False)
    write_memo(output_dir, summary_df, selected_df, float(routes["route_cost"].sum()))
    print(f"Wrote locked fresh-split confirmation to {output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
