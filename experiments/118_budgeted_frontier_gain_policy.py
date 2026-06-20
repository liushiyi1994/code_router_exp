from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TOOL = "deterministic_math_tool"
LOCAL_ACTIONS = [TOOL, "qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
FRONTIER_ACTIONS = ["gemini-3.5-flash", "gpt-5.5", "gemini-3.5-flash-strong-solve", "strong-gpt-5.5"]
ALL_ACTIONS = LOCAL_ACTIONS + FRONTIER_ACTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local-first budgeted frontier-gain policy.")
    parser.add_argument("--output-dir", default="results/controlled/budgeted_frontier_gain_policy")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--utility-ratio-target", type=float, default=0.95)
    parser.add_argument("--frontier-rate-target", type=float, default=0.40)
    parser.add_argument("--fresh-seeds", type=int, default=10)
    parser.add_argument("--first-fresh-seed", type=int, default=1000)
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
    table = add_policy_features(table)
    return table, tool, conf


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def add_policy_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    text = table["query_text"].fillna("").astype(str)
    text = text + " dataset=" + table["dataset"].fillna("").astype(str)
    for action in ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]:
        text = text + f" {action}=" + table[f"{action}_answer_norm"].fillna("").astype(str)
    table["frontier_gain_feature_text"] = text
    table["local_vote_bin"] = table["local_ensemble_votes"].map(vote_bin)
    table["query_len_bin"] = table["query_len"].map(len_bin)
    table["number_count_bin"] = table["number_count"].map(num_bin)
    table["tool_available_str"] = table["tool_available"].astype(str)
    return table


def vote_bin(value: object) -> str:
    try:
        votes = int(float(value))
    except Exception:
        votes = 0
    return "3plus" if votes >= 3 else str(max(votes, 0))


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


def cat_cols() -> list[str]:
    return [
        "dataset",
        "tool_available_str",
        "qwen8_4b_agree",
        "qwen8_06b_agree",
        "small_pair_agree",
        "all_three_agree",
        "local_vote_bin",
        "query_len_bin",
        "number_count_bin",
    ]


def num_cols() -> list[str]:
    return ["query_len", "number_count", "latex_count", "frac_count", "sqrt_count", "local_ensemble_votes"]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in cat_cols():
        frame[column] = frame[column].fillna(False).astype(str)
    for column in num_cols():
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=18000), "frontier_gain_feature_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols()),
            ("num", StandardScaler(with_mean=False), num_cols()),
        ]
    )


def learners() -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesRegressor(n_estimators=80, random_state=31, n_jobs=-1),
        "ridge": Ridge(alpha=1.0),
    }


def row_quality_cost(row: pd.Series, action: str, tool) -> tuple[float, float]:
    if action == TOOL:
        return float(row["tool_quality"]), 0.0
    quality, cost = tool.row_quality_cost(row, action)
    return float(quality), float(cost)


def action_quality_costs(frame: pd.DataFrame, actions: pd.Series, tool) -> tuple[np.ndarray, np.ndarray]:
    action_values = actions.astype(str).reindex(frame.index).to_numpy()
    qualities = np.zeros(len(frame), dtype=float)
    costs = np.zeros(len(frame), dtype=float)
    for action in np.unique(action_values):
        mask = action_values == action
        q, c = constant_action_quality_costs(frame, str(action))
        qualities[mask] = q[mask]
        costs[mask] = c[mask]
    return qualities, costs


def constant_action_quality_costs(frame: pd.DataFrame, action: str) -> tuple[np.ndarray, np.ndarray]:
    if action == TOOL:
        return frame["tool_quality"].astype(float).to_numpy(), np.zeros(len(frame), dtype=float)
    if action == "gemini-3.5-flash-strong-solve":
        return frame["gemini_strong_quality"].astype(float).to_numpy(), frame["gemini_strong_cost"].astype(float).to_numpy()
    if action == "strong-gpt-5.5":
        return frame["strong_quality"].astype(float).to_numpy(), frame["strong_cost"].astype(float).to_numpy()
    return frame[f"{action}_quality"].astype(float).to_numpy(), (
        frame[f"{action}_cost"].astype(float).to_numpy() if f"{action}_cost" in frame.columns else np.zeros(len(frame), dtype=float)
    )


def local_baseline_action(row: pd.Series, baseline: str) -> str:
    if bool(row.get("tool_available", False)):
        return TOOL
    if baseline == "local_min_cost":
        return "qwen3-0.6b-probe"
    if baseline == "local_qwen8":
        return "qwen3-8b-local"
    if baseline == "local_agreement":
        if as_bool(row.get("all_three_agree", False)):
            return "qwen3-8b-local"
        if float(row.get("local_ensemble_votes", 0) or 0) >= 2:
            return str(row.get("local_ensemble_source") or "qwen3-8b-local")
        return "qwen3-14b-awq-local"
    raise ValueError(baseline)


def baseline_actions(frame: pd.DataFrame, baseline: str) -> pd.Series:
    return pd.Series([local_baseline_action(row, baseline) for _, row in frame.iterrows()], index=frame.index, dtype=object)


def split_scale(frame: pd.DataFrame) -> float:
    return max(float(frame["strong_cost"].sum()) / max(len(frame), 1), 1e-12)


def action_row_utility(frame: pd.DataFrame, action: str, tool, lambda_cost: float) -> np.ndarray:
    scale = split_scale(frame)
    quality, cost = constant_action_quality_costs(frame, action)
    return quality - float(lambda_cost) * cost / scale


def frontier_targets(frame: pd.DataFrame, tool, lambda_cost: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    utilities = np.vstack([action_row_utility(frame, action, tool, lambda_cost) for action in FRONTIER_ACTIONS]).T
    best_idx = utilities.argmax(axis=1)
    best_utility = utilities[np.arange(len(frame)), best_idx]
    best_actions = [FRONTIER_ACTIONS[idx] for idx in best_idx]
    return best_utility, best_idx, best_actions


def oracle_target(frame: pd.DataFrame, tool, lambda_cost: float) -> tuple[np.ndarray, list[str]]:
    utilities = np.vstack([action_row_utility(frame, action, tool, lambda_cost) for action in ALL_ACTIONS]).T
    best_idx = utilities.argmax(axis=1)
    return utilities[np.arange(len(frame)), best_idx], [ALL_ACTIONS[idx] for idx in best_idx]


def apply_budget(
    frame: pd.DataFrame,
    *,
    baseline: str,
    gain_scores: np.ndarray,
    action_scores: np.ndarray,
    min_gain: float,
    frontier_cap: float,
    normalized_cost_cap: float,
    tool,
) -> pd.Series:
    actions = baseline_actions(frame, baseline)
    _, base_costs = action_quality_costs(frame, actions, tool)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    current_cost = float(base_costs.sum())
    max_calls = int(np.floor(float(frontier_cap) * len(frame) + 1e-12))
    calls = 0
    for pos in np.argsort(-gain_scores):
        if calls >= max_calls:
            break
        if float(gain_scores[pos]) < float(min_gain):
            continue
        idx = frame.index[pos]
        if str(actions.loc[idx]) == TOOL:
            continue
        frontier_action = FRONTIER_ACTIONS[int(np.argmax(action_scores[pos]))]
        _, old_cost = row_quality_cost(frame.iloc[pos], str(actions.loc[idx]), tool)
        _, new_cost = row_quality_cost(frame.iloc[pos], frontier_action, tool)
        proposed = current_cost - old_cost + new_cost
        if proposed / strong_norm <= float(normalized_cost_cap):
            actions.loc[idx] = frontier_action
            current_cost = proposed
            calls += 1
    return actions


def evaluate_actions(frame: pd.DataFrame, actions: pd.Series, *, method: str, lambda_cost: float, tool) -> dict[str, object]:
    qualities, costs = action_quality_costs(frame, actions, tool)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    mean_quality = float(qualities.mean())
    normalized_cost = float(costs.sum() / strong_norm)
    mean_utility = mean_quality - float(lambda_cost) * normalized_cost
    target_utilities, target_actions = oracle_target(frame, tool, lambda_cost)
    target_quality, target_cost = action_quality_costs(frame, pd.Series(target_actions, index=frame.index), tool)
    target_mean_quality = float(np.mean(target_quality))
    target_normalized_cost = float(np.sum(target_cost) / strong_norm)
    target_mean_utility = target_mean_quality - float(lambda_cost) * target_normalized_cost
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_target_oracle": float(target_mean_quality - mean_quality),
        "normalized_cost_vs_all_strong": normalized_cost,
        "utility_ratio_to_target_oracle": float(mean_utility / target_mean_utility),
        "frontier_call_rate": float(np.mean([str(action) in FRONTIER_ACTIONS for action in actions])),
        "action_counts": json.dumps(actions.astype(str).value_counts().to_dict(), sort_keys=True),
    }


def run_grid(table: pd.DataFrame, args: argparse.Namespace, tool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = prepare_features(table[table["split"].eq("train")].copy())
    eval_frames = {
        split: prepare_features(frame.copy())
        for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False)
    }
    rows = []
    baselines = ["local_min_cost", "local_qwen8", "local_agreement"]
    thresholds = [-0.03, 0.0, 0.03, 0.06]
    frontier_caps = [0.30, 0.35, 0.40]
    cost_caps = [0.20, 0.25, 0.30, 0.35]
    for baseline in baselines:
        base_actions = baseline_actions(train, baseline)
        base_quality, base_cost = action_quality_costs(train, base_actions, tool)
        base_utility = base_quality - float(args.lambda_cost) * base_cost / split_scale(train)
        best_frontier_utility, _, _ = frontier_targets(train, tool, args.lambda_cost)
        y_gain = best_frontier_utility - base_utility
        action_quality_targets = {
            action: np.asarray([row_quality_cost(row, action, tool)[0] for _, row in train.iterrows()], dtype=float)
            for action in FRONTIER_ACTIONS
        }
        for learner_name, learner in learners().items():
            gain_model = make_pipeline(preprocessor(), learner)
            gain_model.fit(train, y_gain)
            action_models = {}
            for action, target in action_quality_targets.items():
                model = make_pipeline(preprocessor(), learners()[learner_name])
                model.fit(train, target)
                action_models[action] = model
            pred_gain = {split: gain_model.predict(frame) for split, frame in eval_frames.items()}
            pred_actions = {
                split: np.vstack([action_models[action].predict(frame) for action in FRONTIER_ACTIONS]).T
                for split, frame in eval_frames.items()
            }
            for min_gain in thresholds:
                for frontier_cap in frontier_caps:
                    for cost_cap in cost_caps:
                        method = (
                            f"budgeted_frontier_{baseline}_{learner_name}_gain{min_gain:g}"
                            f"_fr{frontier_cap:g}_cost{cost_cap:g}"
                        )
                        for split, frame in eval_frames.items():
                            actions = apply_budget(
                                frame,
                                baseline=baseline,
                                gain_scores=pred_gain[split],
                                action_scores=pred_actions[split],
                                min_gain=min_gain,
                                frontier_cap=frontier_cap,
                                normalized_cost_cap=cost_cap,
                                tool=tool,
                            )
                            rows.append(evaluate_actions(frame, actions, method=method, lambda_cost=args.lambda_cost, tool=tool))
    grid = pd.DataFrame(rows)
    selected = select_rows(grid, args)
    return grid, selected


def select_rows(grid: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    val = grid[grid["split"].eq("val")].copy()
    feasible = val[
        val["quality_gap_to_target_oracle"].le(args.quality_gap_target)
        & val["normalized_cost_vs_all_strong"].le(args.cost_target)
        & val["utility_ratio_to_target_oracle"].ge(args.utility_ratio_target)
        & val["frontier_call_rate"].le(args.frontier_rate_target)
    ]
    rows = []
    if len(feasible):
        picks = [
            (
                "validation_feasible_min_frontier",
                feasible.sort_values(["frontier_call_rate", "quality_gap_to_target_oracle", "normalized_cost_vs_all_strong"]).head(1),
            ),
            (
                "validation_feasible_quality_conservative",
                feasible.sort_values(
                    ["mean_quality", "quality_gap_to_target_oracle", "utility_ratio_to_target_oracle", "frontier_call_rate"],
                    ascending=[False, True, False, True],
                ).head(1),
            ),
        ]
    else:
        under_limits = val[
            val["normalized_cost_vs_all_strong"].le(args.cost_target)
            & val["frontier_call_rate"].le(args.frontier_rate_target)
        ]
        picks = [
            (
                "no_validation_feasible_best_gap_under_limits",
                under_limits.sort_values(
                    ["quality_gap_to_target_oracle", "utility_ratio_to_target_oracle"],
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
    diag = grid[
        grid["split"].eq("test")
        & grid["normalized_cost_vs_all_strong"].le(args.cost_target)
        & grid["frontier_call_rate"].le(args.frontier_rate_target)
    ].sort_values(["quality_gap_to_target_oracle", "utility_ratio_to_target_oracle"], ascending=[True, False]).head(1)
    if len(diag):
        rows.append(diag.assign(selection_rule="best_heldout_diagnostic_under_limits"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pass_gate(row: pd.Series, args: argparse.Namespace) -> bool:
    return bool(
        row["quality_gap_to_target_oracle"] <= args.quality_gap_target
        and row["normalized_cost_vs_all_strong"] <= args.cost_target
        and row["utility_ratio_to_target_oracle"] >= args.utility_ratio_target
        and row["frontier_call_rate"] <= args.frontier_rate_target
    )


def summarize(selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
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
                "mean_quality_gap": float(frame["quality_gap_to_target_oracle"].mean()),
                "max_quality_gap": float(frame["quality_gap_to_target_oracle"].max()),
                "mean_normalized_cost": float(frame["normalized_cost_vs_all_strong"].mean()),
                "mean_utility_ratio": float(frame["utility_ratio_to_target_oracle"].mean()),
                "mean_frontier_call_rate": float(frame["frontier_call_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_memo(output_dir: Path, selected: pd.DataFrame, fresh_summary: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "method",
        "split",
        "mean_quality",
        "quality_gap_to_target_oracle",
        "normalized_cost_vs_all_strong",
        "utility_ratio_to_target_oracle",
        "frontier_call_rate",
        "action_counts",
    ]
    lines = [
        "# Budgeted Frontier-Gain Policy",
        "",
        "This cached-only policy starts from local/tool answers and spends a frontier-call budget on rows with high predicted frontier gain.",
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
        values = [f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]) for col in fresh_summary.columns]
        lines.append("| " + " | ".join(values) + " |")
    output_dir.joinpath("BUDGETED_FRONTIER_GAIN_POLICY_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table, tool, conf = load_table()
    grid, selected = run_grid(table, args, tool)
    grid.to_csv(output_dir / "table_budgeted_frontier_gain_policy.csv", index=False)
    selected.to_csv(output_dir / "table_budgeted_frontier_gain_policy_selected.csv", index=False)
    fresh_grids = []
    fresh_selected = []
    for offset in range(args.fresh_seeds):
        seed = int(args.first_fresh_seed + offset)
        fresh_table = add_policy_features(conf.stratified_resplit(table, seed))
        fresh_grid, fresh_sel = run_grid(fresh_table, args, tool)
        fresh_grid.insert(0, "fresh_seed", seed)
        fresh_sel.insert(0, "fresh_seed", seed)
        fresh_grids.append(fresh_grid)
        fresh_selected.append(fresh_sel)
    fresh_grid_df = pd.concat(fresh_grids, ignore_index=True)
    fresh_selected_df = pd.concat(fresh_selected, ignore_index=True)
    fresh_summary = summarize(fresh_selected_df, args)
    fresh_grid_df.to_csv(output_dir / "table_budgeted_frontier_gain_policy_fresh_grid.csv", index=False)
    fresh_selected_df.to_csv(output_dir / "table_budgeted_frontier_gain_policy_fresh_selected.csv", index=False)
    fresh_summary.to_csv(output_dir / "table_budgeted_frontier_gain_policy_fresh_summary.csv", index=False)
    write_memo(output_dir, selected, fresh_summary)
    print(f"Wrote budgeted frontier-gain policy results to {output_dir}")
    print(selected.to_string(index=False))
    print(fresh_summary.to_string(index=False))


if __name__ == "__main__":
    main()
