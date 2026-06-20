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
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
GEMINI_STRONG = "gemini-3.5-flash-strong-solve"
STRONG_GPT = "strong-gpt-5.5"
QWEN14 = "qwen3-14b-awq-local"
QWEN8 = "qwen3-8b-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a budgeted strong-rescue layer over agreement baselines.")
    parser.add_argument("--output-dir", default="results/controlled/budgeted_strong_rescue_policy")
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


def load_table() -> tuple[pd.DataFrame, object, object]:
    tool = load_module("experiments/112_tool_augmented_aime_policy.py", "tool_policy")
    conf = load_module("experiments/113_locked_fresh_split_confirmation.py", "fresh_confirm")
    router = tool.load_router_module()
    table = tool.add_tool_outputs(tool.load_table(router))
    table = add_policy_features(table)
    return table, tool, conf


def add_policy_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    text_parts = [
        table["query_text"].fillna("").astype(str),
        " dataset=" + table["dataset"].fillna("").astype(str),
        " gemini=" + table[f"{GEMINI}_answer_norm"].fillna("").astype(str),
        " base=" + table[f"{BASE_GPT}_answer_norm"].fillna("").astype(str),
        " gemini_strong=" + table["gemini_strong_answer_norm"].fillna("").astype(str),
        " q14=" + table[f"{QWEN14}_answer_norm"].fillna("").astype(str),
        " q8=" + table[f"{QWEN8}_answer_norm"].fillna("").astype(str),
    ]
    text = text_parts[0]
    for part in text_parts[1:]:
        text = text + part
    table["rescue_feature_text"] = text
    table["local_vote_bin"] = table["local_ensemble_votes"].map(vote_bin)
    table["query_len_bin"] = table["query_len"].map(len_bin)
    table["number_count_bin"] = table["number_count"].map(num_bin)
    return table


def vote_bin(value: object) -> str:
    try:
        votes = int(float(value))
    except Exception:
        votes = 0
    return "3plus" if votes >= 3 else str(max(votes, 1))


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


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def agreement_action(row: pd.Series, baseline: str) -> str:
    if bool(row.get("tool_available", False)):
        return TOOL
    if as_bool(row.get("all_three_agree", False)):
        return QWEN8
    if as_bool(row.get("gemini_gpt_agree", False)):
        return GEMINI
    if as_bool(row.get("gemini_strong_agree_base", False)):
        return GEMINI_STRONG
    if as_bool(row.get("gemini_strong_agree_gemini", False)):
        return GEMINI
    votes = float(row.get("local_ensemble_votes", 0) or 0)
    if votes >= (2 if baseline == "agreement_min_cost" else 3):
        return str(row.get("local_ensemble_source") or QWEN8)
    if baseline == "agreement_qwen14_fallback":
        return QWEN14
    if baseline == "agreement_gemini_fallback":
        return GEMINI
    return GEMINI_STRONG


def baseline_actions(frame: pd.DataFrame, baseline: str) -> pd.Series:
    return pd.Series([agreement_action(row, baseline) for _, row in frame.iterrows()], index=frame.index)


def row_quality_cost(row: pd.Series, action: str, tool) -> tuple[float, float]:
    if action == TOOL:
        return float(row["tool_quality"]), 0.0
    return tool.row_quality_cost(row, action)


def action_quality_costs(frame: pd.DataFrame, actions: pd.Series, tool) -> tuple[np.ndarray, np.ndarray]:
    qualities = []
    costs = []
    for idx, row in frame.iterrows():
        quality, cost = row_quality_cost(row, str(actions.loc[idx]), tool)
        qualities.append(quality)
        costs.append(cost)
    return np.asarray(qualities, dtype=float), np.asarray(costs, dtype=float)


def cat_cols() -> list[str]:
    return [
        "dataset",
        "gemini_gpt_agree",
        "gemini_strong_agree_base",
        "gemini_strong_agree_gemini",
        "all_three_agree",
        "local_vote_bin",
        "query_len_bin",
        "number_count_bin",
    ]


def num_cols() -> list[str]:
    return [
        "query_len",
        "number_count",
        "latex_count",
        "frac_count",
        "sqrt_count",
        "local_ensemble_votes",
        "gemini_strong_cost",
        f"{GEMINI}_cost",
        f"{BASE_GPT}_cost",
        "strong_cost",
    ]


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
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000), "rescue_feature_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols()),
            ("num", StandardScaler(with_mean=False), num_cols()),
        ]
    )


def learners() -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesRegressor(n_estimators=240, random_state=23, n_jobs=-1),
        "random_forest": RandomForestRegressor(n_estimators=240, random_state=23, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=23),
        "ridge": Ridge(alpha=1.0),
    }


def rescue_train_target(train: pd.DataFrame, baseline: str, tool) -> np.ndarray:
    actions = baseline_actions(train, baseline)
    base_quality, base_cost = action_quality_costs(train, actions, tool)
    strong_quality = train["strong_quality"].astype(float).to_numpy()
    # Quality improvement is the target; cost is handled by the greedy budget layer.
    return strong_quality - base_quality


def apply_budgeted_rescue(
    frame: pd.DataFrame,
    baseline: str,
    scores: np.ndarray,
    *,
    min_score: float,
    budget_cap: float,
    tool,
) -> pd.Series:
    actions = baseline_actions(frame, baseline)
    _, base_costs = action_quality_costs(frame, actions, tool)
    strong_costs = frame["strong_cost"].astype(float).to_numpy()
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    current_cost = float(base_costs.sum())
    order = np.argsort(-scores)
    values = actions.copy()
    for pos in order:
        idx = frame.index[pos]
        if bool(frame.iloc[pos].get("tool_available", False)):
            continue
        if float(scores[pos]) < float(min_score):
            continue
        action = str(values.loc[idx])
        _, old_cost = row_quality_cost(frame.iloc[pos], action, tool)
        new_cost = float(strong_costs[pos])
        proposed = current_cost - old_cost + new_cost
        if proposed / strong_norm <= float(budget_cap):
            values.loc[idx] = STRONG_GPT
            current_cost = proposed
    return values


def evaluate_actions(frame: pd.DataFrame, actions: pd.Series, *, method: str, lambda_cost: float, tool) -> dict[str, object]:
    qualities, costs = action_quality_costs(frame, actions, tool)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    mean_quality = float(qualities.mean())
    normalized_cost = float(costs.sum() / strong_norm)
    mean_utility = mean_quality - float(lambda_cost) * normalized_cost
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_strong_inclusive_oracle": float(oracle_quality - mean_quality),
        "normalized_cost_vs_all_strong": normalized_cost,
        "utility_ratio_to_strong_inclusive_oracle": float(mean_utility / oracle_utility),
        "frontier_call_rate": float(np.mean([str(action) not in {TOOL, QWEN8, QWEN14, "qwen3-4b-local", "qwen3-0.6b-probe"} for action in actions])),
        "action_counts": json.dumps(actions.astype(str).value_counts().to_dict(), sort_keys=True),
    }


def run_grid(table: pd.DataFrame, args: argparse.Namespace, tool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = prepare_features(table[table["split"].eq("train")].copy())
    eval_frames = {
        split: prepare_features(frame.copy())
        for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False)
    }
    rows = []
    baselines = ["agreement_min_cost", "agreement_quality", "agreement_qwen14_fallback", "agreement_gemini_fallback"]
    thresholds = [-0.05, 0.0, 0.03, 0.06, 0.1, 0.15]
    budgets = [0.2, 0.25, 0.3, 0.35]
    for baseline in baselines:
        y = rescue_train_target(train, baseline, tool)
        for learner_name, learner in learners().items():
            model = make_pipeline(preprocessor(), learner)
            model.fit(train, y)
            pred_by_split = {split: model.predict(frame) for split, frame in eval_frames.items()}
            for threshold in thresholds:
                for budget in budgets:
                    method = f"budgeted_rescue_{baseline}_{learner_name}_thr{threshold:g}_budget{budget:g}"
                    for split, frame in eval_frames.items():
                        actions = apply_budgeted_rescue(
                            frame,
                            baseline,
                            pred_by_split[split],
                            min_score=threshold,
                            budget_cap=budget,
                            tool=tool,
                        )
                        rows.append(evaluate_actions(frame, actions, method=method, lambda_cost=args.lambda_cost, tool=tool))
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
        "frontier_call_rate",
        "action_counts",
    ]
    lines = [
        "# Budgeted Strong-Rescue Policy",
        "",
        "This policy trains a rescue-benefit regressor on train rows and spends a normalized cost budget on predicted strong-GPT rescue rows.",
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
    output_dir.joinpath("BUDGETED_STRONG_RESCUE_POLICY_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table, tool, conf = load_table()
    grid, selected = run_grid(table, args, tool)
    grid.to_csv(output_dir / "table_budgeted_strong_rescue_policy.csv", index=False)
    selected.to_csv(output_dir / "table_budgeted_strong_rescue_policy_selected.csv", index=False)
    fresh_grids = []
    fresh_selected = []
    for offset in range(args.fresh_seeds):
        seed = int(args.first_fresh_seed + offset)
        fresh_table = conf.recompute_strong_inclusive_oracle(conf.stratified_resplit(table, seed), tool, args.lambda_cost)
        fresh_grid, fresh_sel = run_grid(fresh_table, args, tool)
        fresh_grid.insert(0, "fresh_seed", seed)
        fresh_sel.insert(0, "fresh_seed", seed)
        fresh_grids.append(fresh_grid)
        fresh_selected.append(fresh_sel)
    fresh_grid_df = pd.concat(fresh_grids, ignore_index=True)
    fresh_selected_df = pd.concat(fresh_selected, ignore_index=True)
    fresh_summary = summarize(fresh_selected_df, args)
    fresh_grid_df.to_csv(output_dir / "table_budgeted_strong_rescue_policy_fresh_grid.csv", index=False)
    fresh_selected_df.to_csv(output_dir / "table_budgeted_strong_rescue_policy_fresh_selected.csv", index=False)
    fresh_summary.to_csv(output_dir / "table_budgeted_strong_rescue_policy_fresh_summary.csv", index=False)
    write_memo(output_dir, selected, fresh_summary)
    print(f"Wrote budgeted strong-rescue policy results to {output_dir}")
    print(selected.to_string(index=False))
    print(fresh_summary.to_string(index=False))


if __name__ == "__main__":
    main()
