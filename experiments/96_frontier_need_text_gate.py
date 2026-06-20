from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
FRONTIER_MODELS = ["gemini-3.5-flash", "gpt-5.5"]
GEMINI = "gemini-3.5-flash"
GPT = "gpt-5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text+local-output gate for frontier-only exact-math rows.")
    parser.add_argument(
        "--query-table",
        default="results/controlled/expanded_local_pool_qwen14/query_table_expanded_local_pool.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/frontier_need_text_gate")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--frontier-rate-target", type=float, default=0.40)
    return parser.parse_args()


def add_feature_text(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    parts = [
        table["query_text"].fillna("").astype(str),
        " dataset=" + table["dataset"].fillna("").astype(str),
    ]
    for model_id in LOCAL_MODELS:
        answer_col = f"{model_id}_answer_norm" if f"{model_id}_answer_norm" in table.columns else f"{model_id}_answer"
        parts.append(f" {model_id}=" + table[answer_col].fillna("").astype(str))
    table["feature_text"] = parts[0]
    for part in parts[1:]:
        table["feature_text"] = table["feature_text"] + part
    return table


def add_targets(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    table = table.copy()
    local_quality_cols = [f"{model_id}_quality" for model_id in LOCAL_MODELS]
    frontier_quality_cols = [f"{model_id}_quality" for model_id in FRONTIER_MODELS]
    local_utility_cols = [f"{model_id}_quality" for model_id in LOCAL_MODELS]
    frontier_utility_cols = [f"{GEMINI}_utility_selected_cost", f"{GPT}_utility_selected_cost"]

    local_qualities = table[local_quality_cols].copy()
    local_qualities.columns = LOCAL_MODELS
    frontier_qualities = table[frontier_quality_cols].copy()
    frontier_qualities.columns = FRONTIER_MODELS
    local_utilities = table[local_utility_cols].copy()
    local_utilities.columns = LOCAL_MODELS
    frontier_utilities = table[frontier_utility_cols].copy()
    frontier_utilities.columns = FRONTIER_MODELS

    table["local_oracle_model"] = local_qualities.idxmax(axis=1)
    table["frontier_quality_oracle_model"] = frontier_qualities.idxmax(axis=1)
    table["frontier_utility_oracle_model"] = frontier_utilities.idxmax(axis=1)
    table["frontier_only_needed"] = (
        frontier_qualities.max(axis=1).gt(local_qualities.max(axis=1))
        & frontier_qualities.max(axis=1).gt(0.5)
    ).astype(int)
    table["expanded_cost_oracle_utility"] = pd.concat([local_utilities, frontier_utilities], axis=1).max(axis=1)
    return table


def feature_columns(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = ["dataset"]
    cat_cols += [column for column in table.columns if column.startswith("agree__")]
    cat_cols += [
        column
        for column in ["qwen8_4b_agree", "qwen8_06b_agree", "small_pair_agree", "all_three_agree"]
        if column in table
    ]
    num_cols = [
        "query_len",
        "number_count",
        "latex_count",
        "frac_count",
        "sqrt_count",
        "local_max_vote",
        "local_ensemble_votes",
    ]
    num_cols += [f"{model_id}_answer_len" for model_id in LOCAL_MODELS if f"{model_id}_answer_len" in table]
    return list(dict.fromkeys(cat_cols)), list(dict.fromkeys(num_cols))


def make_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=25000), "feature_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ]
    )


def prepare_features(table: pd.DataFrame, cat_cols: Iterable[str], num_cols: Iterable[str]) -> pd.DataFrame:
    table = table.copy()
    for col in cat_cols:
        table[col] = table[col].fillna(False).astype(str)
    for col in num_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    table["feature_text"] = table["feature_text"].fillna("").astype(str)
    return table


def train_models(table: pd.DataFrame) -> dict[str, object]:
    cat_cols, num_cols = feature_columns(table)
    table = prepare_features(table, cat_cols, num_cols)
    train = table[table["split"].eq("train")]
    feature_cols = ["feature_text"] + cat_cols + num_cols
    preprocessor = make_preprocessor(cat_cols, num_cols)
    models: dict[str, object] = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
    for name, target, class_weight in [
        ("frontier_needed", "frontier_only_needed", "balanced"),
        ("local_selector", "local_oracle_model", "balanced"),
        ("frontier_quality_selector", "frontier_quality_oracle_model", "balanced"),
        ("frontier_utility_selector", "frontier_utility_oracle_model", "balanced"),
    ]:
        clf = make_pipeline(
            preprocessor,
            LogisticRegression(max_iter=5000, C=1.5, class_weight=class_weight, solver="saga", random_state=42),
        )
        clf.fit(train[feature_cols], train[target])
        models[name] = clf
    return models


def pick_frontier_action(strategy: str, frame: pd.DataFrame, models: dict[str, object]) -> pd.Series:
    if strategy == "gemini":
        return pd.Series(GEMINI, index=frame.index)
    if strategy == "gpt":
        return pd.Series(GPT, index=frame.index)
    feature_cols = models["feature_cols"]
    if strategy == "frontier_quality_selector":
        return pd.Series(models["frontier_quality_selector"].predict(frame[feature_cols]), index=frame.index)
    if strategy == "frontier_utility_selector":
        return pd.Series(models["frontier_utility_selector"].predict(frame[feature_cols]), index=frame.index)
    raise ValueError(strategy)


def evaluate_actions(
    frame: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    lambda_cost: float,
    all_gpt_cost: float,
) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        if action in LOCAL_MODELS:
            quality = float(row[f"{action}_quality"])
            cost = 0.0
            gpt = False
            gemini = False
        elif action == "local_ensemble":
            quality = float(row["local_ensemble_quality"])
            cost = 0.0
            gpt = False
            gemini = False
        elif action == GEMINI:
            quality = float(row[f"{GEMINI}_quality"])
            cost = float(row[f"{GEMINI}_cost"])
            gpt = False
            gemini = True
        elif action == GPT:
            quality = float(row[f"{GPT}_quality"])
            cost = float(row[f"{GPT}_cost"])
            gpt = True
            gemini = False
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        gpt_calls.append(gpt)
        gemini_calls.append(gemini)
    mean_quality = float(np.mean(qualities))
    normalized_cost = float(np.sum(costs) / all_gpt_cost) if all_gpt_cost > 0 else 0.0
    mean_utility = float(mean_quality - lambda_cost * normalized_cost)
    oracle_utility = float(frame["expanded_cost_oracle_utility"].mean())
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_expanded_oracle": float(frame["expanded_quality_oracle"].mean() - mean_quality),
        "utility_ratio_to_expanded_cost_oracle": float(mean_utility / oracle_utility) if oracle_utility else np.nan,
        "normalized_remote_cost_vs_all_gpt": normalized_cost,
        "frontier_call_rate": float(np.mean([a or b for a, b in zip(gpt_calls, gemini_calls)])),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "action_counts": json.dumps({str(key): int(value) for key, value in actions.value_counts().to_dict().items()}),
    }


def evaluate(table: pd.DataFrame, models: dict[str, object], lambda_cost: float) -> pd.DataFrame:
    table = prepare_features(table, models["cat_cols"], models["num_cols"])
    feature_cols = models["feature_cols"]
    rows: list[dict[str, object]] = []
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    budget_rates = [0.25, 0.30, 0.35, 0.40]
    strategies = ["gemini", "gpt", "frontier_quality_selector", "frontier_utility_selector"]
    for split, frame in table.groupby("split", sort=False):
        if split == "train":
            continue
        all_gpt_cost = float(frame[f"{GPT}_cost"].sum())
        local_actions = pd.Series(models["local_selector"].predict(frame[feature_cols]), index=frame.index)
        frontier_prob = pd.Series(models["frontier_needed"].predict_proba(frame[feature_cols])[:, 1], index=frame.index)
        for strategy in strategies:
            frontier_actions = pick_frontier_action(strategy, frame, models)
            for threshold in thresholds:
                actions = local_actions.copy()
                actions.loc[frontier_prob.ge(threshold)] = frontier_actions.loc[frontier_prob.ge(threshold)]
                row = evaluate_actions(
                    frame,
                    actions,
                    method=f"text_frontier_need_{strategy}_t{threshold:.2f}",
                    lambda_cost=lambda_cost,
                    all_gpt_cost=all_gpt_cost,
                )
                row["threshold"] = threshold
                row["budget_rate"] = np.nan
                row["frontier_strategy"] = strategy
                rows.append(row)
            for budget_rate in budget_rates:
                actions = local_actions.copy()
                budget = int(np.floor(budget_rate * len(frame)))
                if budget > 0:
                    frontier_index = frontier_prob.sort_values(ascending=False).head(budget).index
                    actions.loc[frontier_index] = frontier_actions.loc[frontier_index]
                row = evaluate_actions(
                    frame,
                    actions,
                    method=f"text_frontier_need_{strategy}_budget{budget_rate:.2f}",
                    lambda_cost=lambda_cost,
                    all_gpt_cost=all_gpt_cost,
                )
                row["threshold"] = np.nan
                row["budget_rate"] = budget_rate
                row["frontier_strategy"] = strategy
                rows.append(row)
        rows.append(
            evaluate_actions(
                frame,
                local_actions,
                method="text_local_selector_only",
                lambda_cost=lambda_cost,
                all_gpt_cost=all_gpt_cost,
            )
        )
        rows[-1]["threshold"] = np.nan
        rows[-1]["budget_rate"] = np.nan
        rows[-1]["frontier_strategy"] = "none"
    return pd.DataFrame(rows)


def select_validation_rows(rows: pd.DataFrame, quality_gap_target: float, frontier_rate_target: float) -> pd.DataFrame:
    val = rows[rows["split"].eq("val")].copy()
    feasible = val[
        val["quality_gap_to_expanded_oracle"].le(quality_gap_target)
        & val["frontier_call_rate"].le(frontier_rate_target)
    ].copy()
    if feasible.empty:
        feasible = val.copy()
        feasible["selection_status"] = "no_validation_feasible_row"
    else:
        feasible["selection_status"] = "validation_feasible"
    feasible = feasible.sort_values(
        ["selection_status", "utility_ratio_to_expanded_cost_oracle", "mean_quality"],
        ascending=[True, False, False],
    )
    selected = feasible.head(5)[
        [
            "method",
            "selection_status",
            "mean_quality",
            "quality_gap_to_expanded_oracle",
            "utility_ratio_to_expanded_cost_oracle",
            "normalized_remote_cost_vs_all_gpt",
            "frontier_call_rate",
            "gpt_call_rate",
            "action_counts",
            "threshold",
            "budget_rate",
            "frontier_strategy",
        ]
    ].copy()
    test = rows[rows["split"].eq("test")].copy()
    out = selected.merge(
        test,
        on="method",
        suffixes=("_val", "_test"),
        how="left",
    )
    return out


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    table = add_feature_text(table)
    table = add_targets(table, args.lambda_cost)
    models = train_models(table)
    rows = evaluate(table, models, args.lambda_cost)
    selected = select_validation_rows(rows, args.quality_gap_target, args.frontier_rate_target)

    rows_path = output_dir / "table_frontier_need_text_gate.csv"
    selected_path = output_dir / "table_frontier_need_text_gate_selected.csv"
    table_path = output_dir / "query_table_frontier_need_text_gate.csv"
    memo_path = output_dir / "FRONTIER_NEED_TEXT_GATE_MEMO.md"
    table.to_csv(table_path, index=False)
    rows.to_csv(rows_path, index=False)
    selected.to_csv(selected_path, index=False)

    best_test = rows[rows["split"].eq("test")].sort_values(
        ["utility_ratio_to_expanded_cost_oracle", "mean_quality"], ascending=False
    ).head(10)
    memo = [
        "# Frontier-Need Text Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        "This trains on the train split only, selects thresholds on validation, and reports held-out test rows.",
        "Features include query text, dataset, cheap local answers, local-only answer agreement, and answer lengths.",
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Test Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Interpretation",
        "",
        "This is a deployable-gate probe: no gold labels, oracle outcomes, or test labels are used at prediction time. The method is successful only if the validation-selected test row meets the quality, utility, cost, and frontier-rate targets.",
        "",
        "## Files",
        "",
        f"- `{table_path}`",
        f"- `{rows_path}`",
        f"- `{selected_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
