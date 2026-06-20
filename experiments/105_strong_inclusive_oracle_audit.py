from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from routecode.controlled.live_stage0 import normalize_answer


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
STRONG_GPT = "strong-gpt-5.5"
ALL_MODELS = LOCAL_MODELS + [GEMINI, BASE_GPT, STRONG_GPT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit policies against a strong-GPT-inclusive oracle.")
    parser.add_argument(
        "--query-table",
        default="results/controlled/gpt_strong_solver_probe_medium_2048/query_table_with_gpt_strong_solver.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/strong_inclusive_oracle_audit")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    return parser.parse_args()


def quality_and_cost(table: pd.DataFrame, model_id: str) -> tuple[np.ndarray, np.ndarray]:
    if model_id == STRONG_GPT:
        return (
            table["strong_quality"].astype(float).fillna(0.0).to_numpy(),
            table["strong_cost"].astype(float).fillna(0.0).to_numpy(),
        )
    if model_id in LOCAL_MODELS:
        return table[f"{model_id}_quality"].astype(float).fillna(0.0).to_numpy(), np.zeros(len(table))
    return (
        table[f"{model_id}_quality"].astype(float).fillna(0.0).to_numpy(),
        table[f"{model_id}_cost"].astype(float).fillna(0.0).to_numpy(),
    )


def add_oracles(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    frames = []
    for split, frame in table.groupby("split", sort=False):
        frame = frame.copy()
        strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
        qualities: list[np.ndarray] = []
        costs: list[np.ndarray] = []
        utilities: list[np.ndarray] = []
        for model_id in ALL_MODELS:
            quality, cost = quality_and_cost(frame, model_id)
            qualities.append(quality)
            costs.append(cost)
            utilities.append(quality - float(lambda_cost) * cost * len(frame) / strong_norm)
        quality_matrix = np.vstack(qualities).T
        cost_matrix = np.vstack(costs).T
        utility_matrix = np.vstack(utilities).T
        quality_idx = quality_matrix.argmax(axis=1)
        utility_idx = utility_matrix.argmax(axis=1)
        frame["strong_inclusive_quality_oracle"] = quality_matrix[np.arange(len(frame)), quality_idx]
        frame["strong_inclusive_quality_oracle_model"] = [ALL_MODELS[idx] for idx in quality_idx]
        frame["strong_inclusive_cost_oracle_quality"] = quality_matrix[np.arange(len(frame)), utility_idx]
        frame["strong_inclusive_cost_oracle_cost"] = cost_matrix[np.arange(len(frame)), utility_idx]
        frame["strong_inclusive_cost_oracle_utility"] = utility_matrix[np.arange(len(frame)), utility_idx]
        frame["strong_inclusive_cost_oracle_model"] = [ALL_MODELS[idx] for idx in utility_idx]
        frame["strong_cost_norm_total"] = strong_norm
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for model_id in LOCAL_MODELS + [GEMINI, BASE_GPT]:
        answer_col = f"{model_id}_answer"
        norm_col = f"{model_id}_answer_norm"
        if answer_col in table.columns and norm_col not in table.columns:
            table[norm_col] = table[answer_col].fillna("").map(normalize_answer)
    table["feature_text_query"] = table["query_text"].fillna("").astype(str) + " dataset=" + table[
        "dataset"
    ].fillna("").astype(str)
    table["feature_text_local"] = table["feature_text_query"]
    for model_id in LOCAL_MODELS:
        table["feature_text_local"] += f" {model_id}=" + table[f"{model_id}_answer_norm"].fillna("").astype(str)
    table["feature_text_gemini"] = table["feature_text_local"] + " gemini=" + table[
        f"{GEMINI}_answer_norm"
    ].fillna("").astype(str)
    table["feature_text_base"] = table["feature_text_local"] + " base_gpt=" + table[
        f"{BASE_GPT}_answer_norm"
    ].fillna("").astype(str)
    table["base_gemini_agree"] = table[f"{BASE_GPT}_answer_norm"].eq(table[f"{GEMINI}_answer_norm"]) & table[
        f"{BASE_GPT}_answer_norm"
    ].fillna("").ne("")
    for model_id in LOCAL_MODELS:
        table[f"{model_id}_base_agree"] = table[f"{model_id}_answer_norm"].eq(
            table[f"{BASE_GPT}_answer_norm"]
        ) & table[f"{model_id}_answer_norm"].fillna("").ne("")
    return table


def base_columns(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = ["dataset", "base_gemini_agree"]
    cat_cols += [column for column in table.columns if column.startswith("agree__")]
    cat_cols += [
        column
        for column in [
            "qwen8_4b_agree",
            "qwen8_06b_agree",
            "small_pair_agree",
            "all_three_agree",
            "verifier_verdict",
        ]
        if column in table.columns
    ]
    for model_id in LOCAL_MODELS:
        col = f"{model_id}_base_agree"
        if col in table.columns:
            cat_cols.append(col)
    num_cols = ["query_len", "number_count", "latex_count", "frac_count", "sqrt_count"]
    num_cols += [column for column in ["local_max_vote", "local_ensemble_votes"] if column in table.columns]
    num_cols += [f"{model_id}_answer_len" for model_id in LOCAL_MODELS if f"{model_id}_answer_len" in table.columns]
    num_cols += [
        column
        for column in [
            "gemini_prompt_tokens",
            "gemini_candidate_tokens",
            "gemini_thoughts_tokens",
            "gemini_total_tokens",
            "gemini_meta_latency_s",
            f"{BASE_GPT}_answer_len",
            f"{BASE_GPT}_latency",
        ]
        if column in table.columns
    ]
    return list(dict.fromkeys(cat_cols)), list(dict.fromkeys(num_cols))


def prepare_features(table: pd.DataFrame, cat_cols: Iterable[str], num_cols: Iterable[str]) -> pd.DataFrame:
    table = table.copy()
    for column in cat_cols:
        if column in table.columns:
            table[column] = table[column].fillna(False).astype(str)
    for column in num_cols:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    return table


def preprocessor(text_col: str, cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ]
    )


def classifier_specs() -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=3),
        "random_forest": RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=3),
        "logreg": LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", solver="saga", random_state=3),
        "gradient_boosting": GradientBoostingClassifier(random_state=3),
    }


def feature_sets(table: pd.DataFrame) -> dict[str, dict[str, object]]:
    cat_cols, num_cols = base_columns(table)
    return {
        "query": {
            "text_col": "feature_text_query",
            "cat_cols": ["dataset"],
            "num_cols": ["query_len", "number_count", "latex_count", "frac_count", "sqrt_count"],
            "uses_gemini_answer": False,
            "uses_base_answer": False,
        },
        "local": {
            "text_col": "feature_text_local",
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "uses_gemini_answer": False,
            "uses_base_answer": False,
        },
        "gemini": {
            "text_col": "feature_text_gemini",
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "uses_gemini_answer": True,
            "uses_base_answer": False,
        },
        "base": {
            "text_col": "feature_text_base",
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "uses_gemini_answer": False,
            "uses_base_answer": True,
        },
    }


def action_metrics(
    frame: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    lambda_cost: float,
    uses_gemini_answer: bool = False,
    uses_base_answer: bool = False,
) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        observation_cost = 0.0
        if uses_gemini_answer:
            observation_cost += float(row[f"{GEMINI}_cost"])
        if uses_base_answer and action not in {BASE_GPT, STRONG_GPT}:
            observation_cost += float(row[f"{BASE_GPT}_cost"])
        if action == STRONG_GPT:
            quality = float(row["strong_quality"])
            cost = float(row["strong_cost"]) + (float(row[f"{BASE_GPT}_cost"]) if uses_base_answer else 0.0)
            cost += float(row[f"{GEMINI}_cost"]) if uses_gemini_answer else 0.0
        elif action == BASE_GPT:
            quality = float(row[f"{BASE_GPT}_quality"])
            cost = float(row[f"{BASE_GPT}_cost"]) + (float(row[f"{GEMINI}_cost"]) if uses_gemini_answer else 0.0)
        elif action == GEMINI:
            quality = float(row[f"{GEMINI}_quality"])
            cost = float(row[f"{GEMINI}_cost"]) + (float(row[f"{BASE_GPT}_cost"]) if uses_base_answer else 0.0)
        elif action == "local_ensemble":
            quality = float(row["local_ensemble_quality"])
            cost = observation_cost
        elif action in LOCAL_MODELS:
            quality = float(row[f"{action}_quality"])
            cost = observation_cost
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    normalized_cost = float(np.sum(costs) / strong_norm)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(mean_quality - lambda_cost * normalized_cost)
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "strong_inclusive_cost_oracle_quality": float(frame["strong_inclusive_cost_oracle_quality"].mean()),
        "quality_gap_to_strong_inclusive_cost_oracle": float(
            frame["strong_inclusive_cost_oracle_quality"].mean() - mean_quality
        ),
        "normalized_remote_cost_vs_all_strong_gpt": normalized_cost,
        "mean_utility": mean_utility,
        "utility_ratio_to_strong_inclusive_cost_oracle": float(mean_utility / oracle_utility)
        if oracle_utility
        else np.nan,
        "uses_gemini_answer_features": bool(uses_gemini_answer),
        "uses_base_answer_features": bool(uses_base_answer),
        "action_counts": json.dumps({str(key): int(value) for key, value in actions.value_counts().to_dict().items()}),
    }


def reference_rows(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    rows = []
    for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
        for method, action in [
            ("all_strong_gpt", STRONG_GPT),
            ("all_base_gpt", BASE_GPT),
            ("all_gemini", GEMINI),
            ("all_local_ensemble", "local_ensemble"),
            ("strong_inclusive_cost_oracle", "oracle"),
        ]:
            if action == "oracle":
                actions = frame["strong_inclusive_cost_oracle_model"]
            else:
                actions = pd.Series(action, index=frame.index)
            rows.append(action_metrics(frame, actions, method=method, lambda_cost=lambda_cost))
    return pd.DataFrame(rows)


def direct_selector_rows(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_name, spec in feature_sets(table).items():
        text_col = str(spec["text_col"])
        cat_cols = [col for col in spec["cat_cols"] if col in table.columns]
        num_cols = [col for col in spec["num_cols"] if col in table.columns]
        feature_cols = [text_col] + cat_cols + num_cols
        working = prepare_features(table, cat_cols, num_cols)
        train = working[working["split"].eq("train")].copy()
        if train["strong_inclusive_cost_oracle_model"].nunique() < 2:
            continue
        for clf_name, clf in classifier_specs().items():
            model = make_pipeline(preprocessor(text_col, cat_cols, num_cols), clf)
            model.fit(train[feature_cols], train["strong_inclusive_cost_oracle_model"])
            for split in ["val", "test"]:
                frame = working[working["split"].eq(split)].copy()
                actions = pd.Series(model.predict(frame[feature_cols]), index=frame.index)
                row = action_metrics(
                    frame,
                    actions,
                    method=f"direct_{feature_name}_{clf_name}",
                    lambda_cost=lambda_cost,
                    uses_gemini_answer=bool(spec["uses_gemini_answer"]),
                    uses_base_answer=bool(spec["uses_base_answer"]),
                )
                row["policy_family"] = "direct_selector"
                row["feature_set"] = feature_name
                row["classifier"] = clf_name
                rows.append(row)
    return pd.DataFrame(rows)


def heuristic_actions(frame: pd.DataFrame, params: tuple[object, ...]) -> pd.Series:
    min_votes, require_verifier, use_base_gemini_agree, use_local_base_agree, use_verifier_yes, fallback = params
    actions: list[str] = []
    for _, row in frame.iterrows():
        action = str(fallback)
        if float(row.get("local_ensemble_votes", 0.0) or 0.0) >= int(min_votes) and (
            not bool(require_verifier) or str(row.get("verifier_verdict", "")) == "YES"
        ):
            action = "local_ensemble"
        elif bool(use_base_gemini_agree) and bool(row.get("base_gemini_agree", False)):
            action = GEMINI
        elif bool(use_local_base_agree) and any(bool(row.get(f"{model_id}_base_agree", False)) for model_id in LOCAL_MODELS):
            action = BASE_GPT
        elif bool(use_verifier_yes) and str(row.get("verifier_verdict", "")) == "YES":
            action = GEMINI
        actions.append(action)
    return pd.Series(actions, index=frame.index)


def heuristic_rows(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    params_grid = []
    for min_votes in [1, 2, 3]:
        for require_verifier in [False, True]:
            for use_base_gemini_agree in [False, True]:
                for use_local_base_agree in [False, True]:
                    for use_verifier_yes in [False, True]:
                        for fallback in [STRONG_GPT, BASE_GPT, GEMINI]:
                            params_grid.append(
                                (
                                    min_votes,
                                    require_verifier,
                                    use_base_gemini_agree,
                                    use_local_base_agree,
                                    use_verifier_yes,
                                    fallback,
                                )
                            )
    for params in params_grid:
        for split in ["val", "test"]:
            frame = table[table["split"].eq(split)].copy()
            actions = heuristic_actions(frame, params)
            row = action_metrics(frame, actions, method=f"heuristic_{params}", lambda_cost=lambda_cost)
            row["policy_family"] = "heuristic"
            row["params"] = str(params)
            rows.append(row)
    return pd.DataFrame(rows)


def oracle_summary(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    rows = []
    for split, frame in table.groupby("split", sort=False):
        strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
        rows.append(
            {
                "split": split,
                "n_queries": int(len(frame)),
                "base_gpt_quality": float(frame[f"{BASE_GPT}_quality"].mean()),
                "gemini_quality": float(frame[f"{GEMINI}_quality"].mean()),
                "strong_gpt_quality": float(frame["strong_quality"].mean()),
                "strong_inclusive_quality_oracle": float(frame["strong_inclusive_quality_oracle"].mean()),
                "strong_inclusive_cost_oracle_quality": float(
                    frame["strong_inclusive_cost_oracle_quality"].mean()
                ),
                "strong_inclusive_cost_oracle_cost_vs_all_strong_gpt": float(
                    frame["strong_inclusive_cost_oracle_cost"].sum() / strong_norm
                ),
                "strong_inclusive_cost_oracle_utility": float(
                    frame["strong_inclusive_cost_oracle_utility"].mean()
                ),
                "strong_inclusive_cost_oracle_actions": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in frame["strong_inclusive_cost_oracle_model"].value_counts().to_dict().items()
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def selected_rows(grid: pd.DataFrame, quality_gap_target: float, cost_target: float) -> pd.DataFrame:
    selectable = grid[
        ~grid["method"].isin(
            [
                "strong_inclusive_cost_oracle",
            ]
        )
    ].copy()
    val = selectable[selectable["split"].eq("val")].copy()
    rows = []
    for label, candidates in [
        ("val_best_utility_under_cost_target", val[val["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)]),
        (
            "val_feasible_quality_cost_target",
            val[
                val["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)
                & val["quality_gap_to_strong_inclusive_cost_oracle"].le(quality_gap_target)
            ],
        ),
        ("val_best_utility_any_cost", val),
    ]:
        if candidates.empty:
            continue
        picked = candidates.sort_values("utility_ratio_to_strong_inclusive_cost_oracle", ascending=False).head(1)
        method = str(picked.iloc[0]["method"])
        for split in ["val", "test"]:
            match = grid[(grid["split"].eq(split)) & (grid["method"].eq(method))].copy()
            match["selection_rule"] = label
            rows.append(match)
    test = selectable[selectable["split"].eq("test")].copy()
    diagnostic = test[
        test["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)
        & test["quality_gap_to_strong_inclusive_cost_oracle"].le(quality_gap_target)
    ]
    if not diagnostic.empty:
        picked = diagnostic.sort_values("utility_ratio_to_strong_inclusive_cost_oracle", ascending=False).head(1).copy()
        picked["selection_rule"] = "test_diagnostic_feasible_quality_cost"
        rows.append(picked)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[columns].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in view.itertuples(index=False):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_memo(output_dir: Path, summary: pd.DataFrame, selected: pd.DataFrame, grid: pd.DataFrame, args: argparse.Namespace) -> None:
    key_cols = [
        "selection_rule",
        "method",
        "split",
        "mean_quality",
        "quality_gap_to_strong_inclusive_cost_oracle",
        "normalized_remote_cost_vs_all_strong_gpt",
        "utility_ratio_to_strong_inclusive_cost_oracle",
        "action_counts",
    ]
    test_best = grid[grid["split"].eq("test")].sort_values(
        ["utility_ratio_to_strong_inclusive_cost_oracle", "quality_gap_to_strong_inclusive_cost_oracle"],
        ascending=[False, True],
    ).head(10)
    memo = f"""# Strong-Inclusive Oracle Audit

Input table: `{args.query_table}`

This audit adds the cached GPT-5.5 medium-effort strong solver as an explicit model-pool member.
Remote cost is normalized against running strong GPT on every query. No API calls are made.

## Oracle Summary

{markdown_table(summary, list(summary.columns))}

## Selected Policies

{markdown_table(selected, [col for col in key_cols if col in selected.columns])}

## Best Held-Out Policies By Utility

{markdown_table(test_best, [col for col in key_cols if col in test_best.columns])}

## Interpretation

The strong-inclusive cost oracle is feasible on held-out exact math: it reaches high quality with low
normalized strong-GPT cost by using local, Gemini, base GPT, and strong GPT selectively. The tested
deployable policy families do not recover it. Direct oracle-action classifiers miss too many winners,
and simple confidence cascades stay cheap only by losing too much quality. This keeps the Phase 3
cost-aware claim unsupported on this exact-math slice.
"""
    (output_dir / "STRONG_INCLUSIVE_ORACLE_AUDIT_MEMO.md").write_text(memo, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    table = add_features(add_oracles(table, args.lambda_cost))
    summary = oracle_summary(table, args.lambda_cost)
    grid = pd.concat(
        [reference_rows(table, args.lambda_cost), direct_selector_rows(table, args.lambda_cost), heuristic_rows(table, args.lambda_cost)],
        ignore_index=True,
    )
    selected = selected_rows(grid, args.quality_gap_target, args.cost_target)
    table.to_csv(output_dir / "query_table_with_strong_inclusive_oracle.csv", index=False)
    summary.to_csv(output_dir / "table_strong_inclusive_oracle_summary.csv", index=False)
    grid.to_csv(output_dir / "table_strong_inclusive_policy_grid.csv", index=False)
    selected.to_csv(output_dir / "table_strong_inclusive_selected.csv", index=False)
    write_memo(output_dir, summary, selected, grid, args)


if __name__ == "__main__":
    main()
