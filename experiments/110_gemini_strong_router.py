from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from routecode.controlled.live_stage0 import normalize_answer


LOCAL_MODELS = [
    "qwen3-0.6b-probe",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
]
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
STRONG_GPT = "strong-gpt-5.5"
GEMINI_STRONG = "gemini-3.5-flash-strong-solve"
ALL_ACTIONS = LOCAL_MODELS + [GEMINI, BASE_GPT, GEMINI_STRONG, STRONG_GPT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn routers with Gemini strong-solver as an expanded action.")
    parser.add_argument(
        "--query-table",
        default="results/controlled/strong_inclusive_oracle_audit/query_table_with_strong_inclusive_oracle.csv",
    )
    parser.add_argument(
        "--gemini-train",
        default="results/controlled/gemini_strong_solver_probe_train/table_gemini_strong_solver_outputs.csv",
    )
    parser.add_argument(
        "--gemini-val",
        default="results/controlled/gemini_strong_solver_probe_val/table_gemini_strong_solver_outputs.csv",
    )
    parser.add_argument(
        "--gemini-test",
        default="results/controlled/gemini_strong_solver_probe_test/table_gemini_strong_solver_outputs.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/gemini_strong_router")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--utility-ratio-target", type=float, default=0.95)
    return parser.parse_args()


def merge_gemini_strong(table: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    gemini = pd.concat(frames, ignore_index=True).drop_duplicates("query_id", keep="last")
    gemini = gemini.rename(
        columns={
            "quality": "gemini_strong_quality",
            "cost_usd": "gemini_strong_cost",
            "parsed_answer": "gemini_strong_answer",
            "latency_s": "gemini_strong_latency",
        }
    )
    keep = [
        "query_id",
        "gemini_strong_quality",
        "gemini_strong_cost",
        "gemini_strong_answer",
        "gemini_strong_latency",
    ]
    merged = table.merge(gemini[keep], on="query_id", how="left")
    if merged["gemini_strong_quality"].isna().any():
        missing = merged.loc[merged["gemini_strong_quality"].isna(), "query_id"].head(5).tolist()
        raise ValueError(f"Missing Gemini strong outputs, examples: {missing}")
    return merged


def action_quality_cost(frame: pd.DataFrame, action: str) -> tuple[np.ndarray, np.ndarray]:
    if action == STRONG_GPT:
        return (
            frame["strong_quality"].astype(float).fillna(0.0).to_numpy(),
            frame["strong_cost"].astype(float).fillna(0.0).to_numpy(),
        )
    if action == GEMINI_STRONG:
        return (
            frame["gemini_strong_quality"].astype(float).fillna(0.0).to_numpy(),
            frame["gemini_strong_cost"].astype(float).fillna(0.0).to_numpy(),
        )
    if action in LOCAL_MODELS:
        return frame[f"{action}_quality"].astype(float).fillna(0.0).to_numpy(), np.zeros(len(frame))
    return (
        frame[f"{action}_quality"].astype(float).fillna(0.0).to_numpy(),
        frame[f"{action}_cost"].astype(float).fillna(0.0).to_numpy(),
    )


def add_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for model_id in LOCAL_MODELS + [GEMINI, BASE_GPT]:
        answer_col = f"{model_id}_answer"
        norm_col = f"{model_id}_answer_norm"
        if answer_col in table.columns:
            table[norm_col] = table[answer_col].fillna("").map(normalize_answer)
    table["gemini_strong_answer_norm"] = table["gemini_strong_answer"].fillna("").map(normalize_answer)
    table["feature_text_query"] = table["query_text"].fillna("").astype(str) + " dataset=" + table[
        "dataset"
    ].fillna("").astype(str)
    table["feature_text_local"] = table["feature_text_query"]
    for model_id in LOCAL_MODELS:
        table["feature_text_local"] += f" {model_id}=" + table[f"{model_id}_answer_norm"].fillna("").astype(str)
    table["feature_text_gemini"] = table["feature_text_local"] + " gemini=" + table[
        f"{GEMINI}_answer_norm"
    ].fillna("").astype(str)
    table["feature_text_base"] = table["feature_text_gemini"] + " base_gpt=" + table[
        f"{BASE_GPT}_answer_norm"
    ].fillna("").astype(str)
    table["feature_text_gemini_strong"] = table["feature_text_gemini"] + " gemini_strong=" + table[
        "gemini_strong_answer_norm"
    ].fillna("").astype(str)
    table["gemini_strong_agree_gemini"] = table["gemini_strong_answer_norm"].eq(table[f"{GEMINI}_answer_norm"])
    table["gemini_strong_agree_base"] = table["gemini_strong_answer_norm"].eq(table[f"{BASE_GPT}_answer_norm"])
    for model_id in LOCAL_MODELS:
        table[f"gemini_strong_agree_{model_id}"] = table["gemini_strong_answer_norm"].eq(
            table[f"{model_id}_answer_norm"]
        )
    return table


def feature_sets(table: pd.DataFrame) -> dict[str, dict[str, object]]:
    cat_base = ["dataset"]
    for column in [
        "qwen8_gemini_agree",
        "gemini_gpt_agree",
        "qwen8_4b_agree",
        "qwen8_06b_agree",
        "small_pair_agree",
        "all_three_agree",
        "gemini_strong_agree_gemini",
        "gemini_strong_agree_base",
    ]:
        if column in table.columns:
            cat_base.append(column)
    cat_base += [column for column in table.columns if column.startswith("agree__")]
    cat_base += [column for column in table.columns if column.startswith("gemini_strong_agree_qwen")]
    num_base = [
        "query_len",
        "number_count",
        "latex_count",
        "frac_count",
        "sqrt_count",
        "local_ensemble_votes",
        "gemini_strong_cost",
        "gemini_strong_latency",
    ]
    for model_id in LOCAL_MODELS + [GEMINI, BASE_GPT]:
        col = f"{model_id}_answer_len"
        if col in table.columns:
            num_base.append(col)
    return {
        "query": {
            "text_col": "feature_text_query",
            "cat_cols": ["dataset"],
            "num_cols": ["query_len", "number_count", "latex_count", "frac_count", "sqrt_count"],
            "observed": (),
        },
        "local": {
            "text_col": "feature_text_local",
            "cat_cols": cat_base,
            "num_cols": num_base,
            "observed": (),
        },
        "gemini_observed": {
            "text_col": "feature_text_gemini",
            "cat_cols": cat_base,
            "num_cols": num_base,
            "observed": (GEMINI,),
        },
        "gemini_strong_observed": {
            "text_col": "feature_text_gemini_strong",
            "cat_cols": cat_base,
            "num_cols": num_base,
            "observed": (GEMINI_STRONG,),
        },
        "base_observed": {
            "text_col": "feature_text_base",
            "cat_cols": cat_base,
            "num_cols": num_base,
            "observed": (GEMINI, BASE_GPT),
        },
    }


def prepare_features(table: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    table = table.copy()
    for col in cat_cols:
        if col in table.columns:
            table[col] = table[col].fillna(False).astype(str)
    for col in num_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    return table


def preprocessor(text_col: str, cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ]
    )


def add_oracle_labels(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    frames = []
    for split, frame in table.groupby("split", sort=False):
        frame = frame.copy()
        strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
        quality_cols = []
        utility_cols = []
        for action in ALL_ACTIONS:
            q, c = action_quality_cost(frame, action)
            quality_cols.append(q)
            utility_cols.append(q - float(lambda_cost) * c * len(frame) / strong_norm)
        qmat = np.vstack(quality_cols).T
        umat = np.vstack(utility_cols).T
        idx = umat.argmax(axis=1)
        frame["gemini_strong_cost_oracle_action"] = [ALL_ACTIONS[i] for i in idx]
        frame["gemini_strong_cost_oracle_quality"] = qmat[np.arange(len(frame)), idx]
        frame["gemini_strong_cost_oracle_utility"] = umat[np.arange(len(frame)), idx]
        frame["strong_cost_norm_total"] = strong_norm
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def observation_costs(frame: pd.DataFrame, observed: tuple[str, ...], selected: pd.Series) -> np.ndarray:
    costs = np.zeros(len(frame), dtype=float)
    selected_values = selected.astype(str).to_numpy()
    for model_id in observed:
        if model_id == GEMINI:
            model_cost = frame[f"{GEMINI}_cost"].astype(float).fillna(0.0).to_numpy()
        elif model_id == BASE_GPT:
            model_cost = frame[f"{BASE_GPT}_cost"].astype(float).fillna(0.0).to_numpy()
        elif model_id == GEMINI_STRONG:
            model_cost = frame["gemini_strong_cost"].astype(float).fillna(0.0).to_numpy()
        else:
            continue
        costs += np.where(selected_values == model_id, 0.0, model_cost)
    return costs


def evaluate_actions(
    frame: pd.DataFrame,
    actions: pd.Series,
    *,
    observed: tuple[str, ...],
    lambda_cost: float,
    method: str,
    feature_set: str,
    learner: str,
) -> dict[str, object]:
    qualities = np.zeros(len(frame), dtype=float)
    costs = np.zeros(len(frame), dtype=float)
    for action in ALL_ACTIONS:
        mask = actions.astype(str).eq(action).to_numpy()
        if not mask.any():
            continue
        q, c = action_quality_cost(frame, action)
        qualities[mask] = q[mask]
        costs[mask] = c[mask]
    costs += observation_costs(frame, observed, actions)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    utility = qualities - float(lambda_cost) * costs * len(frame) / strong_norm
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "feature_set": feature_set,
        "learner": learner,
        "n_queries": int(len(frame)),
        "mean_quality": float(qualities.mean()),
        "strong_inclusive_cost_oracle_quality": oracle_quality,
        "quality_gap_to_strong_inclusive_oracle": float(oracle_quality - qualities.mean()),
        "normalized_cost_vs_all_strong": float(costs.sum() / strong_norm),
        "mean_utility": float(utility.mean()),
        "utility_ratio_to_strong_inclusive_oracle": float(utility.mean() / oracle_utility),
        "action_counts": json.dumps(actions.astype(str).value_counts().to_dict(), sort_keys=True),
    }


def classifier_specs() -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=160,
            class_weight="balanced",
            random_state=11,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            class_weight="balanced",
            random_state=11,
            n_jobs=-1,
        ),
        "logreg": LogisticRegression(max_iter=1500, class_weight="balanced", solver="saga", random_state=11),
    }


def regressor_specs() -> dict[str, object]:
    return {
        "extra_trees_reg": ExtraTreesRegressor(n_estimators=160, random_state=17, n_jobs=-1),
        "ridge_reg": Ridge(alpha=1.0),
    }


def fit_expected_quality_router(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    feature_spec: dict[str, object],
    regressor: object,
    cost_scale: float,
    lambda_cost: float,
) -> pd.Series:
    text_col = str(feature_spec["text_col"])
    cat_cols = list(feature_spec["cat_cols"])
    num_cols = list(feature_spec["num_cols"])
    predictions = []
    for action in ALL_ACTIONS:
        y, _ = action_quality_cost(train, action)
        model = make_pipeline(preprocessor(text_col, cat_cols, num_cols), regressor)
        model.fit(train, y)
        pred_q = np.clip(model.predict(eval_frame), 0.0, 1.0)
        _, eval_cost = action_quality_cost(eval_frame, action)
        predictions.append(pred_q - float(lambda_cost) * eval_cost / max(cost_scale, 1e-12))
    scores = np.vstack(predictions).T
    return pd.Series([ALL_ACTIONS[i] for i in scores.argmax(axis=1)], index=eval_frame.index)


def run(table: pd.DataFrame, lambda_cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = add_features(table)
    table = add_oracle_labels(table, lambda_cost)
    train = table[table["split"].eq("train")].copy()
    val = table[table["split"].eq("val")].copy()
    test = table[table["split"].eq("test")].copy()
    train_cost_scale = max(float(train["strong_cost"].sum()) / max(len(train), 1), 1e-12)
    rows = []
    for feature_name, feature_spec in feature_sets(table).items():
        cat_cols = list(feature_spec["cat_cols"])
        num_cols = list(feature_spec["num_cols"])
        text_col = str(feature_spec["text_col"])
        observed = tuple(feature_spec["observed"])
        fit_train = prepare_features(train, cat_cols, num_cols)
        fit_val = prepare_features(val, cat_cols, num_cols)
        fit_test = prepare_features(test, cat_cols, num_cols)

        for learner_name, clf in classifier_specs().items():
            model = make_pipeline(preprocessor(text_col, cat_cols, num_cols), clf)
            model.fit(fit_train, fit_train["gemini_strong_cost_oracle_action"])
            for split_name, frame in [("val", fit_val), ("test", fit_test)]:
                pred = pd.Series(model.predict(frame), index=frame.index)
                rows.append(
                    evaluate_actions(
                        frame,
                        pred,
                        observed=observed,
                        lambda_cost=lambda_cost,
                        method="action_classifier",
                        feature_set=feature_name,
                        learner=learner_name,
                    )
                )

        for learner_name, reg in regressor_specs().items():
            for split_name, frame in [("val", fit_val), ("test", fit_test)]:
                pred = fit_expected_quality_router(
                    fit_train,
                    frame,
                    feature_spec=feature_spec,
                    regressor=reg,
                    cost_scale=train_cost_scale,
                    lambda_cost=lambda_cost,
                )
                rows.append(
                    evaluate_actions(
                        frame,
                        pred,
                        observed=observed,
                        lambda_cost=lambda_cost,
                        method="expected_quality_router",
                        feature_set=feature_name,
                        learner=learner_name,
                    )
                )

    # Diagnostics and always-on baselines for context.
    for action in ALL_ACTIONS:
        for frame in [val, test]:
            pred = pd.Series(action, index=frame.index)
            rows.append(
                evaluate_actions(
                    frame,
                    pred,
                    observed=(),
                    lambda_cost=lambda_cost,
                    method=f"all_{action}",
                    feature_set="constant",
                    learner="constant",
                )
            )
    for frame in [val, test]:
        pred = frame["strong_inclusive_cost_oracle_model"].copy()
        rows.append(
            evaluate_actions(
                frame,
                pred,
                observed=(),
                lambda_cost=lambda_cost,
                method="original_strong_inclusive_oracle",
                feature_set="oracle",
                learner="oracle",
            )
        )
        pred = frame["gemini_strong_cost_oracle_action"].copy()
        rows.append(
            evaluate_actions(
                frame,
                pred,
                observed=(),
                lambda_cost=lambda_cost,
                method="gemini_strong_inclusive_oracle",
                feature_set="oracle",
                learner="oracle",
            )
        )

    results = pd.DataFrame(rows)
    selected = select_results(results)
    return results, selected


def select_results(results: pd.DataFrame) -> pd.DataFrame:
    candidates = results[~results["method"].str.contains("oracle") & ~results["feature_set"].eq("constant")].copy()
    val = candidates[candidates["split"].eq("val")].copy()
    test = candidates[candidates["split"].eq("test")].copy()
    feasible = val[
        (val["quality_gap_to_strong_inclusive_oracle"] <= 0.03)
        & (val["normalized_cost_vs_all_strong"] <= 0.35)
        & (val["utility_ratio_to_strong_inclusive_oracle"] >= 0.95)
    ].copy()
    if len(feasible):
        chosen = feasible.sort_values(
            ["normalized_cost_vs_all_strong", "quality_gap_to_strong_inclusive_oracle"],
            ascending=[True, True],
        ).head(1)
        rule = "validation_feasible_min_cost"
    else:
        under_cost = val[val["normalized_cost_vs_all_strong"] <= 0.35].copy()
        chosen = under_cost.sort_values(
            ["quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle"],
            ascending=[True, False],
        ).head(1)
        rule = "no_validation_feasible_best_gap_under_cost"
    keys = ["method", "feature_set", "learner"]
    selected = chosen.copy()
    selected["selection_rule"] = rule
    if len(chosen):
        mask = np.logical_and.reduce([test[key].eq(chosen.iloc[0][key]) for key in keys])
        selected = pd.concat([selected, test[mask].assign(selection_rule="selected_test")], ignore_index=True)
    diag = test[test["normalized_cost_vs_all_strong"] <= 0.35].sort_values(
        ["quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle"],
        ascending=[True, False],
    ).head(1)
    if len(diag):
        selected = pd.concat([selected, diag.assign(selection_rule="best_heldout_diagnostic_under_cost")])
    return selected.reset_index(drop=True)


def write_memo(output_dir: Path, results: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "method",
        "split",
        "feature_set",
        "learner",
        "mean_quality",
        "quality_gap_to_strong_inclusive_oracle",
        "normalized_cost_vs_all_strong",
        "utility_ratio_to_strong_inclusive_oracle",
        "action_counts",
    ]
    lines = [
        "# Gemini-Strong Router Memo",
        "",
        "Purpose: test whether a thinking-enabled Gemini Flash solver can act as a cheap intermediate action/probe for the strong-inclusive exact-math router.",
        "",
        f"Rows evaluated: `{len(results)}` result rows.",
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
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Interpretation: validation may find Gemini-strong policies that look feasible, but the held-out exact-math slice remains the deciding evidence.",
            "",
        ]
    )
    output_dir.joinpath("GEMINI_STRONG_ROUTER_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    table = merge_gemini_strong(table, [args.gemini_train, args.gemini_val, args.gemini_test])
    results, selected = run(table, args.lambda_cost)
    results.to_csv(output_dir / "table_gemini_strong_router.csv", index=False)
    selected.to_csv(output_dir / "table_gemini_strong_router_selected.csv", index=False)
    write_memo(output_dir, results, selected)
    print(f"Wrote {len(results)} rows to {output_dir / 'table_gemini_strong_router.csv'}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
