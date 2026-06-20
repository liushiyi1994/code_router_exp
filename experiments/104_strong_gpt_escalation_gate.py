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


GPT = "gpt-5.5"
GEMINI = "gemini-3.5-flash"
LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cached strong-GPT escalation policies on the held-out exact-math slice."
    )
    parser.add_argument(
        "--query-table",
        default="results/controlled/gpt_strong_solver_probe_medium_2048/query_table_with_gpt_strong_solver.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/gpt_strong_escalation_gate")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--fit-splits", default="train")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--test-split", default="test")
    return parser.parse_args()


def add_feature_text(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    query = table["query_text"].fillna("").astype(str)
    dataset = " dataset=" + table["dataset"].fillna("").astype(str)
    local_text = query + dataset
    for model_id in LOCAL_MODELS:
        answer_col = f"{model_id}_answer_norm" if f"{model_id}_answer_norm" in table.columns else f"{model_id}_answer"
        if answer_col in table.columns:
            local_text = local_text + f" {model_id}=" + table[answer_col].fillna("").astype(str)
    table["feature_text_query_only"] = query + dataset
    table["feature_text_query_local"] = local_text
    table["feature_text_with_base_answer"] = local_text + " gpt=" + table[f"{GPT}_answer"].fillna("").astype(str)
    table["feature_text_with_gemini_answer"] = local_text + " gemini=" + table[f"{GEMINI}_answer"].fillna("").astype(str)
    table["feature_text_with_base_and_gemini_answer"] = (
        table["feature_text_with_base_answer"] + " gemini=" + table[f"{GEMINI}_answer"].fillna("").astype(str)
    )
    return table


def base_feature_columns(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = ["dataset"]
    cat_cols += [column for column in table.columns if column.startswith("agree__")]
    cat_cols += [
        column
        for column in [
            "qwen8_4b_agree",
            "qwen8_06b_agree",
            "small_pair_agree",
            "all_three_agree",
            "qwen3-0.6b-probe_gemini_agree",
            "qwen3-4b-local_gemini_agree",
            "qwen3-8b-local_gemini_agree",
            "qwen3-14b-awq-local_gemini_agree",
        ]
        if column in table.columns
    ]
    num_cols = ["query_len", "number_count", "latex_count", "frac_count", "sqrt_count"]
    num_cols += [column for column in ["local_max_vote", "local_ensemble_votes"] if column in table.columns]
    num_cols += [f"{model_id}_answer_len" for model_id in LOCAL_MODELS if f"{model_id}_answer_len" in table.columns]
    return list(dict.fromkeys(cat_cols)), list(dict.fromkeys(num_cols))


def feature_sets(table: pd.DataFrame) -> dict[str, dict[str, object]]:
    cat_cols, num_cols = base_feature_columns(table)
    gemini_num = [
        column
        for column in [
            "gemini_prompt_tokens",
            "gemini_candidate_tokens",
            "gemini_thoughts_tokens",
            "gemini_total_tokens",
            "gemini_meta_latency_s",
        ]
        if column in table.columns
    ]
    gemini_cat = [column for column in ["verifier_verdict"] if column in table.columns]
    base_num = [column for column in [f"{GPT}_answer_len", f"{GPT}_latency"] if column in table.columns]
    return {
        "query_only": {
            "text_col": "feature_text_query_only",
            "cat_cols": ["dataset"],
            "num_cols": ["query_len", "number_count", "latex_count", "frac_count", "sqrt_count"],
            "uses_base_answer": False,
            "uses_gemini_answer": False,
        },
        "query_local": {
            "text_col": "feature_text_query_local",
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "uses_base_answer": False,
            "uses_gemini_answer": False,
        },
        "query_local_gemini": {
            "text_col": "feature_text_with_gemini_answer",
            "cat_cols": list(dict.fromkeys(cat_cols + gemini_cat)),
            "num_cols": list(dict.fromkeys(num_cols + gemini_num)),
            "uses_base_answer": False,
            "uses_gemini_answer": True,
        },
        "with_base_answer": {
            "text_col": "feature_text_with_base_answer",
            "cat_cols": cat_cols,
            "num_cols": list(dict.fromkeys(num_cols + base_num)),
            "uses_base_answer": True,
            "uses_gemini_answer": False,
        },
        "with_base_and_gemini": {
            "text_col": "feature_text_with_base_and_gemini_answer",
            "cat_cols": list(dict.fromkeys(cat_cols + gemini_cat)),
            "num_cols": list(dict.fromkeys(num_cols + base_num + gemini_num)),
            "uses_base_answer": True,
            "uses_gemini_answer": True,
        },
    }


def prepare_features(table: pd.DataFrame, cat_cols: Iterable[str], num_cols: Iterable[str]) -> pd.DataFrame:
    table = table.copy()
    for col in cat_cols:
        if col in table.columns:
            table[col] = table[col].fillna(False).astype(str)
    for col in num_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    return table


def make_preprocessor(text_col: str, cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=12000), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ]
    )


def classifier_specs() -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced", min_samples_leaf=1, random_state=7
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", min_samples_leaf=1, random_state=7
        ),
        "logreg": LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, solver="liblinear", random_state=7),
        "gradient_boosting": GradientBoostingClassifier(random_state=7),
    }


def positive_scores(model: object, frame: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    estimator = model[-1]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame[feature_cols])
        classes = list(getattr(estimator, "classes_", []))
        if 1 in classes:
            return pd.Series(proba[:, classes.index(1)], index=frame.index)
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    if hasattr(model, "decision_function"):
        return pd.Series(model.decision_function(frame[feature_cols]), index=frame.index)
    return pd.Series(model.predict(frame[feature_cols]), index=frame.index)


def evaluate_policy(
    frame: pd.DataFrame,
    scores: pd.Series,
    *,
    method: str,
    k: int,
    lambda_cost: float,
    uses_base_answer: bool,
    uses_gemini_answer: bool,
) -> dict[str, object]:
    k = min(max(int(k), 0), len(frame))
    escalated_index = set(scores.sort_values(ascending=False).head(k).index) if k else set()
    escalated = frame.index.to_series().isin(escalated_index).to_numpy()
    base_quality = frame[f"{GPT}_quality"].astype(float).fillna(0.0).to_numpy()
    strong_quality = frame["strong_quality"].astype(float).fillna(0.0).to_numpy()
    quality = np.where(escalated, strong_quality, base_quality)
    base_cost = frame[f"{GPT}_cost"].astype(float).fillna(0.0).to_numpy()
    strong_cost = frame["strong_cost"].astype(float).fillna(0.0).to_numpy()
    gemini_cost = frame[f"{GEMINI}_cost"].astype(float).fillna(0.0).to_numpy()
    base_cost_sum = max(float(base_cost.sum()), 1e-12)

    replacement_cost = np.where(escalated, strong_cost, base_cost)
    additive_cost = base_cost + np.where(escalated, strong_cost, 0.0)
    replacement_with_obs_cost = replacement_cost + (gemini_cost if uses_gemini_answer else 0.0)
    additive_with_obs_cost = additive_cost + (gemini_cost if uses_gemini_answer else 0.0)

    normalized_replacement = float(replacement_cost.sum() / base_cost_sum)
    normalized_additive = float(additive_cost.sum() / base_cost_sum)
    normalized_replacement_with_obs = float(replacement_with_obs_cost.sum() / base_cost_sum)
    normalized_additive_with_obs = float(additive_with_obs_cost.sum() / base_cost_sum)
    mean_quality = float(np.mean(quality))
    oracle_quality = float(frame["expanded_quality_oracle"].astype(float).mean())
    base_plus_strong_oracle_quality = float(np.maximum(base_quality, strong_quality).mean())
    replacement_utility = mean_quality - lambda_cost * normalized_replacement
    additive_utility = mean_quality - lambda_cost * normalized_additive
    replacement_with_obs_utility = mean_quality - lambda_cost * normalized_replacement_with_obs
    additive_with_obs_utility = mean_quality - lambda_cost * normalized_additive_with_obs
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "k_strong_calls": int(k),
        "strong_call_rate": float(np.mean(escalated)),
        "mean_quality": mean_quality,
        "base_gpt_quality": float(np.mean(base_quality)),
        "strong_gpt_quality": float(np.mean(strong_quality)),
        "expanded_quality_oracle": oracle_quality,
        "base_plus_strong_oracle_quality": base_plus_strong_oracle_quality,
        "quality_gap_to_expanded_oracle": oracle_quality - mean_quality,
        "quality_gap_to_base_plus_strong_oracle": base_plus_strong_oracle_quality - mean_quality,
        "normalized_replacement_cost_vs_all_gpt": normalized_replacement,
        "normalized_additive_cost_vs_all_gpt": normalized_additive,
        "normalized_replacement_with_observation_cost_vs_all_gpt": normalized_replacement_with_obs,
        "normalized_additive_with_observation_cost_vs_all_gpt": normalized_additive_with_obs,
        "replacement_utility_vs_all_gpt_norm": float(replacement_utility),
        "additive_utility_vs_all_gpt_norm": float(additive_utility),
        "replacement_with_observation_utility_vs_all_gpt_norm": float(replacement_with_obs_utility),
        "additive_with_observation_utility_vs_all_gpt_norm": float(additive_with_obs_utility),
        "remote_call_rate_replacement": 1.0,
        "remote_call_rate_additive": float(1.0 + np.mean(escalated)),
        "uses_base_answer_features": bool(uses_base_answer),
        "uses_gemini_answer_features": bool(uses_gemini_answer),
        "replacement_protocol_valid": bool(not uses_base_answer),
        "n_base_wrong_strong_right_selected": int(
            np.sum(escalated & (base_quality < 1.0) & (strong_quality > 0.0))
        ),
        "n_base_right_strong_wrong_selected": int(
            np.sum(escalated & (base_quality > 0.0) & (strong_quality < 1.0))
        ),
    }


def reference_rows(table: pd.DataFrame, lambda_cost: float, calibration_split: str, test_split: str) -> pd.DataFrame:
    rows = []
    for split in [calibration_split, test_split]:
        frame = table[table["split"].eq(split)].copy()
        zero_scores = pd.Series(0.0, index=frame.index)
        rows.append(
            evaluate_policy(
                frame,
                zero_scores,
                method="reference_all_base_gpt",
                k=0,
                lambda_cost=lambda_cost,
                uses_base_answer=False,
                uses_gemini_answer=False,
            )
        )
        rows.append(
            evaluate_policy(
                frame,
                zero_scores,
                method="reference_all_strong_gpt",
                k=len(frame),
                lambda_cost=lambda_cost,
                uses_base_answer=False,
                uses_gemini_answer=False,
            )
        )
    return pd.DataFrame(rows)


def run_policies(table: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = add_feature_text(table)
    fit_splits = {item.strip() for item in str(args.fit_splits).split(",") if item.strip()}
    kept_splits = set(fit_splits) | {args.calibration_split, args.test_split}
    table = table[table["split"].isin(kept_splits)].copy()
    table["strong_beats_base"] = (
        table["strong_quality"].astype(float).fillna(0.0) > table[f"{GPT}_quality"].astype(float).fillna(0.0)
    ).astype(int)
    rows = [reference_rows(table, args.lambda_cost, args.calibration_split, args.test_split)]
    predictions = []
    k_values = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 26]
    for feature_name, spec in feature_sets(table).items():
        text_col = str(spec["text_col"])
        cat_cols = [col for col in spec["cat_cols"] if col in table.columns]
        num_cols = [col for col in spec["num_cols"] if col in table.columns]
        feature_cols = [text_col] + cat_cols + num_cols
        working = prepare_features(table, cat_cols, num_cols)
        fitting = working[working["split"].isin(fit_splits)].dropna(subset=["strong_quality"]).copy()
        if fitting["strong_beats_base"].nunique() < 2:
            continue
        for clf_name, clf in classifier_specs().items():
            model = make_pipeline(make_preprocessor(text_col, cat_cols, num_cols), clf)
            model.fit(fitting[feature_cols], fitting["strong_beats_base"])
            for split in [args.calibration_split, args.test_split]:
                frame = working[working["split"].eq(split)].copy()
                scores = positive_scores(model, frame, feature_cols)
                pred_frame = pd.DataFrame(
                    {
                        "query_id": frame["query_id"].to_numpy(),
                        "split": split,
                        "feature_set": feature_name,
                        "classifier": clf_name,
                        "score": scores.to_numpy(),
                        "strong_beats_base": frame["strong_beats_base"].to_numpy(),
                    }
                )
                predictions.append(pred_frame)
                policy_rows = [
                    evaluate_policy(
                        frame,
                        scores,
                        method=f"{feature_name}_{clf_name}_top{k}",
                        k=k,
                        lambda_cost=args.lambda_cost,
                        uses_base_answer=bool(spec["uses_base_answer"]),
                        uses_gemini_answer=bool(spec["uses_gemini_answer"]),
                    )
                    for k in k_values
                ]
                for row in policy_rows:
                    row["feature_set"] = feature_name
                    row["classifier"] = clf_name
                rows.append(pd.DataFrame(policy_rows))
    return pd.concat(rows, ignore_index=True), pd.concat(predictions, ignore_index=True)


def selected_rows(results: pd.DataFrame, target_gap: float, calibration_split: str, test_split: str) -> pd.DataFrame:
    rows = []
    val = results[results["split"].eq(calibration_split)].copy()
    rules = [
        ("val_min_valid_replacement_cost_within_gap", "normalized_replacement_cost_vs_all_gpt", True),
        ("val_best_valid_replacement_utility", "replacement_utility_vs_all_gpt_norm", False),
        ("val_best_additive_utility", "additive_utility_vs_all_gpt_norm", False),
    ]
    for rule, column, ascending in rules:
        candidates = val.copy()
        if "valid_replacement" in rule:
            candidates = candidates[candidates["replacement_protocol_valid"]]
        feasible = candidates[candidates["quality_gap_to_expanded_oracle"].le(target_gap + 1e-12)]
        if feasible.empty:
            feasible = candidates.sort_values("quality_gap_to_expanded_oracle", ascending=True).head(20)
        picked = feasible.sort_values(column, ascending=ascending).head(1)
        if picked.empty:
            continue
        method = str(picked.iloc[0]["method"])
        for split in [calibration_split, test_split]:
            match = results[results["split"].eq(split) & results["method"].eq(method)].copy()
            match["selection_rule"] = rule
            rows.append(match)

    test = results[results["split"].eq(test_split)].copy()
    diagnostic = test[test["quality_gap_to_expanded_oracle"].le(target_gap + 1e-12)]
    if not diagnostic.empty:
        for column, label, valid_only in [
            ("normalized_replacement_cost_vs_all_gpt", "test_diagnostic_min_valid_replacement_cost_within_gap", True),
            ("normalized_additive_cost_vs_all_gpt", "test_diagnostic_min_additive_cost_within_gap", False),
        ]:
            candidates = diagnostic.copy()
            if valid_only:
                candidates = candidates[candidates["replacement_protocol_valid"]]
            if candidates.empty:
                continue
            picked = candidates.sort_values(column, ascending=True).head(1).copy()
            picked["selection_rule"] = label
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


def write_memo(output_dir: Path, results: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> None:
    test = results[results["split"].eq(args.test_split)].copy()
    key_cols = [
        "selection_rule",
        "method",
        "split",
        "k_strong_calls",
        "mean_quality",
        "quality_gap_to_expanded_oracle",
        "normalized_replacement_cost_vs_all_gpt",
        "normalized_additive_cost_vs_all_gpt",
        "strong_call_rate",
        "replacement_protocol_valid",
    ]
    best_diag = (
        test[
            test["quality_gap_to_expanded_oracle"].le(args.quality_gap_target + 1e-12)
            & test["replacement_protocol_valid"]
        ]
        .sort_values("normalized_replacement_cost_vs_all_gpt")
        .head(8)
    )
    memo = f"""# Strong GPT Escalation Gate Memo

Input table: `{args.query_table}`

This run uses cached GPT-5.5 medium-effort strong-solver outputs only. It makes no API calls.
The fit split(s) are `{args.fit_splits}`, the calibration split is `{args.calibration_split}`, and the held-out split is `{args.test_split}`.

The target is to predict rows where the strong GPT solver beats the base GPT-5.5 solver. Policies
rank rows by predicted rescue value and call the strong solver for the top-k rows.

## Selected Policies

{markdown_table(selected, [col for col in key_cols if col in selected.columns])}

## Best Held-Out Diagnostics Within {args.quality_gap_target:.2f} Quality Gap

These rows are diagnostic because they are selected on held-out test performance.

{markdown_table(best_diag, [col for col in key_cols if col in best_diag.columns])}

## Interpretation

Strong GPT escalation is a quality path, not a cost-aware ProbeRoute++ success under the current target.
The held-out diagnostics can match or beat the repaired expanded oracle quality, but normalized remote
cost remains above the all-base-GPT cost because the strong solver is much more expensive than the base
GPT call. Feature sets that use base GPT answers also require paying the base GPT call before escalation,
so their replacement-cost columns are not a deployable pre-routing protocol.
"""
    (output_dir / "STRONG_GPT_ESCALATION_GATE_MEMO.md").write_text(memo, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    required = {f"{GPT}_quality", f"{GPT}_cost", "strong_quality", "strong_cost", "expanded_quality_oracle"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    results, predictions = run_policies(table, args)
    selected = selected_rows(results, args.quality_gap_target, args.calibration_split, args.test_split)
    results.to_csv(output_dir / "table_strong_gpt_escalation_gate.csv", index=False)
    predictions.to_csv(output_dir / "table_strong_gpt_escalation_predictions.csv", index=False)
    selected.to_csv(output_dir / "table_strong_gpt_escalation_selected.csv", index=False)
    write_memo(output_dir, results, selected, args)


if __name__ == "__main__":
    main()
