from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LOCAL_MODELS = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
NO_TOOL_LOCAL_MODELS = [model for model in LOCAL_MODELS if model != "deterministic_math_tool"]
GAIN_QUANTILES = [0.0, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.975]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-agnostic model-action gain override policy. Train an action utility model on train rows, "
            "then replace the current concrete base only when a cheap local candidate is predicted to improve utility."
        )
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument(
        "--probe-features",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv"),
    )
    parser.add_argument(
        "--base-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_policy_variable_verifier_fusion/"
            "table_current_policy_variable_verifier_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default="base_current_policy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_model_action_gain_override_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--frontier-cap", type=float, default=0.40)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp204 = load_module("experiments/204_benchmark_agnostic_local_candidate_selector.py", "local_selector_204_for_211")

    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    probe_features = pd.read_csv(args.probe_features)
    matrix = exp204.build_matrix(outputs)
    base = exp204.load_base(args.base_query_choices, args.base_policy, matrix)
    action_features = build_action_features(outputs, probe_features, matrix)
    feature_cols = action_feature_columns(action_features)

    standard_all, standard_choices = run_scenario(
        matrix,
        base,
        action_features,
        feature_cols,
        train_ids=split_ids(matrix, "train"),
        val_ids=base_split_ids(base, "val"),
        test_ids=base_split_ids(base, "test"),
        scenario="standard",
        heldout="",
        args=args,
    )
    heldout_all = run_benchmark_heldout(matrix, base, action_features, feature_cols, args)
    selected = select_rows(standard_all, float(args.frontier_cap))
    heldout_selected = select_rows(heldout_all, float(args.frontier_cap)) if not heldout_all.empty else heldout_all
    heldout_summary = summarize_heldout(heldout_selected)
    selected_methods = set(selected["method"].astype(str))
    selected_choices = standard_choices[standard_choices["method"].astype(str).isin(selected_methods)].copy()

    action_features.to_csv(args.output_dir / "table_model_action_features.csv", index=False)
    standard_all.to_csv(args.output_dir / "table_model_action_gain_override_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_model_action_gain_override_selected.csv", index=False)
    heldout_all.to_csv(args.output_dir / "table_model_action_gain_override_benchmark_heldout_all.csv", index=False)
    heldout_selected.to_csv(
        args.output_dir / "table_model_action_gain_override_benchmark_heldout_selected.csv",
        index=False,
    )
    heldout_summary.to_csv(args.output_dir / "table_model_action_gain_override_benchmark_heldout_summary.csv", index=False)
    selected_choices.to_csv(args.output_dir / "table_model_action_gain_override_query_choices.csv", index=False)
    write_memo(
        args.output_dir / "MODEL_ACTION_GAIN_OVERRIDE_MEMO.md",
        args,
        feature_cols,
        selected,
        heldout_summary,
    )
    print(f"Wrote model-action gain override policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_action_features(outputs: pd.DataFrame, probe_features: pd.DataFrame, matrix: dict[str, Any]) -> pd.DataFrame:
    probe_cols = probe_feature_columns(probe_features)
    probe_index = probe_features.set_index("query_id")
    local_models = [model for model in LOCAL_MODELS if model in set(outputs["model_id"].astype(str))]
    rows: list[dict[str, Any]] = []
    for query_id, group in outputs.groupby("query_id", sort=False):
        qid = str(query_id)
        local_group = group[group["model_id"].astype(str).isin(local_models)].copy()
        answers = {str(row.model_id): normalize_answer(row.parsed_answer) for row in local_group.itertuples(index=False)}
        counts = Counter(answer for answer in answers.values() if answer)
        top_answer = counts.most_common(1)[0][0] if counts else ""
        top_count = counts[top_answer] if top_answer else 0
        second_count = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        valid_count = sum(1 for answer in answers.values() if answer)
        entropy = answer_entropy(counts)
        for item in group.itertuples(index=False):
            model_id = str(item.model_id)
            is_local = model_id in local_models
            answer = answers.get(model_id, "") if is_local else ""
            support = counts[answer] if answer else 0
            margin_reference = second_count if answer == top_answer else top_count
            row: dict[str, Any] = {
                "query_id": qid,
                "model_id": model_id,
                "split": str(item.split),
                "benchmark": str(item.benchmark),
                "domain": str(item.domain),
                "target_utility": float(item.utility),
                "target_quality": float(item.quality_score),
                "action_is_local": float(bool(item.is_local)),
                "action_is_frontier": float(bool(item.is_frontier)),
                "action_is_probe": float(bool(getattr(item, "is_probe", False))),
                "local_answer_valid": float(bool(answer)),
                "local_answer_chars": float(len(answer)),
                "local_group_support": float(support),
                "local_group_frac": float(support / max(valid_count, 1)),
                "local_group_margin": float((support - margin_reference) / max(valid_count, 1)),
                "local_is_top_group": float(bool(answer and answer == top_answer)),
                "local_answer_entropy": float(entropy),
                "local_valid_count": float(valid_count),
            }
            for candidate in matrix["model_ids"]:
                row[f"action_model_{safe_name(candidate)}"] = float(model_id == candidate)
            provider = str(getattr(item, "provider", ""))
            for provider_name in ["local", "tool", "openai", "google"]:
                row[f"action_provider_{provider_name}"] = float(provider == provider_name)
            for local_model in local_models:
                key = safe_name(local_model)
                row[f"local_action_is_{key}"] = float(model_id == local_model)
                row[f"local_agrees_with_{key}"] = float(bool(answer and answers.get(local_model, "") == answer))
            if qid in probe_index.index:
                for column in probe_cols:
                    row[f"probe_{column}"] = probe_index.loc[qid, column]
            rows.append(row)
    frame = pd.DataFrame(rows)
    cols = action_feature_columns(frame)
    frame[cols] = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame.sort_values(["split", "benchmark", "query_id", "model_id"]).reset_index(drop=True)


def probe_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {"query_id", "query_text", "split", "benchmark", "domain", "metric"}
    cols: list[str] = []
    for col in frame.columns:
        lower = str(col).lower()
        if col in blocked or lower.startswith("tool_") or lower.startswith("benchmark_id_"):
            continue
        if "train_prior" in lower:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def action_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {"query_id", "model_id", "split", "benchmark", "domain", "target_utility", "target_quality"}
    return [col for col in frame.columns if col not in blocked and pd.api.types.is_numeric_dtype(frame[col])]


def run_benchmark_heldout(
    matrix: dict[str, Any],
    base: pd.DataFrame,
    action_features: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_ids = set(base["query_id"].astype(str))
    for heldout in sorted(matrix["meta"]["benchmark"].astype(str).unique()):
        train_ids = matrix["meta"][
            matrix["meta"]["split"].astype(str).eq("train")
            & matrix["meta"]["benchmark"].astype(str).ne(heldout)
        ].index.astype(str).tolist()
        val_ids = [
            query_id
            for query_id in matrix["meta"][
                matrix["meta"]["split"].astype(str).eq("val")
                & matrix["meta"]["benchmark"].astype(str).ne(heldout)
            ].index.astype(str).tolist()
            if query_id in base_ids
        ]
        test_ids = [
            query_id
            for query_id in matrix["meta"][
                matrix["meta"]["split"].astype(str).eq("test")
                & matrix["meta"]["benchmark"].astype(str).eq(heldout)
            ].index.astype(str).tolist()
            if query_id in base_ids
        ]
        if not train_ids or not val_ids or not test_ids:
            continue
        table, _ = run_scenario(
            matrix,
            base,
            action_features,
            feature_cols,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            scenario="benchmark_heldout",
            heldout=heldout,
            args=args,
        )
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_scenario(
    matrix: dict[str, Any],
    base: pd.DataFrame,
    action_features: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    scenario: str,
    heldout: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_series = base.set_index("query_id")["base_model"].astype(str)
    rows: list[dict[str, Any]] = []
    choices: list[pd.DataFrame] = []
    for frame, method, family, diagnostic in reference_choice_frames(matrix, base_series, val_ids, test_ids):
        choices.append(frame)
        rows.extend(metric_rows(frame, method, family, args, scenario, heldout, diagnostic=diagnostic))

    for predictor_name, predictions in fit_action_utility_models(action_features, feature_cols, train_ids, [*val_ids, *test_ids], args):
        scored = action_features.merge(predictions, on=["query_id", "model_id"], how="left")
        for scope_name, candidate_models in {
            "no_tool_locals": [model for model in NO_TOOL_LOCAL_MODELS if model in matrix["model_ids"]],
            "all_locals_diagnostic": [model for model in LOCAL_MODELS if model in matrix["model_ids"]],
        }.items():
            route_candidates = candidate_rows(scored, base_series, [*val_ids, *test_ids], candidate_models)
            frames = gain_override_choice_frames(
                matrix,
                base_series,
                route_candidates,
                val_ids,
                test_ids,
                predictor_name,
                scope_name,
            )
            for choice_frame, method, family, diagnostic in frames:
                choices.append(choice_frame)
                rows.extend(metric_rows(choice_frame, method, family, args, scenario, heldout, diagnostic=diagnostic))
    return pd.DataFrame(rows), pd.concat(choices, ignore_index=True) if choices else pd.DataFrame()


def fit_action_utility_models(
    action_features: pd.DataFrame,
    feature_cols: list[str],
    train_ids: list[str],
    eval_ids: list[str],
    args: argparse.Namespace,
) -> list[tuple[str, pd.DataFrame]]:
    train = action_features[action_features["query_id"].astype(str).isin(train_ids)].copy()
    eval_frame = action_features[action_features["query_id"].astype(str).isin(eval_ids)].copy()
    specs: list[tuple[str, Any]] = [
        ("ridge_alpha10", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))),
        (
            "extratrees_leaf8",
            ExtraTreesRegressor(n_estimators=180, min_samples_leaf=8, random_state=int(args.seed), n_jobs=-1),
        ),
        (
            "hgb_l2_0.1",
            HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, l2_regularization=0.1, random_state=int(args.seed)),
        ),
    ]
    out: list[tuple[str, pd.DataFrame]] = []
    for name, model in specs:
        model.fit(train[feature_cols], train["target_utility"].astype(float))
        pred = eval_frame[["query_id", "model_id"]].copy()
        pred["predicted_utility"] = model.predict(eval_frame[feature_cols])
        out.append((name, pred))
    return out


def candidate_rows(scored: pd.DataFrame, base_series: pd.Series, ids: list[str], local_candidates: list[str]) -> pd.DataFrame:
    wanted: list[dict[str, str]] = []
    for query_id in ids:
        if query_id not in base_series.index:
            continue
        wanted.append({"query_id": str(query_id), "model_id": str(base_series.loc[query_id])})
        for model in local_candidates:
            wanted.append({"query_id": str(query_id), "model_id": str(model)})
    wanted_frame = pd.DataFrame(wanted).drop_duplicates()
    frame = scored.merge(wanted_frame, on=["query_id", "model_id"], how="inner").copy()
    frame["is_base_action"] = [
        str(row.model_id) == str(base_series.loc[str(row.query_id)]) if str(row.query_id) in base_series.index else False
        for row in frame.itertuples(index=False)
    ]
    return frame


def gain_override_choice_frames(
    matrix: dict[str, Any],
    base_series: pd.Series,
    candidates: pd.DataFrame,
    val_ids: list[str],
    test_ids: list[str],
    predictor_name: str,
    scope_name: str,
) -> list[tuple[pd.DataFrame, str, str, bool]]:
    gains = best_predicted_local_gains(candidates, base_series, [*val_ids, *test_ids])
    val_gains = gains[gains["query_id"].astype(str).isin(val_ids)]["predicted_gain"]
    thresholds = threshold_grid(val_gains)
    frames: list[tuple[pd.DataFrame, str, str, bool]] = []
    for threshold in thresholds:
        for mode in ["any", "if_base_frontier"]:
            method = f"{predictor_name}_{scope_name}_{mode}_gain_thr{threshold:.4f}"
            family = "model_action_gain_override" if scope_name == "no_tool_locals" else "model_action_gain_override_with_tool_diagnostic"
            frame = build_gain_choices(matrix, gains, [*val_ids, *test_ids], method, threshold, mode)
            frames.append((frame, method, family, scope_name != "no_tool_locals"))
    return frames


def best_predicted_local_gains(candidates: pd.DataFrame, base_series: pd.Series, ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = candidates[candidates["query_id"].astype(str).isin(ids)].copy()
    for query_id, group in frame.groupby("query_id", sort=False):
        qid = str(query_id)
        base_model = str(base_series.loc[qid])
        base = group[group["model_id"].astype(str).eq(base_model)]
        if base.empty:
            continue
        base_pred = float(base.iloc[0]["predicted_utility"])
        locals_only = group[~group["model_id"].astype(str).eq(base_model)].copy()
        if locals_only.empty:
            best = base.iloc[0]
        else:
            best = locals_only.sort_values("predicted_utility", ascending=False).iloc[0]
        rows.append(
            {
                "query_id": qid,
                "base_model": base_model,
                "candidate_model": str(best["model_id"]),
                "base_predicted_utility": base_pred,
                "candidate_predicted_utility": float(best["predicted_utility"]),
                "predicted_gain": float(best["predicted_utility"]) - base_pred,
            }
        )
    return pd.DataFrame(rows)


def build_gain_choices(
    matrix: dict[str, Any],
    gains_frame: pd.DataFrame,
    ids: list[str],
    method: str,
    threshold: float,
    mode: str,
) -> pd.DataFrame:
    gains = gains_frame[gains_frame["query_id"].astype(str).isin(ids)].set_index("query_id")
    rows: list[dict[str, Any]] = []
    for query_id in ids:
        if query_id not in gains.index:
            continue
        row = gains.loc[query_id]
        base_model = str(row["base_model"])
        candidate_model = str(row["candidate_model"])
        active = float(row["predicted_gain"]) >= float(threshold)
        if mode == "if_base_frontier":
            active = active and bool(matrix["frontier"].loc[query_id, base_model])
        selected = candidate_model if active else base_model
        rows.append(
            choice_row(
                matrix,
                query_id,
                selected,
                method,
                base_model=base_model,
                predicted_gain=float(row["predicted_gain"]),
                gain_threshold=float(threshold),
                base_predicted_utility=float(row["base_predicted_utility"]),
                candidate_predicted_utility=float(row["candidate_predicted_utility"]),
            )
        )
    return pd.DataFrame(rows)


def reference_choice_frames(
    matrix: dict[str, Any],
    base_series: pd.Series,
    val_ids: list[str],
    test_ids: list[str],
) -> list[tuple[pd.DataFrame, str, str, bool]]:
    frames: list[tuple[pd.DataFrame, str, str, bool]] = []
    no_tool_locals = [model for model in NO_TOOL_LOCAL_MODELS if model in matrix["model_ids"]]
    all_locals = [model for model in LOCAL_MODELS if model in matrix["model_ids"]]
    for ids in [val_ids, test_ids]:
        base_selected = base_series.reindex(ids).dropna().astype(str)
        frames.append((choices_from_series(matrix, base_selected, "current_base"), "current_base", "current_base", False))
        full_oracle = matrix["utility"].loc[ids].idxmax(axis=1).astype(str)
        frames.append((choices_from_series(matrix, full_oracle, "full_oracle"), "full_oracle", "full_oracle_upper_bound", True))
        for label, models in [
            ("current_base_plus_no_tool_locals_oracle", no_tool_locals),
            ("current_base_plus_all_locals_oracle", all_locals),
        ]:
            repaired: dict[str, str] = {}
            for query_id, base_model in base_selected.items():
                candidates = [str(base_model), *models]
                repaired[str(query_id)] = max(candidates, key=lambda model: float(matrix["utility"].loc[str(query_id), model]))
            frames.append((choices_from_series(matrix, pd.Series(repaired), label), label, f"{label}_upper_bound", True))
    return frames


def choices_from_series(matrix: dict[str, Any], selected: pd.Series, method: str) -> pd.DataFrame:
    return pd.DataFrame([choice_row(matrix, str(query_id), str(model), method) for query_id, model in selected.items()])


def choice_row(
    matrix: dict[str, Any],
    query_id: str,
    selected_model: str,
    method: str,
    *,
    base_model: str | None = None,
    predicted_gain: float = math.nan,
    gain_threshold: float = math.nan,
    base_predicted_utility: float = math.nan,
    candidate_predicted_utility: float = math.nan,
) -> dict[str, Any]:
    meta = matrix["meta"].loc[query_id]
    base = selected_model if base_model is None else base_model
    return {
        "query_id": query_id,
        "query_text": str(meta["query_text"]),
        "split": str(meta["split"]),
        "benchmark": str(meta["benchmark"]),
        "domain": str(meta["domain"]),
        "metric": str(meta["metric"]),
        "method": method,
        "base_model": base,
        "selected_model": selected_model,
        "selected_quality": float(matrix["quality"].loc[query_id, selected_model]),
        "selected_utility": float(matrix["utility"].loc[query_id, selected_model]),
        "selected_normalized_cost": float(matrix["cost"].loc[query_id, selected_model]),
        "selected_frontier": bool(matrix["frontier"].loc[query_id, selected_model]),
        "changed": bool(selected_model != base),
        "predicted_gain": float(predicted_gain),
        "gain_threshold": float(gain_threshold),
        "base_predicted_utility": float(base_predicted_utility),
        "candidate_predicted_utility": float(candidate_predicted_utility),
        "oracle_utility": float(matrix["oracle_utility"].loc[query_id]),
        "oracle_quality": float(matrix["oracle_quality"].loc[query_id]),
    }


def metric_rows(
    choices: pd.DataFrame,
    method: str,
    family: str,
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
    *,
    diagnostic: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if choices.empty:
        return rows
    for split, group in choices.groupby("split", sort=False):
        utility = group["selected_utility"].astype(float).to_numpy()
        quality = group["selected_quality"].astype(float).to_numpy()
        oracle_utility = group["oracle_utility"].astype(float).to_numpy()
        oracle_quality = group["oracle_quality"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(utility, int(args.bootstrap_samples), int(args.seed))
        rows.append(
            {
                "scenario": scenario,
                "heldout_benchmark": heldout,
                "method": method,
                "family": family,
                "split": str(split),
                "n_queries": int(len(group)),
                "mean_quality": float(quality.mean()),
                "mean_utility": float(utility.mean()),
                "mean_utility_ci_low": ci_low,
                "mean_utility_ci_high": ci_high,
                "mean_normalized_cost": float(group["selected_normalized_cost"].astype(float).mean()),
                "oracle_mean_quality": float(oracle_quality.mean()),
                "oracle_mean_utility": float(oracle_utility.mean()),
                "oracle_utility_ratio": float(utility.mean() / max(float(oracle_utility.mean()), 1e-12)),
                "utility_gap_to_oracle": float(oracle_utility.mean() - utility.mean()),
                "quality_gap_to_oracle": float(oracle_quality.mean() - quality.mean()),
                "frontier_call_rate": float(group["selected_frontier"].astype(bool).mean()),
                "override_rate": float(group["changed"].astype(bool).mean()),
                "diagnostic": bool(diagnostic),
                "selected_models_json": json.dumps(group["selected_model"].value_counts().sort_index().to_dict(), sort_keys=True),
            }
        )
    return rows


def select_rows(table: pd.DataFrame, frontier_cap: float) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in table.groupby(["scenario", "heldout_benchmark", "family"], dropna=False):
        val = group[group["split"].astype(str).eq("val")].copy()
        test = group[group["split"].astype(str).eq("test")].copy()
        if val.empty:
            continue
        best = val.sort_values(["mean_utility", "frontier_call_rate", "override_rate"], ascending=[False, True, True]).iloc[0]
        rows.extend(mark_selected_pair(best, test, "val_best_mean_utility"))
        capped = val[val["frontier_call_rate"].astype(float) <= frontier_cap].copy()
        if not capped.empty:
            best_cap = capped.sort_values(["mean_utility", "frontier_call_rate", "override_rate"], ascending=[False, True, True]).iloc[0]
            rows.extend(mark_selected_pair(best_cap, test, f"val_best_frontier_cap_{frontier_cap:g}"))
    standard_test = table[table["scenario"].astype(str).eq("standard") & table["split"].astype(str).eq("test")].copy()
    for _, row in standard_test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(12).iterrows():
        item = row.copy()
        item["selection_rule"] = "top_standard_test_diagnostic"
        rows.append(item)
    return pd.DataFrame(rows).drop_duplicates(["scenario", "heldout_benchmark", "family", "method", "split", "selection_rule"])


def mark_selected_pair(best: pd.Series, test: pd.DataFrame, rule: str) -> list[pd.Series]:
    rows: list[pd.Series] = []
    val_row = best.copy()
    val_row["selection_rule"] = rule
    rows.append(val_row)
    match = test[test["method"].astype(str).eq(str(best["method"]))]
    if not match.empty:
        test_row = match.iloc[0].copy()
        test_row["selection_rule"] = f"{rule}_test"
        rows.append(test_row)
    return rows


def summarize_heldout(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    test = selected[selected["split"].astype(str).eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    return (
        test.groupby("family", as_index=False)
        .agg(
            mean_heldout_quality=("mean_quality", "mean"),
            mean_heldout_utility=("mean_utility", "mean"),
            mean_heldout_oracle_ratio=("oracle_utility_ratio", "mean"),
            mean_frontier_call_rate=("frontier_call_rate", "mean"),
            mean_override_rate=("override_rate", "mean"),
            n_heldout_benchmarks=("heldout_benchmark", "nunique"),
        )
        .sort_values("mean_heldout_utility", ascending=False)
    )


def threshold_grid(values: pd.Series) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return [float("inf")]
    grid = {float(value) for value in np.quantile(clean.to_numpy(), GAIN_QUANTILES)}
    grid.add(0.0)
    grid.add(float(clean.max()) + 1e-9)
    return sorted(grid)


def split_ids(matrix: dict[str, Any], split: str) -> list[str]:
    return matrix["meta"][matrix["meta"]["split"].astype(str).eq(split)].index.astype(str).tolist()


def base_split_ids(base: pd.DataFrame, split: str) -> list[str]:
    return base[base["split"].astype(str).eq(split)]["query_id"].astype(str).tolist()


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def normalize_answer(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    return text.removeprefix("answer:").strip().strip("$").strip().rstrip(".")


def answer_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(value).replace("-", "_")).strip("_").lower()


def write_memo(
    path: Path,
    args: argparse.Namespace,
    feature_cols: list[str],
    selected: pd.DataFrame,
    heldout_summary: pd.DataFrame,
) -> None:
    cols = [
        "scenario",
        "heldout_benchmark",
        "family",
        "method",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "override_rate",
        "diagnostic",
        "selection_rule",
    ]
    standard = selected[selected["scenario"].astype(str).eq("standard")].copy() if not selected.empty else pd.DataFrame()
    lines = [
        "# Model-Action Gain Override Policy",
        "",
        "This cached experiment trains a benchmark-agnostic model-action utility predictor on train action rows,",
        "then uses validation-selected gain thresholds to decide whether a cheap local candidate should replace",
        "the current concrete base action.",
        "",
        "No provider calls, vLLM calls, local generation calls, or benchmark-specific verifier calls are made.",
        "Main `model_action_gain_override` rows exclude deterministic-tool local overrides; the tool-inclusive rows are diagnostics.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/211_model_action_gain_override_policy.py",
        "```",
        "",
        "## Feature Policy",
        "",
        f"- Numeric feature count: `{len(feature_cols)}`",
        "- Excludes benchmark ID, domain, metric, direct deterministic-tool probe columns, and outcome labels from features.",
        "- Uses local answer agreement/support features only for cheap local candidates already cached before final routing.",
        "",
        "## Standard Selected Rows",
        "",
        markdown_table(standard[[c for c in cols if c in standard.columns]]) if not standard.empty else "No selected rows.",
        "",
        "## Benchmark-Heldout Summary",
        "",
        markdown_table(heldout_summary) if not heldout_summary.empty else "No heldout rows.",
        "",
        "## Interpretation",
        "",
        "- `current_base_plus_no_tool_locals_oracle` is the diagnostic ceiling for replacing the current base with non-tool locals.",
        "- A valid improvement must appear in validation-selected test rows, not only in top-test diagnostics.",
        "- If the gain override does not beat `current_base`, the remaining bottleneck is candidate reliability observability.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
