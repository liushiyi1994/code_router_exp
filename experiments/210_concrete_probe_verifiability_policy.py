from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOCAL_MODELS = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
TOOL_MODEL = "deterministic_math_tool"
CLASSIFIERS = ["extratrees_d3_leaf8", "gb_depth2", "logreg_c0.3"]
VERIF_QUANTILES = [0.50, 0.65, 0.75, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
LOCAL_QUANTILES = [0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.975]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concrete benchmark-agnostic ProbeCode bridge: learn broad verifiability and local-candidate "
            "scores from train only, then test whether they can repair the current concrete base policy."
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
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
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
        default=Path("results/controlled/broad100_concrete_probe_verifiability_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--frontier-cap", type=float, default=0.40)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp204 = load_module("experiments/204_benchmark_agnostic_local_candidate_selector.py", "local_selector_204_for_210")

    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    probe_features = pd.read_csv(args.probe_features)
    target = pd.read_csv(args.target_table)
    matrix = exp204.build_matrix(outputs)
    base = exp204.load_base(args.base_query_choices, args.base_policy, matrix)
    candidate_features = exp204.build_candidate_features(outputs, probe_features, matrix)
    verif_frame, verif_features = build_verifiability_frame(target, probe_features, outputs)

    standard_all, standard_choices, standard_scores = run_scenario(
        exp204,
        matrix,
        base,
        candidate_features,
        verif_frame,
        verif_features,
        train_ids=split_ids(matrix, "train"),
        val_ids=base_split_ids(base, "val"),
        test_ids=base_split_ids(base, "test"),
        scenario="standard",
        heldout="",
        args=args,
    )
    heldout_all, heldout_scores = run_benchmark_heldout(
        exp204,
        matrix,
        base,
        candidate_features,
        verif_frame,
        verif_features,
        args,
    )
    selected = select_rows(standard_all, float(args.frontier_cap))
    heldout_selected = select_rows(heldout_all, float(args.frontier_cap)) if not heldout_all.empty else heldout_all
    selected_methods = set(selected["method"].dropna().astype(str))
    selected_choices = standard_choices[standard_choices["method"].astype(str).isin(selected_methods)].copy()
    score_table = pd.concat([standard_scores, heldout_scores], ignore_index=True)

    standard_all.to_csv(args.output_dir / "table_concrete_probe_verifiability_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_concrete_probe_verifiability_selected.csv", index=False)
    heldout_all.to_csv(args.output_dir / "table_concrete_probe_verifiability_benchmark_heldout_all.csv", index=False)
    heldout_selected.to_csv(
        args.output_dir / "table_concrete_probe_verifiability_benchmark_heldout_selected.csv",
        index=False,
    )
    selected_choices.to_csv(args.output_dir / "table_concrete_probe_verifiability_query_choices.csv", index=False)
    score_table.to_csv(args.output_dir / "table_concrete_probe_verifiability_scores.csv", index=False)
    write_memo(
        args.output_dir / "CONCRETE_PROBE_VERIFIABILITY_POLICY_MEMO.md",
        args,
        verif_features,
        selected,
        heldout_selected,
    )
    print(f"Wrote concrete probe-verifiability policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_verifiability_frame(target: pd.DataFrame, probe_features: pd.DataFrame, outputs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    tool = outputs[outputs["model_id"].astype(str).eq(TOOL_MODEL)][["query_id", "quality_score", "status"]].copy()
    tool["tool_available_label"] = tool["quality_score"].astype(float) > 0.0
    frame = target.drop(columns=[col for col in ["tool_available"] if col in target.columns]).merge(
        tool[["query_id", "tool_available_label"]],
        on="query_id",
        how="left",
    )
    frame["tool_available_label"] = frame["tool_available_label"].fillna(False).astype(bool)
    probe_cols: list[str] = []
    for col in probe_features.columns:
        if col == "query_id" or col in frame.columns:
            continue
        lower = col.lower()
        if "benchmark" in lower or lower in {"domain", "metric", "split"}:
            continue
        if lower.startswith("tool_") or lower.startswith("q32_choice_"):
            continue
        if pd.api.types.is_numeric_dtype(probe_features[col]):
            probe_cols.append(col)
    frame = frame.merge(probe_features[["query_id", *probe_cols]], on="query_id", how="left")
    frame = add_text_features(frame)
    features = verifiability_feature_columns(frame)
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame, features


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    text = out["query_text"].fillna("").astype(str)
    out["text_chars"] = text.str.len().astype(float)
    out["text_words"] = text.str.split().map(len).astype(float)
    out["text_digits"] = text.str.count(r"\d").astype(float)
    out["text_math_symbols"] = text.str.count(r"[=+\-*/^<>]|\\frac|\\sqrt|\\sum|\\int").astype(float)
    out["text_option_markers"] = text.str.count(r"(?m)^\s*[A-E][\).:]").astype(float)
    out["text_code_markers"] = text.str.count(r"\b(def|class|return|import|for|while|function|array|string|SQL)\b").astype(float)
    out["text_newlines"] = text.str.count(r"\n").astype(float)
    return out


def verifiability_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "query_id",
        "query_text",
        "split",
        "benchmark",
        "domain",
        "metric",
        "gold_answer",
        "best_local_action",
        "best_large_action",
        "local_quality",
        "large_quality",
        "local_utility",
        "large_utility",
        "delta_large",
        "need_large",
        "need_large_positive_gain",
        "local_normalized_cost",
        "large_normalized_cost",
        "local_cost_usd",
        "large_cost_usd",
        "local_latency_s",
        "large_latency_s",
        "tool_available_label",
    }
    cols: list[str] = []
    for col in frame.columns:
        lower = col.lower()
        if col in blocked or lower.startswith("tool_") or "train_prior" in lower:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return sorted(dict.fromkeys(cols))


def run_benchmark_heldout(
    exp204: Any,
    matrix: dict[str, Any],
    base: pd.DataFrame,
    candidate_features: pd.DataFrame,
    verif_frame: pd.DataFrame,
    verif_features: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    scores: list[pd.DataFrame] = []
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
        table, _, score_table = run_scenario(
            exp204,
            matrix,
            base,
            candidate_features,
            verif_frame,
            verif_features,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            scenario="benchmark_heldout",
            heldout=heldout,
            args=args,
        )
        rows.append(table)
        scores.append(score_table)
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(),
    )


def run_scenario(
    exp204: Any,
    matrix: dict[str, Any],
    base: pd.DataFrame,
    candidate_features: pd.DataFrame,
    verif_frame: pd.DataFrame,
    verif_features: list[str],
    *,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    scenario: str,
    heldout: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_series = base.set_index("query_id")["base_model"].astype(str)
    verif_scores, classifier_summary = fit_verifiability_scores(verif_frame, verif_features, train_ids, val_ids, test_ids, int(args.seed))
    local_choices = fit_local_ranker(exp204, candidate_features, matrix, train_ids, val_ids, test_ids, int(args.seed))
    rows: list[dict[str, Any]] = []
    choices: list[pd.DataFrame] = []

    references = reference_choice_frames(matrix, base_series, val_ids, test_ids)
    for frame, method, family, diagnostic in references:
        choices.append(frame)
        rows.extend(metric_rows(frame, method, family, args, scenario, heldout, diagnostic=diagnostic))

    policy_frames = concrete_policy_frames(matrix, base_series, local_choices, verif_scores, val_ids, test_ids)
    for frame, method, family in policy_frames:
        choices.append(frame)
        rows.extend(metric_rows(frame, method, family, args, scenario, heldout, diagnostic=False))

    table = pd.DataFrame(rows)
    choice_table = pd.concat(choices, ignore_index=True) if choices else pd.DataFrame()
    classifier_summary = classifier_summary.assign(scenario=scenario, heldout_benchmark=heldout)
    return table, choice_table, classifier_summary


def fit_verifiability_scores(
    frame: pd.DataFrame,
    features: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = frame.set_index("query_id")
    train = index.loc[train_ids]
    eval_ids = [*val_ids, *test_ids]
    eval_frame = index.loc[eval_ids].copy()
    y_train = train["tool_available_label"].astype(bool).to_numpy()
    specs = classifier_specs(seed)
    score_table = eval_frame[["split", "benchmark", "tool_available_label"]].copy()
    summary_rows: list[dict[str, Any]] = []
    for name, model in specs.items():
        model.fit(train[features], y_train)
        scores = model.predict_proba(eval_frame[features])[:, 1]
        score_table[name] = scores
        for split, group in score_table.groupby("split"):
            values = group[name].astype(float).to_numpy()
            labels = group["tool_available_label"].astype(bool).to_numpy()
            summary_rows.append(
                {
                    "classifier": name,
                    "split": str(split),
                    "n_queries": int(len(group)),
                    "positive_rate": float(labels.mean()) if len(labels) else math.nan,
                    "score_mean": float(values.mean()) if len(values) else math.nan,
                    "score_p90": float(np.quantile(values, 0.90)) if len(values) else math.nan,
                }
            )
    score_table = score_table.reset_index().rename(columns={"index": "query_id"})
    return score_table, pd.DataFrame(summary_rows)


def classifier_specs(seed: int) -> dict[str, Pipeline]:
    return {
        "extratrees_d3_leaf8": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=3,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gb_depth2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=80,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "logreg_c0.3": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed),
                ),
            ]
        ),
    }


def fit_local_ranker(
    exp204: Any,
    candidate_features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    seed: int,
) -> pd.DataFrame:
    cols = exp204.candidate_feature_columns(candidate_features)
    train = candidate_features[candidate_features["query_id"].astype(str).isin(train_ids)].copy()
    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4, random_state=seed, n_jobs=-1)
    model.fit(train[cols], train["target_utility"].astype(float))
    scored = candidate_features.copy()
    scored["predicted_local_utility"] = model.predict(scored[cols])
    rows: list[dict[str, Any]] = []
    for query_id, group in scored[scored["query_id"].astype(str).isin([*val_ids, *test_ids])].groupby("query_id", sort=False):
        group = group.sort_values("predicted_local_utility", ascending=False)
        best = group.iloc[0]
        second = float(group.iloc[1]["predicted_local_utility"]) if len(group) > 1 else 0.0
        rows.append(
            {
                "query_id": str(query_id),
                "split": str(matrix["meta"].loc[str(query_id), "split"]),
                "benchmark": str(matrix["meta"].loc[str(query_id), "benchmark"]),
                "selected_local_model": str(best["candidate_model"]),
                "selected_local_score": float(best["predicted_local_utility"]),
                "selected_local_margin": float(float(best["predicted_local_utility"]) - second),
                "selected_local_support": float(best.get("candidate_group_support", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def reference_choice_frames(
    matrix: dict[str, Any],
    base_series: pd.Series,
    val_ids: list[str],
    test_ids: list[str],
) -> list[tuple[pd.DataFrame, str, str, bool]]:
    frames: list[tuple[pd.DataFrame, str, str, bool]] = []
    for ids in [val_ids, test_ids]:
        base_selected = base_series.reindex(ids).dropna().astype(str)
        frames.append((choices_from_series(matrix, base_selected, "current_base"), "current_base", "current_base", False))
        full_oracle = matrix["utility"].loc[ids].idxmax(axis=1).astype(str)
        frames.append((choices_from_series(matrix, full_oracle, "full_oracle"), "full_oracle", "full_oracle_upper_bound", True))
        local_oracle = matrix["utility"].loc[ids, matrix["local_models"]].idxmax(axis=1).astype(str)
        frames.append((choices_from_series(matrix, local_oracle, "local_action_oracle"), "local_action_oracle", "local_action_oracle_upper_bound", True))
        repaired = {}
        for query_id, base_model in base_selected.items():
            candidates = [str(base_model), *matrix["local_models"]]
            repaired[str(query_id)] = max(candidates, key=lambda model: float(matrix["utility"].loc[str(query_id), model]))
        frames.append(
            (
                choices_from_series(matrix, pd.Series(repaired), "current_base_plus_all_locals_oracle"),
                "current_base_plus_all_locals_oracle",
                "current_base_plus_all_locals_oracle_upper_bound",
                True,
            )
        )
    return frames


def concrete_policy_frames(
    matrix: dict[str, Any],
    base_series: pd.Series,
    local_choices: pd.DataFrame,
    verif_scores: pd.DataFrame,
    val_ids: list[str],
    test_ids: list[str],
) -> list[tuple[pd.DataFrame, str, str]]:
    frames: list[tuple[pd.DataFrame, str, str]] = []
    local_index = local_choices.set_index("query_id")
    score_index = verif_scores.set_index("query_id")
    ids = [*val_ids, *test_ids]
    for classifier in CLASSIFIERS:
        thresholds = threshold_grid(score_index.reindex(val_ids)[classifier], VERIF_QUANTILES)
        for threshold in thresholds:
            for mode in ["any", "if_base_frontier"]:
                method = f"{classifier}_to_tool_{mode}_thr{threshold:.4f}"
                frame = build_gate_choices(
                    matrix,
                    base_series,
                    local_index,
                    score_index,
                    ids,
                    method,
                    classifier,
                    threshold,
                    mode,
                    action="tool",
                )
                frames.append((frame, method, "learned_verifiability_to_tool"))
    local_thresholds = threshold_grid(local_index.reindex(val_ids)["selected_local_score"], LOCAL_QUANTILES)
    for threshold in local_thresholds:
        method = f"local_ranker_if_base_frontier_score_thr{threshold:.4f}"
        frame = build_local_score_choices(matrix, base_series, local_index, ids, method, threshold, mode="if_base_frontier")
        frames.append((frame, method, "local_ranker_override"))
    for classifier in CLASSIFIERS:
        verif_thresholds = threshold_grid(score_index.reindex(val_ids)[classifier], [0.75, 0.85, 0.90, 0.95])
        for verif_threshold in verif_thresholds:
            for local_threshold in local_thresholds:
                method = f"verif_tool_else_local_ranker_{classifier}_v{verif_threshold:.4f}_l{local_threshold:.4f}"
                frame = build_combo_choices(
                    matrix,
                    base_series,
                    local_index,
                    score_index,
                    ids,
                    method,
                    classifier,
                    verif_threshold,
                    local_threshold,
                )
                frames.append((frame, method, "verif_tool_plus_local_ranker_override"))
    return frames


def build_gate_choices(
    matrix: dict[str, Any],
    base_series: pd.Series,
    local_index: pd.DataFrame,
    score_index: pd.DataFrame,
    ids: list[str],
    method: str,
    classifier: str,
    threshold: float,
    mode: str,
    *,
    action: str,
) -> pd.DataFrame:
    del local_index
    rows: list[dict[str, Any]] = []
    for query_id in ids:
        base_model = str(base_series.loc[query_id])
        active = float(score_index.loc[query_id, classifier]) >= float(threshold)
        if mode == "if_base_frontier":
            active = active and bool(matrix["frontier"].loc[query_id, base_model])
        selected = TOOL_MODEL if active and action == "tool" else base_model
        rows.append(
            choice_row(
                matrix,
                query_id,
                selected,
                method,
                base_model=base_model,
                verif_score=float(score_index.loc[query_id, classifier]),
                verif_threshold=float(threshold),
            )
        )
    return pd.DataFrame(rows)


def build_local_score_choices(
    matrix: dict[str, Any],
    base_series: pd.Series,
    local_index: pd.DataFrame,
    ids: list[str],
    method: str,
    threshold: float,
    *,
    mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id in ids:
        base_model = str(base_series.loc[query_id])
        local_model = str(local_index.loc[query_id, "selected_local_model"])
        active = float(local_index.loc[query_id, "selected_local_score"]) >= float(threshold)
        if mode == "if_base_frontier":
            active = active and bool(matrix["frontier"].loc[query_id, base_model])
        selected = local_model if active else base_model
        rows.append(
            choice_row(
                matrix,
                query_id,
                selected,
                method,
                base_model=base_model,
                local_score=float(local_index.loc[query_id, "selected_local_score"]),
                local_threshold=float(threshold),
            )
        )
    return pd.DataFrame(rows)


def build_combo_choices(
    matrix: dict[str, Any],
    base_series: pd.Series,
    local_index: pd.DataFrame,
    score_index: pd.DataFrame,
    ids: list[str],
    method: str,
    classifier: str,
    verif_threshold: float,
    local_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id in ids:
        base_model = str(base_series.loc[query_id])
        verif_active = float(score_index.loc[query_id, classifier]) >= float(verif_threshold)
        local_active = (
            bool(matrix["frontier"].loc[query_id, base_model])
            and float(local_index.loc[query_id, "selected_local_score"]) >= float(local_threshold)
        )
        selected = base_model
        if verif_active:
            selected = TOOL_MODEL
        elif local_active:
            selected = str(local_index.loc[query_id, "selected_local_model"])
        rows.append(
            choice_row(
                matrix,
                query_id,
                selected,
                method,
                base_model=base_model,
                verif_score=float(score_index.loc[query_id, classifier]),
                verif_threshold=float(verif_threshold),
                local_score=float(local_index.loc[query_id, "selected_local_score"]),
                local_threshold=float(local_threshold),
            )
        )
    return pd.DataFrame(rows)


def choices_from_series(matrix: dict[str, Any], selected: pd.Series, method: str) -> pd.DataFrame:
    return pd.DataFrame([choice_row(matrix, str(query_id), str(model), method) for query_id, model in selected.items()])


def choice_row(
    matrix: dict[str, Any],
    query_id: str,
    selected_model: str,
    method: str,
    *,
    base_model: str | None = None,
    verif_score: float = math.nan,
    verif_threshold: float = math.nan,
    local_score: float = math.nan,
    local_threshold: float = math.nan,
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
        "verif_score": float(verif_score),
        "verif_threshold": float(verif_threshold),
        "local_score": float(local_score),
        "local_threshold": float(local_threshold),
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
    keys = ["scenario", "heldout_benchmark", "family"]
    for _, group in table.groupby(keys, dropna=False):
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


def threshold_grid(values: pd.Series, quantiles: list[float]) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return [float("inf")]
    grid = {float(value) for value in np.quantile(clean.to_numpy(), quantiles)}
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


def write_memo(
    path: Path,
    args: argparse.Namespace,
    verif_features: list[str],
    selected: pd.DataFrame,
    heldout_selected: pd.DataFrame,
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
    heldout_test = heldout_selected[heldout_selected["split"].astype(str).eq("test")].copy() if not heldout_selected.empty else pd.DataFrame()
    heldout_summary = (
        heldout_test.groupby("family", as_index=False)
        .agg(
            mean_heldout_utility=("mean_utility", "mean"),
            mean_heldout_oracle_ratio=("oracle_utility_ratio", "mean"),
            mean_frontier_call_rate=("frontier_call_rate", "mean"),
        )
        .sort_values("mean_heldout_utility", ascending=False)
        if not heldout_test.empty
        else pd.DataFrame()
    )
    lines = [
        "# Concrete Probe-Verifiability Policy",
        "",
        "This cached experiment tests whether learned broad verifiability and a train-fit local-candidate ranker can repair",
        "the current concrete Broad100 base policy without benchmark-specific checkers.",
        "",
        "```text",
        "query + cheap local behavior -> learned verifiability/local-candidate scores -> concrete action",
        "```",
        "",
        "No provider calls, vLLM calls, local generation calls, or benchmark-specific verifier calls are made.",
        "For benchmark-heldout rows, the verifiability classifier and local ranker are fit without the held-out benchmark.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/210_concrete_probe_verifiability_policy.py",
        "```",
        "",
        "## Feature Policy",
        "",
        f"- Verifiability feature count: `{len(verif_features)}`",
        "- Blocked from learned verifiability features: benchmark ID, domain, metric, outcome quality/utility/cost,",
        "  train benchmark priors, and direct deterministic-tool output/status features.",
        "- Concrete policies can choose only cached concrete actions already present in the Broad100 action matrix.",
        "",
        "## Standard Selected Rows",
        "",
        markdown_table(selected[selected["scenario"].astype(str).eq("standard")][[c for c in cols if c in selected.columns]]),
        "",
        "## Benchmark-Heldout Transfer Summary",
        "",
        markdown_table(heldout_summary),
        "",
        "## Interpretation",
        "",
        "- `current_base_plus_all_locals_oracle` is a diagnostic ceiling, not a deployable method.",
        "- If learned gates do not beat `current_base`, then broad verifiability/local-candidate scores still do not solve",
        "  concrete action identity.",
        "- A valid positive result must improve the validation-selected held-out test rows, not just a test-only diagnostic row.",
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
