from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
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
K_VALUES = [4, 8, 16, 32, 64]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
OVERRIDE_MODES = ["any", "if_base_frontier", "if_base_local"]
THRESHOLD_VIEWS = ["score", "margin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-agnostic local-candidate ProbeCode bridge over cached Broad100 outcomes. "
            "This script makes no provider, vLLM, or benchmark-specific checker calls."
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
        default=Path("results/controlled/broad100_benchmark_agnostic_local_candidate_selector"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    probe_features = pd.read_csv(args.probe_features)
    matrix = build_matrix(outputs)
    base = load_base(args.base_query_choices, args.base_policy, matrix)
    candidate_features = build_candidate_features(outputs, probe_features, matrix)

    standard_all, standard_details = run_standard(candidate_features, probe_features, matrix, base, args)
    heldout = run_benchmark_heldout(candidate_features, probe_features, matrix, base, args)
    selected = select_rows(standard_all)
    selected_methods = set(selected["method"].dropna().astype(str))
    selected_details = standard_details[standard_details["method"].astype(str).isin(selected_methods)].copy()

    candidate_features.to_csv(args.output_dir / "table_local_candidate_features.csv", index=False)
    standard_all.to_csv(args.output_dir / "table_local_candidate_selector_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_local_candidate_selector_selected.csv", index=False)
    heldout.to_csv(args.output_dir / "table_local_candidate_selector_benchmark_heldout.csv", index=False)
    selected_details.to_csv(args.output_dir / "table_local_candidate_selector_query_choices.csv", index=False)
    write_figure(args.output_dir, standard_all)
    write_memo(args.output_dir / "LOCAL_CANDIDATE_SELECTOR_MEMO.md", args, candidate_features, selected, heldout)
    print(f"Wrote benchmark-agnostic local-candidate selector results to {args.output_dir}")


def build_matrix(outputs: pd.DataFrame) -> dict[str, Any]:
    model_ids = sorted(outputs["model_id"].astype(str).unique().tolist())
    utility = outputs.pivot(index="query_id", columns="model_id", values="utility").reindex(columns=model_ids)
    quality = outputs.pivot(index="query_id", columns="model_id", values="quality_score").reindex(columns=model_ids)
    cost = outputs.pivot(index="query_id", columns="model_id", values="normalized_remote_cost").reindex(columns=model_ids)
    frontier = outputs.pivot(index="query_id", columns="model_id", values="is_frontier").fillna(False).astype(bool)
    meta = outputs.drop_duplicates("query_id").set_index("query_id")[["query_text", "split", "benchmark", "domain", "metric"]]
    oracle_idx = utility.to_numpy().argmax(axis=1)
    oracle_utility = pd.Series(utility.to_numpy()[np.arange(len(utility)), oracle_idx], index=utility.index)
    oracle_quality = pd.Series(quality.to_numpy()[np.arange(len(quality)), oracle_idx], index=quality.index)
    return {
        "model_ids": model_ids,
        "local_models": [model for model in LOCAL_MODELS if model in model_ids],
        "utility": utility,
        "quality": quality,
        "cost": cost,
        "frontier": frontier,
        "meta": meta,
        "oracle_utility": oracle_utility,
        "oracle_quality": oracle_quality,
    }


def load_base(path: Path, policy: str, matrix: dict[str, Any]) -> pd.DataFrame:
    base = pd.read_csv(path)
    base = base[base["policy"].astype(str).eq(str(policy))].copy()
    if base.empty:
        raise RuntimeError(f"Base policy {policy!r} not found in {path}.")
    selected_col = "selected_model" if "selected_model" in base.columns else "selected_model_id"
    base = base[["query_id", selected_col]].rename(columns={selected_col: "base_model"})
    meta = matrix["meta"].reset_index()
    return meta.merge(base, on="query_id", how="inner")


def build_candidate_features(outputs: pd.DataFrame, probe_features: pd.DataFrame, matrix: dict[str, Any]) -> pd.DataFrame:
    local_outputs = outputs[outputs["model_id"].astype(str).isin(matrix["local_models"])].copy()
    probe_cols = feature_columns(probe_features)
    probe_index = probe_features.set_index("query_id")
    train_ids = matrix["meta"][matrix["meta"]["split"].astype(str).eq("train")].index.astype(str).tolist()
    priors = (
        local_outputs[local_outputs["query_id"].astype(str).isin(train_ids)]
        .groupby("model_id")
        .agg(candidate_train_prior_utility=("utility", "mean"), candidate_train_prior_quality=("quality_score", "mean"))
        .to_dict("index")
    )

    rows: list[dict[str, Any]] = []
    for query_id, group in local_outputs.groupby("query_id", sort=False):
        qid = str(query_id)
        answers = {str(row.model_id): normalize_answer(row.parsed_answer) for row in group.itertuples(index=False)}
        counts = Counter(answer for answer in answers.values() if answer)
        top_answer = counts.most_common(1)[0][0] if counts else ""
        top_count = counts[top_answer] if top_answer else 0
        second_count = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        valid_count = sum(1 for answer in answers.values() if answer)
        entropy = answer_entropy(counts)
        for item in group.itertuples(index=False):
            model_id = str(item.model_id)
            answer = answers.get(model_id, "")
            support = counts[answer] if answer else 0
            margin_reference = second_count if answer == top_answer else top_count
            row: dict[str, Any] = {
                "query_id": qid,
                "candidate_model": model_id,
                "split": str(item.split),
                "benchmark": str(item.benchmark),
                "domain": str(item.domain),
                "target_utility": float(item.utility),
                "target_quality": float(item.quality_score),
                "candidate_valid": float(bool(answer)),
                "candidate_answer_chars": float(len(answer)),
                "candidate_output_tokens": float(getattr(item, "output_tokens", 0.0) or 0.0),
                "candidate_latency_s": float(getattr(item, "latency_s", 0.0) or 0.0),
                "candidate_group_support": float(support),
                "candidate_group_frac": float(support / max(valid_count, 1)),
                "candidate_group_margin": float((support - margin_reference) / max(valid_count, 1)),
                "candidate_is_top_group": float(bool(answer and answer == top_answer)),
                "candidate_answer_entropy": float(entropy),
                "candidate_train_prior_utility": float(priors.get(model_id, {}).get("candidate_train_prior_utility", 0.0)),
                "candidate_train_prior_quality": float(priors.get(model_id, {}).get("candidate_train_prior_quality", 0.0)),
            }
            for local_model in matrix["local_models"]:
                key = safe_name(local_model)
                row[f"candidate_is_{key}"] = float(model_id == local_model)
                row[f"candidate_agrees_with_{key}"] = float(bool(answer and answers.get(local_model, "") == answer))
            if qid in probe_index.index:
                for column in probe_cols:
                    row[f"probe_{column}"] = probe_index.loc[qid, column]
            rows.append(row)
    frame = pd.DataFrame(rows)
    numeric = candidate_feature_columns(frame)
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame.sort_values(["split", "benchmark", "query_id", "candidate_model"]).reset_index(drop=True)


def run_standard(
    candidate_features: pd.DataFrame,
    probe_features: pd.DataFrame,
    matrix: dict[str, Any],
    base: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = split_ids(matrix, "train")
    val_ids = [query_id for query_id in split_ids(matrix, "val") if query_id in set(base["query_id"].astype(str))]
    test_ids = [query_id for query_id in split_ids(matrix, "test") if query_id in set(base["query_id"].astype(str))]
    rows, details = run_candidates(
        candidate_features,
        probe_features,
        matrix,
        base,
        train_ids,
        val_ids,
        test_ids,
        args,
        scenario="standard",
        heldout="",
    )
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def run_benchmark_heldout(
    candidate_features: pd.DataFrame,
    probe_features: pd.DataFrame,
    matrix: dict[str, Any],
    base: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    base_ids = set(base["query_id"].astype(str))
    for heldout in sorted(matrix["meta"]["benchmark"].astype(str).unique().tolist()):
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
        rows, _ = run_candidates(
            candidate_features,
            probe_features,
            matrix,
            base,
            train_ids,
            val_ids,
            test_ids,
            args,
            scenario="benchmark_heldout",
            heldout=heldout,
        )
        all_rows.append(select_rows(pd.DataFrame(rows)))
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def run_candidates(
    candidate_features: pd.DataFrame,
    probe_features: pd.DataFrame,
    matrix: dict[str, Any],
    base: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    *,
    scenario: str,
    heldout: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    base_series = base.set_index("query_id")["base_model"].astype(str)

    reference = reference_choice_frames(matrix, base_series, train_ids, val_ids, test_ids, scenario, heldout)
    for choices, method, family, diagnostic in reference:
        details.append(choices)
        rows.extend(metric_rows(choices, method, family, args, scenario, heldout, diagnostic=diagnostic))

    local_policy_frames = local_baseline_choice_frames(
        candidate_features,
        probe_features,
        matrix,
        train_ids,
        val_ids,
        test_ids,
        args,
        scenario,
        heldout,
    )
    for choices, method, family in local_policy_frames:
        details.append(choices)
        rows.extend(metric_rows(choices, method, family, args, scenario, heldout))

    ranker_frames = train_candidate_rankers(candidate_features, matrix, train_ids, val_ids, test_ids, args)
    for choices, method, family in ranker_frames:
        details.append(choices)
        rows.extend(metric_rows(choices, method, family, args, scenario, heldout))
        override_frames = override_base_with_local_selector(choices, base_series, matrix, val_ids, test_ids)
        for override_choices, override_method in override_frames:
            details.append(override_choices)
            rows.extend(
                metric_rows(
                    override_choices,
                    override_method,
                    "current_base_plus_local_selector",
                    args,
                    scenario,
                    heldout,
                )
            )
    return rows, details


def reference_choice_frames(
    matrix: dict[str, Any],
    base_series: pd.Series,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    scenario: str,
    heldout: str,
) -> list[tuple[pd.DataFrame, str, str, bool]]:
    del train_ids, scenario, heldout
    frames: list[tuple[pd.DataFrame, str, str, bool]] = []
    for split, ids in [("val", val_ids), ("test", test_ids)]:
        base_selected = base_series.reindex(ids).dropna().astype(str)
        frames.append((choices_from_series(matrix, base_selected, "current_base"), "current_base", "current_base", False))
        full_oracle = matrix["utility"].loc[ids].idxmax(axis=1).astype(str)
        frames.append((choices_from_series(matrix, full_oracle, "full_oracle"), "full_oracle", "full_oracle_upper_bound", True))
        local_oracle = matrix["utility"].loc[ids, matrix["local_models"]].idxmax(axis=1).astype(str)
        frames.append(
            (choices_from_series(matrix, local_oracle, "local_action_oracle"), "local_action_oracle", "local_action_oracle_upper_bound", True)
        )
        base_plus_local = {}
        for query_id, base_model in base_selected.items():
            candidates = [str(base_model), *matrix["local_models"]]
            base_plus_local[str(query_id)] = max(candidates, key=lambda model: float(matrix["utility"].loc[str(query_id), model]))
        frames.append(
            (
                choices_from_series(matrix, pd.Series(base_plus_local), "current_base_plus_all_locals_oracle"),
                "current_base_plus_all_locals_oracle",
                "current_base_plus_all_locals_oracle_upper_bound",
                True,
            )
        )
    return frames


def local_baseline_choice_frames(
    candidate_features: pd.DataFrame,
    probe_features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[tuple[pd.DataFrame, str, str]]:
    del scenario, heldout
    frames: list[tuple[pd.DataFrame, str, str]] = []
    global_model = best_model_for_ids(matrix, train_ids, matrix["local_models"])
    for split, ids in [("val", val_ids), ("test", test_ids)]:
        selected = pd.Series(global_model, index=ids)
        frames.append((choices_from_series(matrix, selected, "global_best_local"), "global_best_local", "global_best_local"))

    train_meta = matrix["meta"].loc[train_ids]
    mapping = {
        benchmark: best_model_for_ids(matrix, group.index.astype(str).tolist(), matrix["local_models"])
        for benchmark, group in train_meta.groupby("benchmark", sort=False)
    }
    for split, ids in [("val", val_ids), ("test", test_ids)]:
        selected = pd.Series(
            [mapping.get(str(matrix["meta"].loc[query_id, "benchmark"]), global_model) for query_id in ids],
            index=ids,
        )
        frames.append((choices_from_series(matrix, selected, "benchmark_lookup_local"), "benchmark_lookup_local", "benchmark_lookup_local"))

    frames.extend(text_local_utility_frames(probe_features, matrix, train_ids, val_ids, test_ids))
    frames.extend(probe_local_utility_frames(probe_features, matrix, train_ids, val_ids, test_ids))
    frames.extend(probe_state_local_frames(probe_features, matrix, train_ids, val_ids, test_ids, args))
    return frames


def text_local_utility_frames(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> list[tuple[pd.DataFrame, str, str]]:
    rows: list[tuple[pd.DataFrame, str, str]] = []
    index = features.set_index("query_id")
    train_texts = index.loc[train_ids, "query_text"].astype(str).tolist()
    y_train = matrix["utility"].loc[train_ids, matrix["local_models"]].to_numpy()
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(TfidfVectorizer(max_features=4096, min_df=2, ngram_range=(1, 2)), Ridge(alpha=float(alpha)))
        model.fit(train_texts, y_train)
        for split, ids in [("val", val_ids), ("test", test_ids)]:
            scores = model.predict(index.loc[ids, "query_text"].astype(str).tolist())
            selected = pd.Series([matrix["local_models"][int(idx)] for idx in np.asarray(scores).argmax(axis=1)], index=ids)
            method = f"text_only_local_utility_ridge_alpha{alpha:g}"
            rows.append((choices_from_series(matrix, selected, method), method, "text_only_local_utility_router"))
    return rows


def probe_local_utility_frames(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> list[tuple[pd.DataFrame, str, str]]:
    rows: list[tuple[pd.DataFrame, str, str]] = []
    cols = feature_columns(features)
    index = features.set_index("query_id")
    y_train = matrix["utility"].loc[train_ids, matrix["local_models"]].to_numpy()
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(index.loc[train_ids, cols], y_train)
        for split, ids in [("val", val_ids), ("test", test_ids)]:
            scores = model.predict(index.loc[ids, cols])
            selected = pd.Series([matrix["local_models"][int(idx)] for idx in np.asarray(scores).argmax(axis=1)], index=ids)
            method = f"probe_only_local_utility_ridge_alpha{alpha:g}"
            rows.append((choices_from_series(matrix, selected, method), method, "probe_only_local_utility_router"))
    return rows


def probe_state_local_frames(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
) -> list[tuple[pd.DataFrame, str, str]]:
    rows: list[tuple[pd.DataFrame, str, str]] = []
    cols = feature_columns(features)
    index = features.set_index("query_id")
    x_train = index.loc[train_ids, cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train))
    fallback = best_model_for_ids(matrix, train_ids, matrix["local_models"])
    for k in K_VALUES:
        kmeans = KMeans(n_clusters=int(k), random_state=int(args.seed), n_init=10)
        train_labels = kmeans.fit_predict(x_train)
        label_to_action = best_local_by_label(matrix, train_ids, train_labels)
        for split, ids in [("val", val_ids), ("test", test_ids)]:
            x_eval = scaler.transform(imputer.transform(index.loc[ids, cols]))
            labels = kmeans.predict(x_eval)
            selected = pd.Series([label_to_action.get(int(label), fallback) for label in labels], index=ids)
            method = f"probe_state_local_k{k}"
            rows.append((choices_from_series(matrix, selected, method), method, "probe_state_local_kmeans"))
    return rows


def train_candidate_rankers(
    candidate_features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
) -> list[tuple[pd.DataFrame, str, str]]:
    train = candidate_features[candidate_features["query_id"].astype(str).isin(train_ids)].copy()
    cols = candidate_feature_columns(candidate_features)
    models: list[tuple[str, Any]] = [
        (
            "candidate_ridge_alpha10",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        ),
        (
            "candidate_extra_trees_leaf4",
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4, random_state=int(args.seed), n_jobs=-1),
        ),
        (
            "candidate_random_forest_leaf4",
            RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=int(args.seed), n_jobs=-1),
        ),
        (
            "candidate_hgb_l2_0.1",
            HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, l2_regularization=0.1, random_state=int(args.seed)),
        ),
    ]
    frames: list[tuple[pd.DataFrame, str, str]] = []
    for method, model in models:
        model.fit(train[cols], train["target_utility"].astype(float))
        scored = candidate_features.copy()
        scored["predicted_local_utility"] = model.predict(scored[cols])
        pieces: list[pd.DataFrame] = []
        for split, ids in [("val", val_ids), ("test", test_ids)]:
            del split
            pieces.append(candidate_ranker_choices(scored, matrix, ids, method))
        choices = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        frames.append((choices, method, "local_candidate_ranker"))
    return frames


def candidate_ranker_choices(scored: pd.DataFrame, matrix: dict[str, Any], ids: list[str], method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = scored[scored["query_id"].astype(str).isin(ids)].copy()
    for query_id, group in frame.groupby("query_id", sort=False):
        group = group.sort_values("predicted_local_utility", ascending=False)
        best = group.iloc[0]
        second_score = float(group.iloc[1]["predicted_local_utility"]) if len(group) > 1 else 0.0
        rows.append(
            choice_row(
                matrix,
                str(query_id),
                str(best["candidate_model"]),
                method,
                selected_local_score=float(best["predicted_local_utility"]),
                selected_local_margin=float(float(best["predicted_local_utility"]) - second_score),
                selected_local_support=float(best["candidate_group_support"]),
            )
        )
    return pd.DataFrame(rows)


def override_base_with_local_selector(
    local_choices: pd.DataFrame,
    base_series: pd.Series,
    matrix: dict[str, Any],
    val_ids: list[str],
    test_ids: list[str],
) -> list[tuple[pd.DataFrame, str]]:
    frames: list[tuple[pd.DataFrame, str]] = []
    local_index = local_choices.set_index("query_id")
    val_scores = local_index.reindex(val_ids)["selected_local_score"].astype(float).dropna()
    val_margins = local_index.reindex(val_ids)["selected_local_margin"].astype(float).dropna()
    thresholds = {
        "score": threshold_grid(val_scores),
        "margin": threshold_grid(val_margins),
    }
    base_method = str(local_choices["method"].iloc[0])
    for view in THRESHOLD_VIEWS:
        for threshold in thresholds[view]:
            for mode in OVERRIDE_MODES:
                method = f"{base_method}_override_{mode}_{view}_thr{threshold:g}"
                rows: list[dict[str, Any]] = []
                for query_id in [*val_ids, *test_ids]:
                    if query_id not in local_index.index or query_id not in base_series.index:
                        continue
                    local_row = local_index.loc[query_id]
                    base_model = str(base_series.loc[query_id])
                    active = float(local_row[f"selected_local_{view}"]) >= float(threshold)
                    if mode == "if_base_frontier":
                        active = active and bool(matrix["frontier"].loc[query_id, base_model])
                    elif mode == "if_base_local":
                        active = active and not bool(matrix["frontier"].loc[query_id, base_model])
                    selected_model = str(local_row["selected_model"]) if active else base_model
                    rows.append(
                        choice_row(
                            matrix,
                            query_id,
                            selected_model,
                            method,
                            selected_local_score=float(local_row["selected_local_score"]),
                            selected_local_margin=float(local_row["selected_local_margin"]),
                            selected_local_support=float(local_row["selected_local_support"]),
                            changed=selected_model != base_model,
                        )
                    )
                frames.append((pd.DataFrame(rows), method))
    return frames


def choices_from_series(matrix: dict[str, Any], selected: pd.Series, method: str) -> pd.DataFrame:
    return pd.DataFrame([choice_row(matrix, str(query_id), str(model), method) for query_id, model in selected.items()])


def choice_row(
    matrix: dict[str, Any],
    query_id: str,
    selected_model: str,
    method: str,
    *,
    selected_local_score: float = math.nan,
    selected_local_margin: float = math.nan,
    selected_local_support: float = math.nan,
    changed: bool = False,
) -> dict[str, Any]:
    meta = matrix["meta"].loc[query_id]
    return {
        "query_id": query_id,
        "query_text": str(meta["query_text"]),
        "split": str(meta["split"]),
        "benchmark": str(meta["benchmark"]),
        "domain": str(meta["domain"]),
        "metric": str(meta["metric"]),
        "method": method,
        "selected_model": selected_model,
        "selected_quality": float(matrix["quality"].loc[query_id, selected_model]),
        "selected_utility": float(matrix["utility"].loc[query_id, selected_model]),
        "selected_normalized_cost": float(matrix["cost"].loc[query_id, selected_model]),
        "selected_frontier": bool(matrix["frontier"].loc[query_id, selected_model]),
        "selected_local_score": float(selected_local_score),
        "selected_local_margin": float(selected_local_margin),
        "selected_local_support": float(selected_local_support),
        "changed": bool(changed),
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
    diagnostic: bool = False,
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


def select_rows(table: pd.DataFrame) -> pd.DataFrame:
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
        val_row = best.copy()
        val_row["selection_rule"] = "val_best_mean_utility"
        rows.append(val_row)
        match = test[test["method"].astype(str).eq(str(best["method"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = "val_best_mean_utility_test"
            rows.append(test_row)
    if rows:
        selected = pd.DataFrame(rows)
    else:
        selected = pd.DataFrame()

    standard_test = table[table["scenario"].astype(str).eq("standard") & table["split"].astype(str).eq("test")].copy()
    top_rows = []
    for _, row in standard_test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(12).iterrows():
        item = row.copy()
        item["selection_rule"] = "top_standard_test_diagnostic"
        top_rows.append(item)
    if top_rows:
        selected = pd.concat([selected, pd.DataFrame(top_rows)], ignore_index=True)
    return selected.drop_duplicates(["scenario", "heldout_benchmark", "family", "method", "split", "selection_rule"])


def best_model_for_ids(matrix: dict[str, Any], ids: list[str], models: list[str]) -> str:
    return str(matrix["utility"].loc[ids, models].mean(axis=0).idxmax())


def best_local_by_label(matrix: dict[str, Any], ids: list[str], labels: np.ndarray) -> dict[int, str]:
    frame = matrix["utility"].loc[ids, matrix["local_models"]].copy()
    frame["_label"] = labels
    out: dict[int, str] = {}
    for label, group in frame.groupby("_label"):
        out[int(label)] = str(group.drop(columns=["_label"]).mean(axis=0).idxmax())
    return out


def split_ids(matrix: dict[str, Any], split: str) -> list[str]:
    return matrix["meta"][matrix["meta"]["split"].astype(str).eq(split)].index.astype(str).tolist()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {"query_id", "query_text", "split", "benchmark", "domain", "metric"}
    return [column for column in frame.columns if column not in blocked and pd.api.types.is_numeric_dtype(frame[column])]


def candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {"query_id", "candidate_model", "split", "benchmark", "domain", "target_utility", "target_quality"}
    return [column for column in frame.columns if column not in blocked and pd.api.types.is_numeric_dtype(frame[column])]


def threshold_grid(values: pd.Series) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return [float("inf")]
    quantiles = np.quantile(clean.to_numpy(), np.linspace(0.0, 1.0, 11))
    grid = sorted({float(value) for value in quantiles})
    grid.append(float(clean.max()) + 1e-9)
    return grid


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


def safe_name(model_id: str) -> str:
    return (
        str(model_id)
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("deterministic_math_tool", "tool")
    )


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_figure(output_dir: Path, table: pd.DataFrame) -> None:
    test = table[table["scenario"].eq("standard") & table["split"].eq("test")].copy()
    if test.empty:
        return
    plot = test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(18)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#5b7480")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Benchmark-Agnostic Local Candidate Selector")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_local_candidate_selector_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    candidate_features: pd.DataFrame,
    selected: pd.DataFrame,
    heldout: pd.DataFrame,
) -> None:
    cols = [
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
    heldout_test = heldout[heldout["split"].astype(str).eq("test")].copy() if not heldout.empty else pd.DataFrame()
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
        "# Benchmark-Agnostic Local Candidate Selector",
        "",
        "This cached Broad100 experiment tests whether broad local behavior can choose a concrete cheap/local action,",
        "then optionally override the current base policy. It does not use benchmark-specific checkers and makes no",
        "provider or vLLM calls.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/204_benchmark_agnostic_local_candidate_selector.py",
        "```",
        "",
        "## Data And Models",
        "",
        f"- Outcome table: `{args.outputs}`",
        f"- Probe-state feature table: `{args.probe_features}`",
        f"- Current base choices: `{args.base_query_choices}`",
        "- Candidate local actions: deterministic math tool, Qwen3 4B, Qwen3 8B, Qwen3 14B AWQ, Qwen3 32B AWQ,",
        "  and Qwen3 32B self-consistency n=3.",
        "- Expensive/frontier actions remain in the oracle/base comparison: `gpt-5.5`, `gemini-3.5-flash`, and",
        "  `gemini-3.5-flash-strong-solve`.",
        f"- Candidate rows: `{len(candidate_features)}`",
        f"- Numeric candidate/probe features: `{len(candidate_feature_columns(candidate_features))}`",
        "",
        "## Standard Split Selected Rows",
        "",
        markdown_table(selected[selected["scenario"].astype(str).eq("standard")][cols]) if not selected.empty else "No rows.",
        "",
        "## Benchmark-Heldout Transfer Summary",
        "",
        markdown_table(heldout_summary) if not heldout_summary.empty else "No heldout rows.",
        "",
        "## Interpretation",
        "",
        "- `current_base_plus_all_locals_oracle` is a diagnostic ceiling: it chooses post hoc between the current base action",
        "  and all cached local candidates using true held-out utility.",
        "- Main learned rows exclude benchmark ID and use train-only candidate labels. Thresholds are selected on validation.",
        "- Benchmark-heldout rows train on other benchmark families, select on their validation rows, and report on the",
        "  held-out benchmark test rows.",
        "- A positive result would improve over text-only local routing and benchmark lookup while preserving transfer.",
        "  If learned concrete local selection underperforms the oracle ceiling, the bottleneck is observability of local",
        "  candidate identity rather than lack of cheap local alternatives.",
        "",
        "## Artifacts",
        "",
        f"- Candidate features: `{path.parent / 'table_local_candidate_features.csv'}`",
        f"- All policies: `{path.parent / 'table_local_candidate_selector_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_local_candidate_selector_selected.csv'}`",
        f"- Benchmark-heldout transfer: `{path.parent / 'table_local_candidate_selector_benchmark_heldout.csv'}`",
        f"- Query choices: `{path.parent / 'table_local_candidate_selector_query_choices.csv'}`",
        f"- Figure: `{path.parent / 'fig_local_candidate_selector_utility.pdf'}`",
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
