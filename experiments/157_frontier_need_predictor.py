from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from routecode.controlled.live_stage0 import normalize_answer


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
TOOL_MODEL_ID = "deterministic_math_tool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a deployable frontier-need gate over cached broad100 outputs.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/table_vllm_self_consistency_probe.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_frontier_need_predictor"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--max-frontier-rate", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")

    outputs = load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    probe = load_probe(args.probe_table)
    table = run_frontier_need_predictors(
        package,
        outputs,
        probe,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
        max_frontier_rate=float(args.max_frontier_rate),
    )
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_frontier_need_predictor_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_frontier_need_predictor_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "FRONTIER_NEED_PREDICTOR_MEMO.md", args, table, selected)
    print(f"Wrote frontier-need predictor results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    for column in ["quality_score", "cost_total_usd", "normalized_remote_cost", "latency_s"]:
        if column not in outputs:
            outputs[column] = 0.0
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    if "utility" not in outputs:
        outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    outputs["utility"] = pd.to_numeric(outputs["utility"], errors="coerce").fillna(0.0)
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["split"] = outputs["split"].astype(str)
    for column in ["is_frontier", "is_local"]:
        if column not in outputs:
            outputs[column] = False
        outputs[column] = outputs[column].astype(bool)
    return outputs.drop_duplicates(["query_id", "model_id"], keep="last")


def load_probe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["query_id", "split"])
    probe = pd.read_csv(path).copy()
    probe["query_id"] = probe["query_id"].astype(str)
    probe["split"] = probe["split"].astype(str)
    for column in [
        "n_samples",
        "valid_count",
        "top_vote_count",
        "vote_frac",
        "vote_margin",
        "vote_entropy",
        "latency_s",
        "input_tokens",
        "output_tokens",
    ]:
        if column in probe:
            probe[column] = pd.to_numeric(probe[column], errors="coerce").fillna(0.0)
    return probe.drop_duplicates("query_id", keep="last")


def run_frontier_need_predictors(
    package,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    lambda_cost: float,
    max_features: int,
    max_frontier_rate: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frontier_ids = frontier_model_ids(outputs)
    local_outputs = outputs[~outputs["model_id"].isin(frontier_ids)].copy()
    base_builders = {
        "local_observable_state": lambda split: package.observable_local_state_selection(local_outputs, split=split),
        "local_tool_profile_v4": lambda split: package.profile_v4_selection_for_split(
            local_outputs, split=split, exclude_models=set(frontier_ids)
        ),
    }

    for base_name, builder in base_builders.items():
        base = {split: normalize_selection(builder(split)) for split in ["train", "val", "test"]}
        for split in ["val", "test"]:
            rows.append(
                evaluate_selection(
                    package,
                    outputs,
                    base[split],
                    split=split,
                    method=base_name,
                    family="local_reference",
                    lambda_cost=lambda_cost,
                )
            )
            rows.append(
                evaluate_selection(
                    package,
                    outputs,
                    oracle_between_local_and_frontier(outputs, base[split], frontier_ids),
                    split=split,
                    method=f"{base_name}_oracle_between_local_and_frontier",
                    family="diagnostic_oracle",
                    lambda_cost=lambda_cost,
                )
            )

        train = build_feature_frame(outputs, probe, base["train"], frontier_ids, split="train")
        val = build_feature_frame(outputs, probe, base["val"], frontier_ids, split="val")
        test = build_feature_frame(outputs, probe, base["test"], frontier_ids, split="test")
        if train.empty or val.empty or test.empty:
            continue
        rows.extend(
            run_ridge_utility_models(
                package,
                outputs,
                train,
                val,
                test,
                base,
                frontier_ids,
                base_name=base_name,
                lambda_cost=lambda_cost,
                max_features=max_features,
                max_frontier_rate=max_frontier_rate,
            )
        )
        rows.extend(
            run_ridge_gain_models(
                package,
                outputs,
                train,
                val,
                test,
                base,
                frontier_ids,
                base_name=base_name,
                lambda_cost=lambda_cost,
                max_features=max_features,
                max_frontier_rate=max_frontier_rate,
            )
        )
        rows.extend(
            run_logistic_need_models(
                package,
                outputs,
                train,
                val,
                test,
                base,
                frontier_ids,
                base_name=base_name,
                lambda_cost=lambda_cost,
                max_features=max_features,
                max_frontier_rate=max_frontier_rate,
            )
        )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def frontier_model_ids(outputs: pd.DataFrame) -> list[str]:
    available = set(outputs["model_id"].astype(str))
    preferred = ["gemini-3.5-flash", "gpt-5.5", STRONG_MODEL_ID]
    ids = [model for model in preferred if model in available]
    extra = sorted(
        str(model_id)
        for model_id in outputs.loc[outputs["is_frontier"].astype(bool), "model_id"].unique()
        if str(model_id) not in ids
    )
    return ids + extra


def run_ridge_utility_models(
    package,
    outputs: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    base: dict[str, pd.Series],
    frontier_ids: list[str],
    *,
    base_name: str,
    lambda_cost: float,
    max_features: int,
    max_frontier_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_cols = ["utility_local", *[target_col(model_id) for model_id in frontier_ids]]
    for feature_view in ["metadata_numeric", "metadata_numeric_text"]:
        x_train, x_val, x_test = featurize(train, val, test, feature_view=feature_view, max_features=max_features)
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            val_scores = pd.DataFrame(index=val["query_id"].astype(str))
            test_scores = pd.DataFrame(index=test["query_id"].astype(str))
            for column in target_cols:
                model = Ridge(alpha=float(alpha), solver="lsqr")
                model.fit(x_train, train[column].to_numpy(dtype=float))
                val_scores[column] = np.asarray(model.predict(x_val), dtype=float)
                test_scores[column] = np.asarray(model.predict(x_test), dtype=float)
            prefix = f"{base_name}_frontier_utility_ridge_{feature_view}_alpha{alpha:g}"
            rows.extend(
                selected_val_and_test_rows(
                    package,
                    outputs,
                    base,
                    val_scores,
                    test_scores,
                    frontier_ids,
                    method_prefix=prefix,
                    family="frontier_utility_ridge",
                    lambda_cost=lambda_cost,
                    base_name=base_name,
                    feature_view=feature_view,
                    alpha=alpha,
                    max_frontier_rate=max_frontier_rate,
                    score_mode="utility",
                )
            )
    return rows


def run_ridge_gain_models(
    package,
    outputs: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    base: dict[str, pd.Series],
    frontier_ids: list[str],
    *,
    base_name: str,
    lambda_cost: float,
    max_features: int,
    max_frontier_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frontier_lookup = frontier_train_lookup(train, frontier_ids)
    for feature_view in ["metadata_numeric", "metadata_numeric_text"]:
        x_train, x_val, x_test = featurize(train, val, test, feature_view=feature_view, max_features=max_features)
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(x_train, train["frontier_gain"].to_numpy(dtype=float))
            val_scores = pd.DataFrame({"gain": np.asarray(model.predict(x_val), dtype=float)}, index=val["query_id"].astype(str))
            test_scores = pd.DataFrame({"gain": np.asarray(model.predict(x_test), dtype=float)}, index=test["query_id"].astype(str))
            prefix = f"{base_name}_frontier_gain_ridge_{feature_view}_alpha{alpha:g}"
            rows.extend(
                selected_val_and_test_rows(
                    package,
                    outputs,
                    base,
                    val_scores,
                    test_scores,
                    frontier_ids,
                    method_prefix=prefix,
                    family="frontier_gain_ridge",
                    lambda_cost=lambda_cost,
                    base_name=base_name,
                    feature_view=feature_view,
                    alpha=alpha,
                    max_frontier_rate=max_frontier_rate,
                    score_mode="gain",
                    frontier_lookup=frontier_lookup,
                )
            )
    return rows


def run_logistic_need_models(
    package,
    outputs: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    base: dict[str, pd.Series],
    frontier_ids: list[str],
    *,
    base_name: str,
    lambda_cost: float,
    max_features: int,
    max_frontier_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_train = (train["frontier_gain"].to_numpy(dtype=float) > 0.0).astype(int)
    if len(set(y_train.tolist())) < 2:
        return rows
    frontier_lookup = frontier_train_lookup(train, frontier_ids)
    for feature_view in ["metadata_numeric"]:
        x_train, x_val, x_test = featurize(train, val, test, feature_view=feature_view, max_features=max_features)
        for c_value in [0.1, 1.0, 10.0]:
            model = LogisticRegression(C=float(c_value), class_weight="balanced", max_iter=1000, solver="liblinear")
            model.fit(x_train, y_train)
            val_scores = pd.DataFrame({"gain": model.predict_proba(x_val)[:, 1]}, index=val["query_id"].astype(str))
            test_scores = pd.DataFrame({"gain": model.predict_proba(x_test)[:, 1]}, index=test["query_id"].astype(str))
            prefix = f"{base_name}_frontier_need_logistic_{feature_view}_C{c_value:g}"
            rows.extend(
                selected_val_and_test_rows(
                    package,
                    outputs,
                    base,
                    val_scores,
                    test_scores,
                    frontier_ids,
                    method_prefix=prefix,
                    family="frontier_need_logistic",
                    lambda_cost=lambda_cost,
                    base_name=base_name,
                    feature_view=feature_view,
                    alpha=np.nan,
                    max_frontier_rate=max_frontier_rate,
                    score_mode="gain",
                    frontier_lookup=frontier_lookup,
                    classifier_c=c_value,
                )
            )
    return rows


def selected_val_and_test_rows(
    package,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    val_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    frontier_ids: list[str],
    *,
    method_prefix: str,
    family: str,
    lambda_cost: float,
    max_frontier_rate: float,
    score_mode: str,
    frontier_lookup: dict[str, str] | None = None,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = []
    for threshold in candidate_thresholds(frontier_delta(val_scores, frontier_ids, score_mode=score_mode)):
        for cap in [0.25, 0.35, max_frontier_rate, 0.50, 1.00]:
            cap = float(cap)
            val_selected = scores_to_selection(
                base["val"],
                val_scores,
                frontier_ids,
                threshold=threshold,
                cap=cap,
                score_mode=score_mode,
                query_info=query_info(outputs, "val"),
                frontier_lookup=frontier_lookup,
            )
            method = f"{method_prefix}_thr{threshold:.4f}_cap{cap:.2f}"
            row = evaluate_selection(
                package,
                outputs,
                val_selected,
                split="val",
                method=method,
                family=family,
                lambda_cost=lambda_cost,
            )
            row.update(extra)
            row.update({"threshold": float(threshold), "frontier_cap": cap, "score_mode": score_mode})
            candidates.append(row)
    candidates = sorted(candidates, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)
    if not candidates:
        return rows
    best = candidates[0]
    rows.append(best)
    test_selected = scores_to_selection(
        base["test"],
        test_scores,
        frontier_ids,
        threshold=float(best["threshold"]),
        cap=float(best["frontier_cap"]),
        score_mode=score_mode,
        query_info=query_info(outputs, "test"),
        frontier_lookup=frontier_lookup,
    )
    test_row = evaluate_selection(
        package,
        outputs,
        test_selected,
        split="test",
        method=str(best["method"]),
        family=family,
        lambda_cost=lambda_cost,
    )
    test_row.update(extra)
    test_row.update({"threshold": float(best["threshold"]), "frontier_cap": float(best["frontier_cap"]), "score_mode": score_mode})
    rows.append(test_row)
    return rows


def candidate_thresholds(delta: pd.Series) -> list[float]:
    values = np.asarray(delta.dropna(), dtype=float)
    if values.size == 0:
        return [0.0]
    quantiles = np.quantile(values, [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    fixed = np.asarray([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90])
    return sorted({round(float(value), 6) for value in np.concatenate([quantiles, fixed])})


def scores_to_selection(
    base: pd.Series,
    scores: pd.DataFrame,
    frontier_ids: list[str],
    *,
    threshold: float,
    cap: float,
    score_mode: str,
    query_info: pd.DataFrame,
    frontier_lookup: dict[str, str] | None,
) -> pd.Series:
    selected = normalize_selection(base)
    delta = frontier_delta(scores, frontier_ids, score_mode=score_mode)
    eligible = delta[delta > float(threshold)].sort_values(ascending=False)
    if cap < 1.0:
        eligible = eligible.head(max(1, int(np.floor(float(cap) * len(selected)))))
    for query_id in eligible.index.astype(str):
        if score_mode == "gain":
            benchmark = str(query_info.loc[query_id, "benchmark"]) if query_id in query_info.index else ""
            selected.loc[query_id] = str((frontier_lookup or {}).get(benchmark, frontier_ids[0]))
        else:
            selected.loc[query_id] = best_predicted_frontier(scores.loc[query_id], frontier_ids)
    return selected


def frontier_delta(scores: pd.DataFrame, frontier_ids: list[str], *, score_mode: str) -> pd.Series:
    if score_mode == "gain":
        return scores["gain"].astype(float)
    frontier_cols = [target_col(model_id) for model_id in frontier_ids if target_col(model_id) in scores]
    frontier_best = scores[frontier_cols].max(axis=1)
    return frontier_best.astype(float) - scores["utility_local"].astype(float)


def best_predicted_frontier(row: pd.Series, frontier_ids: list[str]) -> str:
    best_model = frontier_ids[0]
    best_score = -float("inf")
    for model_id in frontier_ids:
        score = float(row.get(target_col(model_id), -float("inf")))
        if score > best_score:
            best_model = model_id
            best_score = score
    return best_model


def build_feature_frame(
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    base_selection: pd.Series,
    frontier_ids: list[str],
    *,
    split: str,
) -> pd.DataFrame:
    queries = query_info(outputs, split)
    by_query = outputs.set_index(["query_id", "model_id"])
    probe_by_query = probe[probe["split"].eq(split)].set_index("query_id") if not probe.empty else pd.DataFrame()
    local_models = [model for model in ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local", DEFAULT_SELF_MODEL_ID] if model in set(outputs["model_id"])]
    rows: list[dict[str, Any]] = []
    for query_id, local_model in base_selection.items():
        query_id = str(query_id)
        local_model = str(local_model)
        if query_id not in queries.index or (query_id, local_model) not in by_query.index:
            continue
        info = queries.loc[query_id]
        local_row = by_query.loc[(query_id, local_model)]
        probe_row = probe_by_query.loc[query_id] if not probe_by_query.empty and query_id in probe_by_query.index else pd.Series(dtype=object)
        local_answer_norm = normalize_answer(str(local_row.get("parsed_answer", "")))
        majority_norm = normalize_answer(str(probe_row.get("majority_answer_norm", "")))
        local_agree = local_answer_agreement_count(outputs, query_id, majority_norm, local_models)
        row: dict[str, Any] = {
            "query_id": query_id,
            "query_text": str(info.get("query_text", "")),
            "benchmark": str(info.get("benchmark", "")),
            "domain": str(info.get("domain", "")),
            "metric": str(info.get("metric", "")),
            "local_model_id": local_model,
            "local_provider": str(local_row.get("provider", "")),
            "local_answer_norm": local_answer_norm,
            "majority_answer_norm": majority_norm,
            "local_equals_majority": bool(local_answer_norm and majority_norm and local_answer_norm == majority_norm),
            "n_samples": float(probe_row.get("n_samples", 0.0) or 0.0),
            "valid_count": float(probe_row.get("valid_count", 0.0) or 0.0),
            "top_vote_count": float(probe_row.get("top_vote_count", 0.0) or 0.0),
            "vote_frac": float(probe_row.get("vote_frac", 0.0) or 0.0),
            "vote_margin": float(probe_row.get("vote_margin", 0.0) or 0.0),
            "vote_entropy": float(probe_row.get("vote_entropy", 0.0) or 0.0),
            "local_agree_with_majority_count": float(local_agree),
            "local_answer_len": float(len(local_answer_norm)),
            "majority_answer_len": float(len(majority_norm)),
            "utility_local": float(local_row["utility"]),
        }
        best_frontier_utility = -1.0
        best_frontier_model = frontier_ids[0] if frontier_ids else ""
        for model_id in frontier_ids:
            utility = -1.0
            if (query_id, model_id) in by_query.index:
                utility = float(by_query.loc[(query_id, model_id)]["utility"])
            row[target_col(model_id)] = utility
            if utility > best_frontier_utility:
                best_frontier_utility = utility
                best_frontier_model = model_id
        row["best_frontier_utility"] = best_frontier_utility
        row["best_frontier_model"] = best_frontier_model
        row["frontier_gain"] = best_frontier_utility - float(row["utility_local"])
        row["feature_text"] = feature_text(row)
        rows.append(row)
    return pd.DataFrame(rows)


def local_answer_agreement_count(outputs: pd.DataFrame, query_id: str, majority_norm: str, local_models: list[str]) -> int:
    if not majority_norm:
        return 0
    by_query = outputs.set_index(["query_id", "model_id"])
    count = 0
    for model_id in local_models:
        if (query_id, model_id) not in by_query.index:
            continue
        answer = normalize_answer(str(by_query.loc[(query_id, model_id)].get("parsed_answer", "")))
        count += int(bool(answer) and answer == majority_norm)
    return count


def feature_text(row: dict[str, Any]) -> str:
    pieces = [
        str(row.get("benchmark", "")),
        str(row.get("domain", "")),
        str(row.get("metric", "")),
        str(row.get("local_model_id", "")),
        f"local_answer={row.get('local_answer_norm', '')}",
        f"self_majority={row.get('majority_answer_norm', '')}",
        "local_equals_self" if row.get("local_equals_majority") else "local_differs_self",
        str(row.get("query_text", "")),
    ]
    return " ".join(pieces)


def featurize(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_view: str,
    max_features: int,
):
    numeric_columns = [
        "n_samples",
        "valid_count",
        "top_vote_count",
        "vote_frac",
        "vote_margin",
        "vote_entropy",
        "local_agree_with_majority_count",
        "local_answer_len",
        "majority_answer_len",
    ]
    categorical_columns = [
        "benchmark",
        "domain",
        "metric",
        "local_model_id",
        "local_provider",
        "local_equals_majority",
    ]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(frame_to_dicts(train, numeric_columns, categorical_columns))
    x_val = vectorizer.transform(frame_to_dicts(val, numeric_columns, categorical_columns))
    x_test = vectorizer.transform(frame_to_dicts(test, numeric_columns, categorical_columns))
    if feature_view == "metadata_numeric":
        return x_train, x_val, x_test
    if feature_view != "metadata_numeric_text":
        raise ValueError(feature_view)
    text = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    train_text = text.fit_transform(train["feature_text"].fillna("").astype(str))
    val_text = text.transform(val["feature_text"].fillna("").astype(str))
    test_text = text.transform(test["feature_text"].fillna("").astype(str))
    return hstack([x_train, train_text]), hstack([x_val, val_text]), hstack([x_test, test_text])


def frame_to_dicts(frame: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        features: dict[str, Any] = {}
        for column in numeric_columns:
            features[column] = float(row.get(column, 0.0) or 0.0)
        for column in categorical_columns:
            features[f"{column}={row.get(column, '')}"] = 1.0
        rows.append(features)
    return rows


def frontier_train_lookup(train: pd.DataFrame, frontier_ids: list[str]) -> dict[str, str]:
    rows = []
    for benchmark, group in train.groupby("benchmark"):
        for model_id in frontier_ids:
            column = target_col(model_id)
            if column not in group:
                continue
            rows.append({"benchmark": benchmark, "model_id": model_id, "mean_utility": float(group[column].mean())})
    if not rows:
        return {}
    table = pd.DataFrame(rows).sort_values(["benchmark", "mean_utility"], ascending=[True, False])
    return table.drop_duplicates("benchmark").set_index("benchmark")["model_id"].astype(str).to_dict()


def oracle_between_local_and_frontier(outputs: pd.DataFrame, base: pd.Series, frontier_ids: list[str]) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    selected = normalize_selection(base)
    for query_id, local_model in base.items():
        best_model = str(local_model)
        best_utility = -float("inf")
        best_quality = -float("inf")
        for model_id in [str(local_model), *frontier_ids]:
            key = (str(query_id), str(model_id))
            if key not in by_query.index:
                continue
            row = by_query.loc[key]
            utility = float(row["utility"])
            quality = float(row["quality_score"])
            if utility > best_utility or (abs(utility - best_utility) <= 1e-12 and quality > best_quality):
                best_model = str(model_id)
                best_utility = utility
                best_quality = quality
        selected.loc[str(query_id)] = best_model
    return selected


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    method: str,
    family: str,
    lambda_cost: float,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["family"] = family
    row["frontier_call_rate"] = float(selected_rows["is_frontier"].astype(bool).mean()) if not selected_rows.empty else 0.0
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean()) if not selected_rows.empty else 0.0
    row["self_action_rate"] = float(selected_rows["model_id"].eq(DEFAULT_SELF_MODEL_ID).mean()) if not selected_rows.empty else 0.0
    return row


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in table.groupby("family"):
        if family == "diagnostic_oracle":
            continue
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = group[group["split"].eq("test") & group["method"].eq(method)]
        if not test.empty:
            rows.append(test.head(1).assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(16)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def query_info(outputs: pd.DataFrame, split: str) -> pd.DataFrame:
    return outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")


def target_col(model_id: str) -> str:
    return "utility_" + str(model_id).replace("-", "_").replace(".", "_")


def write_figure(output_dir: Path, table: pd.DataFrame) -> None:
    if table.empty:
        return
    test = table[table["split"].eq("test")].copy()
    if test.empty:
        return
    test = test.sort_values("mean_utility", ascending=False).head(18)
    labels = [str(value)[:52] for value in test["method"]]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * len(test))))
    ax.barh(labels[::-1], test["mean_utility"].to_numpy(dtype=float)[::-1])
    ax.set_xlabel("Held-out mean utility")
    ax.set_title("Frontier-Need Predictor")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_frontier_need_predictor_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Frontier-Need Predictor",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        "",
        "This run makes no model/provider API calls. It trains on train, selects thresholds/caps on validation, and reports held-out test rows.",
        "",
        "## Validation-Selected Rows",
        "",
    ]
    if selected.empty:
        lines.append("No rows were selected.")
    else:
        lines.append(markdown_table(selected))
    lines.extend(["", "## Best Held-Out Diagnostics", ""])
    test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    if test.empty:
        lines.append("No test rows were produced.")
    else:
        lines.append(markdown_table(test))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The local side is a deployable local/self/tool policy, not a local utility oracle.",
            "- Frontier candidates include cached GPT/Gemini rows already present in the matrix.",
            "- Rows selected only because they are best on held-out test are diagnostic, not achieved methods.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                text = str(value).replace("\n", " ")
                values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
