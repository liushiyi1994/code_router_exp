from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


LOCAL_MODELS = ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observable local-feature routers for broad100.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_observable_feature_router"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-text-features", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--include-text-view",
        action="store_true",
        help="Also include query/local-answer TF-IDF features. Defaults off for quick observable-state probing.",
    )
    parser.add_argument(
        "--include-classifiers",
        action="store_true",
        help="Also run oracle-action classifiers. Defaults off because they are slower and usually less useful.",
    )
    parser.add_argument(
        "--include-state-tables",
        action="store_true",
        help="Also run observable state-action table variants. Defaults off for the focused first pass.",
    )
    parser.add_argument(
        "--include-tree-learners",
        action="store_true",
        help="Also run slower tree/forest learners. Defaults off for a quick probe pass.",
    )
    parser.add_argument(
        "--include-existing-reference",
        action="store_true",
        help="Also recompute slower imported reference policies. Defaults off; prior results already cover them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    log_stage(started, "loaded package")
    outputs = normalize_split_columns(package.load_outputs(args.outputs, lambda_cost=args.lambda_cost))
    log_stage(started, "loaded outputs")
    feature_store = fit_feature_store(package, outputs, max_text_features=int(args.max_text_features))
    log_stage(started, "fit observable feature store")

    reference = run_reference_policies(
        package,
        outputs,
        lambda_cost=float(args.lambda_cost),
        include_existing_reference=bool(args.include_existing_reference),
    )
    log_stage(started, "ran reference policies")
    utility = run_utility_regressors(
        package,
        outputs,
        feature_store,
        lambda_cost=float(args.lambda_cost),
        seed=int(args.seed),
        include_text_view=bool(args.include_text_view),
        include_tree_learners=bool(args.include_tree_learners),
    )
    log_stage(started, "ran utility regressors")
    classifiers = (
        run_oracle_classifiers(
            package,
            outputs,
            feature_store,
            lambda_cost=float(args.lambda_cost),
            seed=int(args.seed),
            include_text_view=bool(args.include_text_view),
            include_tree_learners=bool(args.include_tree_learners),
        )
        if args.include_classifiers
        else pd.DataFrame()
    )
    log_stage(started, "ran oracle classifiers")
    benchmark_state = (
        run_benchmark_state_tables(package, outputs, lambda_cost=float(args.lambda_cost))
        if args.include_state_tables
        else pd.DataFrame()
    )
    log_stage(started, "ran benchmark state tables")
    pieces = [reference, benchmark_state, utility]
    pieces = [piece for piece in pieces if not piece.empty]
    if not classifiers.empty:
        pieces.append(classifiers)
    combined = pd.concat(pieces, ignore_index=True)
    selected = validation_selected_rows(combined)

    reference.to_csv(args.output_dir / "table_observable_reference_policies.csv", index=False)
    benchmark_state.to_csv(args.output_dir / "table_observable_state_tables.csv", index=False)
    utility.to_csv(args.output_dir / "table_observable_utility_regressors.csv", index=False)
    classifiers.to_csv(args.output_dir / "table_observable_oracle_classifiers.csv", index=False)
    combined.to_csv(args.output_dir / "table_observable_feature_router_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_observable_feature_router_selected.csv", index=False)
    write_figure(args.output_dir, combined)
    write_memo(args.output_dir / "OBSERVABLE_FEATURE_ROUTER_MEMO.md", args.outputs, combined, selected)
    log_stage(started, "wrote outputs")
    print(f"Wrote observable feature router experiment to {args.output_dir}")


def log_stage(started: float, label: str) -> None:
    print(f"[{time.perf_counter() - started:7.1f}s] {label}", flush=True)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_split_columns(outputs: pd.DataFrame) -> pd.DataFrame:
    outputs = outputs.copy()
    if "split" not in outputs.columns:
        for candidate in ["split_y", "split_x"]:
            if candidate in outputs.columns:
                outputs["split"] = outputs[candidate].astype(str)
                break
    if "rank_in_benchmark" not in outputs.columns:
        for candidate in ["rank_in_benchmark_y", "rank_in_benchmark_x"]:
            if candidate in outputs.columns:
                outputs["rank_in_benchmark"] = outputs[candidate]
                break
    drop_columns = [
        column
        for column in outputs.columns
        if column in {"split_x", "split_y", "rank_in_benchmark_x", "rank_in_benchmark_y"}
    ]
    if drop_columns:
        outputs = outputs.drop(columns=drop_columns)
    if "split" not in outputs.columns:
        raise KeyError("split")
    if {"query_id", "model_id"}.issubset(outputs.columns):
        outputs["query_id"] = outputs["query_id"].astype(str)
        outputs["model_id"] = outputs["model_id"].astype(str)
        outputs = outputs.drop_duplicates(["query_id", "model_id"], keep="last")
    return outputs


def fit_feature_store(package, outputs: pd.DataFrame, *, max_text_features: int) -> dict[str, Any]:
    query_cache = build_query_cache(outputs)
    outputs.attrs["query_cache"] = query_cache
    train_ids = split_query_ids(outputs, "train")
    dicts = [query_cache[query_id]["dict"] for query_id in train_ids]
    texts = [query_cache[query_id]["text"] for query_id in train_ids]
    dict_vectorizer = DictVectorizer(sparse=True)
    text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_text_features, norm="l2")
    dict_vectorizer.fit(dicts)
    text_vectorizer.fit(texts)
    return {"dict_vectorizer": dict_vectorizer, "text_vectorizer": text_vectorizer, "query_cache": query_cache}


def run_reference_policies(package, outputs: pd.DataFrame, *, lambda_cost: float, include_existing_reference: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        methods = [("benchmark_lookup_train_utility", benchmark_lookup_selection(package, outputs, split=split))]
        if include_existing_reference:
            methods.extend(
                [
                    ("observable_local_state_v5", package.observable_local_state_selection(outputs, split=split)),
                    ("tool_probe_profile_v4", package.profile_v4_selection_for_split(outputs, split=split)),
                ]
            )
        for method, selected in methods:
            row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
            row.update({"method": method, "family": "reference_policy", "feature_view": "existing_policy", "learner": "none"})
            rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark_state_tables(package, outputs: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_view in ["benchmark_answer_stats", "benchmark_local_agreement", "benchmark_metric_domain"]:
        table = fit_state_action_table(package, outputs, state_view=state_view)
        for split in ["val", "test"]:
            selected = state_action_selection(package, outputs, table, state_view=state_view, split=split)
            row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
            row.update({"method": f"state_table_{state_view}", "family": "state_action_table", "feature_view": state_view, "learner": "train_state_mean_utility"})
            rows.append(row)
    return pd.DataFrame(rows)


def run_utility_regressors(
    package,
    outputs: pd.DataFrame,
    feature_store: dict[str, Any],
    *,
    lambda_cost: float,
    seed: int,
    include_text_view: bool,
    include_tree_learners: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    learners = utility_learners(seed, include_tree_learners=include_tree_learners)
    views = ["dict", "dict_text"] if include_text_view else ["dict"]
    for view in views:
        train_x = transform_features(package, outputs, feature_store, split_query_ids(outputs, "train"), view=view)
        candidate_models = candidate_model_ids(package, outputs)
        utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="first")
        train_y = utility.loc[split_query_ids(outputs, "train"), candidate_models].to_numpy(dtype=float)
        for learner_name, model in learners.items():
            model.fit(train_x, train_y)
            for split in ["val", "test"]:
                selected = predict_utility(package, outputs, feature_store, model, candidate_models, view=view, split=split)
                row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
                row.update({"method": f"utility_{learner_name}_{view}", "family": "observable_utility_regressor", "feature_view": view, "learner": learner_name})
                rows.append(row)
    return pd.DataFrame(rows)


def run_oracle_classifiers(
    package,
    outputs: pd.DataFrame,
    feature_store: dict[str, Any],
    *,
    lambda_cost: float,
    seed: int,
    include_text_view: bool,
    include_tree_learners: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_ids = split_query_ids(outputs, "train")
    oracle = outputs[outputs["split"].eq("train")].loc[outputs[outputs["split"].eq("train")].groupby("query_id")["utility"].idxmax()]
    oracle = oracle.set_index("query_id").loc[train_ids]
    labels = oracle["model_id"].astype(str).tolist()
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    views = ["dict", "dict_text"] if include_text_view else ["dict"]
    for view in views:
        train_x = transform_features(package, outputs, feature_store, train_ids, view=view)
        for learner_name, model in classifier_learners(seed, include_tree_learners=include_tree_learners).items():
            model.fit(train_x, y)
            for split in ["val", "test"]:
                selected = predict_classifier(package, outputs, feature_store, model, encoder, view=view, split=split)
                row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
                row.update({"method": f"oracle_classifier_{learner_name}_{view}", "family": "oracle_action_classifier", "feature_view": view, "learner": learner_name})
                rows.append(row)
    return pd.DataFrame(rows)


def utility_learners(seed: int, *, include_tree_learners: bool) -> dict[str, Any]:
    learners: dict[str, Any] = {
        "ridge_a10": Ridge(alpha=10.0, solver="lsqr"),
        "ridge_a100": Ridge(alpha=100.0, solver="lsqr"),
    }
    if include_tree_learners:
        learners.update(
            {
        "extra_trees_d4": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees_d8": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "random_forest_d8": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
            }
        )
    return learners


def classifier_learners(seed: int, *, include_tree_learners: bool) -> dict[str, Any]:
    learners: dict[str, Any] = {
        "logreg": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed),
        ),
    }
    if include_tree_learners:
        learners.update(
            {
        "extra_trees_d6": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "random_forest_d8": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
            }
        )
    return learners


def predict_utility(package, outputs: pd.DataFrame, feature_store: dict[str, Any], model, candidate_models: list[str], *, view: str, split: str) -> pd.Series:
    query_ids = split_query_ids(outputs, split)
    x = transform_features(package, outputs, feature_store, query_ids, view=view)
    pred = np.asarray(model.predict(x), dtype=float)
    by_query = outputs.set_index(["query_id", "model_id"])
    selected: dict[str, str] = {}
    for row_index, query_id in enumerate(query_ids):
        tool_choice = package.deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            selected[query_id] = tool_choice
        else:
            selected[query_id] = candidate_models[int(np.argmax(pred[row_index]))]
    return pd.Series(selected)


def predict_classifier(package, outputs: pd.DataFrame, feature_store: dict[str, Any], model, encoder: LabelEncoder, *, view: str, split: str) -> pd.Series:
    query_ids = split_query_ids(outputs, split)
    x = transform_features(package, outputs, feature_store, query_ids, view=view)
    labels = encoder.inverse_transform(np.asarray(model.predict(x), dtype=int))
    by_query = outputs.set_index(["query_id", "model_id"])
    selected: dict[str, str] = {}
    for query_id, label in zip(query_ids, labels, strict=True):
        tool_choice = package.deterministic_tool_choice(by_query, query_id)
        selected[query_id] = tool_choice or str(label)
    return pd.Series(selected)


def transform_features(package, outputs: pd.DataFrame, feature_store: dict[str, Any], query_ids: list[str], *, view: str):
    query_cache = feature_store["query_cache"]
    dicts = [query_cache[query_id]["dict"] for query_id in query_ids]
    dict_x = feature_store["dict_vectorizer"].transform(dicts)
    if view == "dict":
        return dict_x
    if view == "dict_text":
        texts = [query_cache[query_id]["text"] for query_id in query_ids]
        text_x = feature_store["text_vectorizer"].transform(texts)
        return hstack([dict_x, text_x], format="csr")
    raise ValueError(f"Unknown feature view: {view}")


def build_query_cache(outputs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    query_rows = outputs.drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    available_models = set(outputs["model_id"].astype(str))
    cache: dict[str, dict[str, Any]] = {}
    for query_id, query in query_rows.iterrows():
        query_key = str(query_id)
        answers = {model_id: local_answer(by_query, query_key, model_id) for model_id in LOCAL_MODELS if model_id in available_models}
        nonempty = [answer for answer in answers.values() if answer]
        counts = pd.Series(nonempty).value_counts() if nonempty else pd.Series(dtype=int)
        stats = {
            "valid": len(nonempty),
            "unique": int(len(counts)),
            "majority": int(counts.iloc[0]) if not counts.empty else 0,
            "agree_pairs": local_agree_pair_count(list(answers.values())),
        }
        cache[query_key] = {
            "query": query,
            "answers": answers,
            "stats": stats,
            "dict": observable_dict_from_query(query, answers, stats),
            "text": observable_text_from_query(query, answers),
        }
    return cache


def observable_dict(package, outputs: pd.DataFrame, query_id: str) -> dict[str, Any]:
    query = outputs.drop_duplicates("query_id").set_index("query_id").loc[query_id]
    by_query = outputs.set_index(["query_id", "model_id"])
    answers = {model_id: local_answer(by_query, query_id, model_id) for model_id in LOCAL_MODELS if model_id in set(outputs["model_id"])}
    nonempty = [answer for answer in answers.values() if answer]
    counts = pd.Series(nonempty).value_counts() if nonempty else pd.Series(dtype=int)
    stats = {
        "valid": len(nonempty),
        "unique": int(len(counts)),
        "majority": int(counts.iloc[0]) if not counts.empty else 0,
        "agree_pairs": local_agree_pair_count(list(answers.values())),
    }
    return observable_dict_from_query(query, answers, stats)


def observable_dict_from_query(query: pd.Series, answers: dict[str, str], stats: dict[str, int]) -> dict[str, Any]:
    features: dict[str, Any] = {
        f"benchmark={query.get('benchmark', '')}": 1,
        f"domain={query.get('domain', '')}": 1,
        f"metric={query.get('metric', '')}": 1,
        "valid_count": int(stats["valid"]),
        "unique_count": int(stats["unique"]),
        "majority_count": int(stats["majority"]),
        "agree_pair_count": int(stats["agree_pairs"]),
        "has_majority": int(stats["majority"] >= 2),
    }
    for model_id, answer in answers.items():
        features[f"{model_id}:valid"] = int(bool(answer))
        features[f"{model_id}:ans_len_bin={length_bin(answer)}"] = 1
        features[f"{model_id}:ans_prefix={answer[:16]}"] = 1 if answer else 0
    models = list(answers)
    for idx, first in enumerate(models):
        for second in models[idx + 1 :]:
            agree = bool(answers[first]) and answers[first] == answers[second]
            features[f"agree:{first}:{second}"] = int(agree)
    return features


def observable_text(package, outputs: pd.DataFrame, query_id: str) -> str:
    query = outputs.drop_duplicates("query_id").set_index("query_id").loc[query_id]
    by_query = outputs.set_index(["query_id", "model_id"])
    answers = {model_id: local_answer(by_query, query_id, model_id) for model_id in LOCAL_MODELS if model_id in set(outputs["model_id"])}
    return observable_text_from_query(query, answers)


def observable_text_from_query(query: pd.Series, answers: dict[str, str]) -> str:
    parts = [
        str(query.get("query_text", "")),
        f"benchmark_{query.get('benchmark', '')}",
        f"domain_{query.get('domain', '')}",
        f"metric_{query.get('metric', '')}",
    ]
    for model_id, answer in answers.items():
        parts.append(f"{model_id}_answer_{answer[:80] if answer else 'empty'}")
    return " ".join(parts)


def local_agree_pair_count(answers: list[str]) -> int:
    total = 0
    for idx, first in enumerate(answers):
        for second in answers[idx + 1 :]:
            if first and first == second:
                total += 1
    return total


def length_bin(answer: str) -> str:
    if not answer:
        return "empty"
    size = len(answer)
    if size <= 1:
        return "one"
    if size <= 4:
        return "short"
    if size <= 16:
        return "medium"
    return "long"


def local_answer(by_query: pd.DataFrame, query_id: str, model_id: str) -> str:
    try:
        row = by_query.loc[(query_id, model_id)]
    except KeyError:
        return ""
    if str(row.get("status", "")) != "success":
        return ""
    value = row.get("parsed_answer", "")
    if pd.isna(value):
        return ""
    answer = str(value).strip().lower()
    if not answer or answer in {"nan", "none", "null", "no_code"} or answer.startswith("failed"):
        return ""
    return answer


def benchmark_lookup_selection(package, outputs: pd.DataFrame, *, split: str) -> pd.Series:
    train = outputs[outputs["split"].eq("train")]
    actions = candidate_model_ids(package, outputs)
    scores = (
        train[train["model_id"].isin(actions)]
        .groupby(["benchmark", "model_id"])
        .agg(mean_utility=("utility", "mean"), mean_quality=("quality_score", "mean"), mean_cost=("normalized_remote_cost", "mean"))
        .reset_index()
        .sort_values(["benchmark", "mean_utility", "mean_quality", "mean_cost"], ascending=[True, False, False, True])
    )
    table = scores.drop_duplicates("benchmark").set_index("benchmark")["model_id"].astype(str).to_dict()
    by_query = outputs.set_index(["query_id", "model_id"])
    queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").sort_values(["benchmark", "query_id"])
    selected: dict[str, str] = {}
    for row in queries.itertuples(index=False):
        query_id = str(row.query_id)
        tool_choice = package.deterministic_tool_choice(by_query, query_id)
        selected[query_id] = tool_choice or table.get(str(row.benchmark), "qwen3-14b-awq-local")
    return pd.Series(selected)


def fit_state_action_table(package, outputs: pd.DataFrame, *, state_view: str) -> dict[tuple[Any, ...], str]:
    train = outputs[outputs["split"].eq("train")].drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    actions = candidate_model_ids(package, outputs)
    table: dict[tuple[Any, ...], str] = {}
    fallback = package.benchmark_utility_fallback(outputs)
    for state, query_ids in state_groups(outputs, train.index.astype(str).tolist(), state_view=state_view).items():
        scores = []
        for model_id in actions:
            selected_rows = []
            for query_id in query_ids:
                try:
                    selected_rows.append(by_query.loc[(query_id, model_id)])
                except KeyError:
                    continue
            if selected_rows:
                frame = pd.DataFrame(selected_rows)
                scores.append(
                    {
                        "model_id": model_id,
                        "utility": float(frame["utility"].mean()),
                        "quality": float(frame["quality_score"].mean()),
                        "cost": float(frame["normalized_remote_cost"].mean()),
                    }
                )
        if scores:
            best = pd.DataFrame(scores).sort_values(["utility", "quality", "cost"], ascending=[False, False, True]).iloc[0]
            table[state] = str(best["model_id"])
    table[("__fallback__",)] = json.dumps(fallback)
    return table


def state_action_selection(package, outputs: pd.DataFrame, table: dict[tuple[Any, ...], str], *, state_view: str, split: str) -> pd.Series:
    query_ids = split_query_ids(outputs, split)
    by_query = outputs.set_index(["query_id", "model_id"])
    fallback = json.loads(table.get(("__fallback__",), "{}"))
    query_cache = outputs.attrs.get("query_cache", {})
    selected: dict[str, str] = {}
    for query_id in query_ids:
        tool_choice = package.deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            selected[query_id] = tool_choice
            continue
        state = state_for_query(outputs, query_id, state_view=state_view)
        query = query_cache.get(query_id, {}).get("query")
        if query is None:
            query = outputs.drop_duplicates("query_id").set_index("query_id").loc[query_id]
        selected[query_id] = table.get(state, fallback.get(str(query["benchmark"]), "qwen3-14b-awq-local"))
    return pd.Series(selected)


def state_groups(outputs: pd.DataFrame, query_ids: list[str], *, state_view: str) -> dict[tuple[Any, ...], list[str]]:
    groups: dict[tuple[Any, ...], list[str]] = {}
    for query_id in query_ids:
        groups.setdefault(state_for_query(outputs, query_id, state_view=state_view), []).append(query_id)
    return groups


def state_for_query(outputs: pd.DataFrame, query_id: str, *, state_view: str) -> tuple[Any, ...]:
    cached = outputs.attrs.get("query_cache", {}).get(query_id)
    if cached:
        query = cached["query"]
        valid = int(cached["stats"]["valid"])
        unique = int(cached["stats"]["unique"])
        majority = int(cached["stats"]["majority"])
        agree_pair_count = int(cached["stats"]["agree_pairs"])
    else:
        query = outputs.drop_duplicates("query_id").set_index("query_id").loc[query_id]
        by_query = outputs.set_index(["query_id", "model_id"])
        answers = [local_answer(by_query, query_id, model_id) for model_id in LOCAL_MODELS if model_id in set(outputs["model_id"])]
        counts = pd.Series([answer for answer in answers if answer]).value_counts()
        majority = int(counts.iloc[0]) if not counts.empty else 0
        valid = sum(bool(answer) for answer in answers)
        unique = int(len(counts))
        agree_pair_count = local_agree_pair_count(answers)
    if state_view == "benchmark_answer_stats":
        return (str(query["benchmark"]), valid, unique, majority)
    if state_view == "benchmark_local_agreement":
        return (str(query["benchmark"]), valid, unique, majority, agree_pair_count)
    if state_view == "benchmark_metric_domain":
        return (str(query["benchmark"]), str(query["domain"]), str(query["metric"]), valid, unique, majority)
    raise ValueError(f"Unknown state view: {state_view}")


def selection_for_split(selected: pd.Series, outputs: pd.DataFrame, split: str) -> pd.Series:
    ids = split_query_ids(outputs, split)
    return selected.loc[[query_id for query_id in ids if query_id in selected.index]]


def candidate_model_ids(package, outputs: pd.DataFrame) -> list[str]:
    return [model_id for model_id in sorted(outputs["model_id"].astype(str).unique()) if model_id != package.TOOL_MODEL]


def split_query_ids(outputs: pd.DataFrame, split: str) -> list[str]:
    return (
        outputs[outputs["split"].eq(split)]
        .drop_duplicates("query_id")
        .sort_values(["benchmark", "query_id"])["query_id"]
        .astype(str)
        .tolist()
    )


def evaluate_selection(package, outputs: pd.DataFrame, selected: pd.Series, *, split: str, lambda_cost: float) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    return package.evaluation_row("candidate", selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in table.groupby("family"):
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.iloc[0]
        rows.append(best)
        test = group[group["split"].eq("test") & group["method"].eq(best["method"])]
        if not test.empty:
            rows.append(test.iloc[0])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["selection_rule"] = "validation_best_mean_utility_by_family"
    return out


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    labels = plot["family"].str.replace("_", " ", regex=False) + " / " + plot["learner"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], plot["mean_utility"].iloc[::-1], color="#4c78a8")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Observable Feature Routing On Broad100")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_observable_feature_router_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, outputs_path: Path, combined: pd.DataFrame, selected: pd.DataFrame) -> None:
    best_test = combined[combined["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(10)
    lines = [
        "# Observable Feature Router",
        "",
        f"Source outputs: `{outputs_path}`.",
        "",
        "This run makes no model/provider API calls. It fits deployable-feature routers on train, selects by validation utility, and reports held-out test rows.",
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Interpretation",
        "",
        "- Features include benchmark/domain/metric, local answer validity, local answer agreement, local answer length/prefix buckets, and optional query/local-answer text.",
        "- This tests whether a more decision-aware learner can recover the broad100 oracle gap from current observable local probes.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
