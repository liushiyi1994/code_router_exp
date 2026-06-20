from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from routecode.controlled.live_stage0 import normalize_answer


BASE_OUTPUTS = Path("results/controlled/live_broad100_stage0/model_outputs.parquet")
AUGMENTED_OUTPUTS = Path("results/controlled/broad100_gemini_strong_solver/model_outputs_with_gemini_strong.parquet")
TOOL_MODEL = "deterministic_math_tool"
LOCAL_MODELS = [
    TOOL_MODEL,
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broad100 paper-inspired probe-signal routing sandbox.")
    parser.add_argument("--outputs", type=Path, default=BASE_OUTPUTS)
    parser.add_argument("--augmented-outputs", type=Path, default=AUGMENTED_OUTPUTS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_probe_signal_sandbox"))
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--include-augmented", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_broad_package()
    suites: list[tuple[str, pd.DataFrame]] = [("base", package.load_outputs(args.outputs, lambda_cost=args.lambda_cost))]
    if args.include_augmented and args.augmented_outputs.exists():
        suites.append(("augmented_gemini_strong", load_precomputed_outputs(args.augmented_outputs, lambda_cost=args.lambda_cost)))

    all_rows: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    all_features: list[pd.DataFrame] = []
    for matrix_name, outputs in suites:
        eval_table, selected, features = run_suite(outputs, matrix_name=matrix_name, package=package, lambda_cost=args.lambda_cost)
        all_rows.append(eval_table)
        all_selected.append(selected)
        all_features.append(features.assign(matrix_name=matrix_name))

    eval_out = pd.concat(all_rows, ignore_index=True)
    selected_out = pd.concat(all_selected, ignore_index=True)
    features_out = pd.concat(all_features, ignore_index=True)
    eval_out.to_csv(args.output_dir / "table_broad_probe_signal_eval.csv", index=False)
    selected_out.to_csv(args.output_dir / "table_broad_probe_signal_selected.csv", index=False)
    features_out.to_csv(args.output_dir / "table_broad_probe_signal_features.csv", index=False)
    write_memo(args.output_dir, eval_out, selected_out)
    print(f"Wrote broad probe-signal sandbox to {args.output_dir}")


def load_broad_package():
    path = Path("experiments/125_phase3_broad_target_method_package.py")
    spec = importlib.util.spec_from_file_location("broad_package", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_precomputed_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce").fillna(0.0)
    for column in ["cost_total_usd", "latency_s", "normalized_remote_cost"]:
        if column not in outputs:
            outputs[column] = 0.0
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    if "utility" not in outputs:
        outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    else:
        outputs["utility"] = pd.to_numeric(outputs["utility"], errors="coerce").fillna(0.0)
    return outputs


def run_suite(outputs: pd.DataFrame, *, matrix_name: str, package, lambda_cost: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_features = build_query_features(outputs)
    methods: dict[str, dict[str, pd.Series]] = {"val": {}, "test": {}}
    for split in ["val", "test"]:
        methods[split]["baseline_observable_local_state_v5"] = package.observable_local_state_selection(outputs, split=split)
        methods[split]["baseline_tool_probe_profile_v4"] = package.profile_v4_selection_for_split(outputs, split=split)
        for k in [1, 3, 5, 10, 20]:
            methods[split][f"knn_query_utility_k{k}"] = knn_utility_selection(
                outputs, query_features, split=split, k=k, use_probe_text=False
            )
            methods[split][f"knn_probe_utility_k{k}"] = knn_utility_selection(
                outputs, query_features, split=split, k=k, use_probe_text=True
            )

    for target_name in ["utility", "regret"]:
        for learner_name in ["ridge", "extra_trees"]:
            pred = regression_route_selection(outputs, query_features, target_name=target_name, learner_name=learner_name)
            for split in ["val", "test"]:
                methods[split][f"rank_{target_name}_{learner_name}"] = pred[split]["all"]
                for cap, selected in pred[split]["budgeted"].items():
                    methods[split][f"rank_{target_name}_{learner_name}_frontier_cap{cap:g}"] = selected

    if "gemini-3.5-flash-strong-solve" in set(outputs["model_id"].astype(str)):
        for split in ["val", "test"]:
            query_ids = outputs[outputs["split"].eq(split)].drop_duplicates("query_id")["query_id"].astype(str)
            methods[split]["all_gemini_strong"] = pd.Series("gemini-3.5-flash-strong-solve", index=query_ids)
        methods["test"]["val_benchmark_strong_else_observable"] = strong_by_validation_benchmark(outputs, package=package)

    eval_rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        split_outputs = outputs[outputs["split"].eq(split)]
        cost_oracle = split_outputs.loc[split_outputs.groupby("query_id")["utility"].idxmax()]
        quality_oracle = split_outputs.loc[split_outputs.groupby("query_id")["quality_score"].idxmax()]
        for method, selected in methods[split].items():
            selected_rows = package.selected_to_rows(outputs, selected, split=split)
            if selected_rows.empty:
                continue
            row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
            row["matrix_name"] = matrix_name
            eval_rows.append(row)
    eval_table = pd.DataFrame(eval_rows)
    selected = validation_selected_rows(eval_table)
    return eval_table, selected, query_features


def answer_key(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "no_code"} or text.lower().startswith("failed"):
        return ""
    if text.lower() == "passed":
        return "passed"
    return normalize_answer(text)


def build_query_features(outputs: pd.DataFrame) -> pd.DataFrame:
    local_models = [model for model in LOCAL_MODELS if model in set(outputs["model_id"].astype(str))]
    rows: list[dict[str, Any]] = []
    by_query = outputs.set_index(["query_id", "model_id"])
    for _, query in outputs.drop_duplicates("query_id").iterrows():
        query_id = str(query["query_id"])
        answers: dict[str, str] = {}
        answer_models: dict[str, list[str]] = {}
        for model_id in local_models:
            try:
                row = by_query.loc[(query_id, model_id)]
            except KeyError:
                answer = ""
            else:
                answer = answer_key(row.get("parsed_answer", "")) if str(row.get("status", "")) == "success" else ""
            answers[model_id] = answer
            if answer:
                answer_models.setdefault(answer, []).append(model_id)
        counts = sorted((len(models) for models in answer_models.values()), reverse=True)
        max_vote = counts[0] if counts else 0
        second_vote = counts[1] if len(counts) > 1 else 0
        total_votes = sum(counts)
        probs = [count / total_votes for count in counts] if total_votes else []
        entropy = -sum(prob * math.log2(prob) for prob in probs if prob > 0)
        local_answer_blob = " ".join(f"{model_id}={answer}" for model_id, answer in answers.items() if answer)
        rows.append(
            {
                "query_id": query_id,
                "split": str(query["split"]),
                "benchmark": str(query["benchmark"]),
                "domain": str(query["domain"]),
                "metric": str(query["metric"]),
                "query_text": str(query["query_text"]),
                "query_len": len(str(query["query_text"])),
                "number_count": len(re.findall(r"[-+]?\d+(?:\.\d+)?", str(query["query_text"]))),
                "latex_count": str(query["query_text"]).count("\\"),
                "local_answer_blob": local_answer_blob,
                "query_feature_text": str(query["query_text"]),
                "probe_feature_text": f"{query['query_text']} {local_answer_blob}",
                "local_valid_answer_count": sum(bool(answer) for answer in answers.values()),
                "local_unique_answer_count": len(answer_models),
                "local_max_vote": max_vote,
                "local_vote_margin": max_vote - second_vote,
                "local_answer_entropy": entropy,
                "tool_available": bool(query.get("tool_available", False)),
            }
        )
    return pd.DataFrame(rows)


def available_models_for_query(outputs: pd.DataFrame, query_id: str, candidate_models: list[str]) -> list[str]:
    query_rows = outputs[outputs["query_id"].astype(str).eq(str(query_id))]
    available: list[str] = []
    for model_id in candidate_models:
        rows = query_rows[query_rows["model_id"].astype(str).eq(model_id)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        if model_id == TOOL_MODEL and not bool(row.get("tool_available", False)):
            continue
        if str(row.get("status", "success")) != "success":
            continue
        available.append(model_id)
    return available


def train_supported_models(outputs: pd.DataFrame) -> list[str]:
    train = outputs[outputs["split"].eq("train")]
    models = []
    for model_id, frame in train.groupby("model_id"):
        if frame["quality_score"].notna().any():
            models.append(str(model_id))
    return sorted(models)


def knn_utility_selection(
    outputs: pd.DataFrame,
    query_features: pd.DataFrame,
    *,
    split: str,
    k: int,
    use_probe_text: bool,
) -> pd.Series:
    text_col = "probe_feature_text" if use_probe_text else "query_feature_text"
    train_features = query_features[query_features["split"].eq("train")].copy()
    target_features = query_features[query_features["split"].eq(split)].copy()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
    x_train = vectorizer.fit_transform(train_features[text_col])
    x_target = vectorizer.transform(target_features[text_col])
    n_neighbors = min(max(1, int(k)), len(train_features))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(x_train)
    _, neighbor_pos = nn.kneighbors(x_target)
    train_qids = train_features["query_id"].astype(str).to_numpy()
    train_util = outputs[outputs["split"].eq("train")].pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean")
    candidate_models = train_supported_models(outputs)
    selected: dict[str, str] = {}
    for row_idx, row in enumerate(target_features.itertuples(index=False)):
        neighbor_ids = train_qids[neighbor_pos[row_idx]]
        scores = train_util.reindex(neighbor_ids).mean(axis=0)
        available = available_models_for_query(outputs, str(row.query_id), candidate_models)
        if not available:
            continue
        best = scores.reindex(available).sort_values(ascending=False)
        selected[str(row.query_id)] = str(best.index[0]) if len(best) else available[0]
    return pd.Series(selected)


def model_reliability_features(outputs: pd.DataFrame) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    train = outputs[outputs["split"].eq("train")].copy()
    global_rel = {
        str(model): float((frame["quality_score"].sum() + 1.0) / (len(frame) + 2.0))
        for model, frame in train.groupby("model_id")
    }
    bench_rel = {
        (str(bench), str(model)): float((frame["quality_score"].sum() + 1.0) / (len(frame) + 2.0))
        for (bench, model), frame in train.groupby(["benchmark", "model_id"])
    }
    return global_rel, bench_rel


def row_feature_table(outputs: pd.DataFrame, query_features: pd.DataFrame) -> pd.DataFrame:
    global_rel, bench_rel = model_reliability_features(outputs)
    qf = query_features.set_index("query_id")
    local_models = [model for model in LOCAL_MODELS if model in set(outputs["model_id"].astype(str))]
    rows: list[dict[str, Any]] = []
    for _, row in outputs.iterrows():
        query_id = str(row["query_id"])
        model_id = str(row["model_id"])
        query = qf.loc[query_id]
        model_answer = answer_key(row.get("parsed_answer", "")) if model_id in local_models else ""
        group_count = 0
        if model_answer:
            for part in str(query["local_answer_blob"]).split():
                if part.endswith(f"={model_answer}") or f"={model_answer}" in part:
                    group_count += 1
        oracle_utility = float(outputs[outputs["query_id"].astype(str).eq(query_id)]["utility"].max())
        rows.append(
            {
                "query_id": query_id,
                "split": str(row["split"]),
                "benchmark": str(row["benchmark"]),
                "domain": str(row["domain"]),
                "metric": str(row["metric"]),
                "model_id": model_id,
                "provider": str(row.get("provider", "")),
                "query_feature_text": str(query["query_feature_text"]),
                "probe_feature_text": str(query["probe_feature_text"]),
                "status": str(row.get("status", "")),
                "is_local": int(bool(row.get("is_local", False))),
                "is_frontier": int(bool(row.get("is_frontier", False))),
                "norm_cost": float(row.get("normalized_remote_cost", 0.0)),
                "query_len": float(query["query_len"]),
                "number_count": float(query["number_count"]),
                "latex_count": float(query["latex_count"]),
                "local_valid_answer_count": float(query["local_valid_answer_count"]),
                "local_unique_answer_count": float(query["local_unique_answer_count"]),
                "local_max_vote": float(query["local_max_vote"]),
                "local_vote_margin": float(query["local_vote_margin"]),
                "local_answer_entropy": float(query["local_answer_entropy"]),
                "candidate_local_group_count": float(group_count),
                "model_global_reliability": float(global_rel.get(model_id, 0.5)),
                "model_benchmark_reliability": float(bench_rel.get((str(row["benchmark"]), model_id), global_rel.get(model_id, 0.5))),
                "quality": float(row["quality_score"]),
                "utility": float(row["utility"]),
                "regret": oracle_utility - float(row["utility"]),
            }
        )
    return pd.DataFrame(rows)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("query_text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=16000), "probe_feature_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["benchmark", "domain", "metric", "model_id", "provider", "status"]),
            (
                "num",
                StandardScaler(with_mean=False),
                [
                    "is_local",
                    "is_frontier",
                    "norm_cost",
                    "query_len",
                    "number_count",
                    "latex_count",
                    "local_valid_answer_count",
                    "local_unique_answer_count",
                    "local_max_vote",
                    "local_vote_margin",
                    "local_answer_entropy",
                    "candidate_local_group_count",
                    "model_global_reliability",
                    "model_benchmark_reliability",
                ],
            ),
        ]
    )


def learner(name: str):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=160, random_state=17, n_jobs=-1, min_samples_leaf=2)
    raise ValueError(name)


def regression_route_selection(
    outputs: pd.DataFrame,
    query_features: pd.DataFrame,
    *,
    target_name: str,
    learner_name: str,
) -> dict[str, dict[str, Any]]:
    features = row_feature_table(outputs, query_features)
    candidate_models = train_supported_models(outputs)
    features = features[features["model_id"].isin(candidate_models)].copy()
    train = features[features["split"].eq("train")].copy()
    target = "utility" if target_name == "utility" else "regret"
    model = Pipeline([("pre", preprocessor()), ("model", learner(learner_name))])
    model.fit(train, train[target])
    result: dict[str, dict[str, Any]] = {}
    for split in ["val", "test"]:
        frame = features[features["split"].eq(split)].copy()
        frame["pred"] = model.predict(frame)
        if target_name == "utility":
            frame["score"] = frame["pred"]
        else:
            frame["score"] = -frame["pred"]
        result[split] = {
            "all": select_from_predicted_frame(outputs, frame, score_column="score", local_only=False),
            "budgeted": {},
        }
        for cap in [0.25, 0.35, 0.40]:
            result[split]["budgeted"][cap] = budgeted_selection_from_predictions(outputs, frame, frontier_cap=cap)
    return result


def select_from_predicted_frame(outputs: pd.DataFrame, frame: pd.DataFrame, *, score_column: str, local_only: bool) -> pd.Series:
    selected: dict[str, str] = {}
    for query_id, group in frame.groupby("query_id"):
        group = group.copy()
        if local_only:
            group = group[group["is_frontier"].eq(0)]
        available = set(available_models_for_query(outputs, str(query_id), group["model_id"].astype(str).tolist()))
        group = group[group["model_id"].astype(str).isin(available)]
        if group.empty:
            continue
        selected[str(query_id)] = str(group.sort_values(score_column, ascending=False).iloc[0]["model_id"])
    return pd.Series(selected)


def budgeted_selection_from_predictions(outputs: pd.DataFrame, frame: pd.DataFrame, *, frontier_cap: float) -> pd.Series:
    selected: dict[str, str] = {}
    candidates: list[tuple[str, str, str, float]] = []
    for query_id, group in frame.groupby("query_id"):
        available = set(available_models_for_query(outputs, str(query_id), group["model_id"].astype(str).tolist()))
        group = group[group["model_id"].astype(str).isin(available)].copy()
        if group.empty:
            continue
        local = group[group["is_frontier"].eq(0)].sort_values("score", ascending=False)
        all_rows = group.sort_values("score", ascending=False)
        local_model = str(local.iloc[0]["model_id"]) if len(local) else str(all_rows.iloc[0]["model_id"])
        all_model = str(all_rows.iloc[0]["model_id"])
        local_score = float(local.iloc[0]["score"]) if len(local) else float("-inf")
        all_score = float(all_rows.iloc[0]["score"])
        if bool(all_rows.iloc[0]["is_frontier"]):
            candidates.append((str(query_id), all_model, local_model, all_score - local_score))
        selected[str(query_id)] = local_model
    max_frontier = int(math.floor(float(frontier_cap) * max(len(selected), 1)))
    for query_id, frontier_model, _, gain in sorted(candidates, key=lambda item: item[3], reverse=True)[:max_frontier]:
        if gain > 0:
            selected[query_id] = frontier_model
    return pd.Series(selected)


def strong_by_validation_benchmark(outputs: pd.DataFrame, *, package) -> pd.Series:
    strong = "gemini-3.5-flash-strong-solve"
    val_queries = outputs[outputs["split"].eq("val")].drop_duplicates("query_id").set_index("query_id")
    selected_benchmarks: set[str] = set()
    base_val = package.observable_local_state_selection(outputs, split="val")
    for benchmark, frame in val_queries.groupby("benchmark"):
        ids = set(frame.index.astype(str))
        strong_sel = pd.Series(strong, index=frame.index.astype(str))
        base_sel = base_val[base_val.index.astype(str).isin(ids)]
        strong_rows = package.selected_to_rows(outputs, strong_sel, split="val")
        base_rows = package.selected_to_rows(outputs, base_sel, split="val")
        if len(strong_rows) and float(strong_rows["utility"].mean()) >= float(base_rows["utility"].mean()):
            selected_benchmarks.add(str(benchmark))
    test_queries = outputs[outputs["split"].eq("test")].drop_duplicates("query_id").set_index("query_id")
    selected = package.observable_local_state_selection(outputs, split="test")
    for query_id, row in test_queries.iterrows():
        if str(row["benchmark"]) in selected_benchmarks:
            selected.loc[str(query_id)] = strong
    return selected


def validation_selected_rows(eval_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for matrix_name, matrix in eval_table.groupby("matrix_name"):
        val = matrix[matrix["split"].eq("val")].copy()
        if val.empty:
            continue
        picks = {
            "val_best_utility": val.sort_values(["mean_utility", "mean_quality"], ascending=False).head(1),
            "val_best_quality": val.sort_values(["mean_quality", "mean_utility"], ascending=False).head(1),
            "val_best_under_frontier_040": val[val["frontier_call_rate"].le(0.40)].sort_values(
                ["mean_utility", "mean_quality"], ascending=False
            ).head(1),
        }
        seen: set[str] = set()
        for rule, picked in picks.items():
            if picked.empty:
                continue
            method = str(picked.iloc[0]["method"])
            if method in seen:
                continue
            seen.add(method)
            rows.append(picked.assign(selection_rule=rule))
            test = matrix[matrix["method"].eq(method) & matrix["split"].eq("test")]
            if not test.empty:
                rows.append(test.assign(selection_rule=f"{rule}_test"))
        diagnostic = matrix[matrix["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(5)
        rows.append(diagnostic.assign(selection_rule="top5_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_memo(output_dir: Path, eval_table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "matrix_name",
        "selection_rule",
        "method",
        "split",
        "mean_quality",
        "mean_utility",
        "quality_oracle_mean_quality",
        "cost_oracle_mean_utility",
        "quality_gap_to_oracle",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "remote_cost_total_usd",
    ]
    lines = [
        "# Broad100 Probe-Signal Sandbox",
        "",
        "This sandbox tests paper-inspired routing signals on cached broad100 outputs.",
        "",
        "Implemented here:",
        "- kNN utility-neighborhood routing over query text and query+local-probe answer text.",
        "- Local answer consensus features: vote count, margin, entropy, local answer groups.",
        "- Supervised utility/regret ranking from train only.",
        "- Budgeted frontier ranking with 25%, 35%, and 40% frontier caps.",
        "",
        "Not yet implemented in this cached-only pass:",
        "- True token-logprob confidence probes from vLLM.",
        "- Prefill/hidden-activation probes.",
        "",
        "## Validation-Selected And Diagnostic Rows",
        "",
        "```csv",
        selected[cols].to_csv(index=False).strip() if not selected.empty else "",
        "```",
        "",
        "## Top Test Rows",
        "",
    ]
    top = eval_table[eval_table["split"].eq("test")].sort_values(["matrix_name", "mean_utility", "mean_quality"], ascending=[True, False, False])
    lines.extend(["```csv", top.head(30).to_csv(index=False).strip(), "```", ""])
    output_dir.joinpath("BROAD_PROBE_SIGNAL_SANDBOX_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
