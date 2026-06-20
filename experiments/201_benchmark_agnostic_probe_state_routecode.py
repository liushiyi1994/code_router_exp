from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import pairwise_distances_argmin
from sklearn.multiclass import OneVsRestClassifier
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
SMALL_MODELS = ["qwen3-4b-local", "qwen3-8b-local"]
MEDIUM_MODELS = ["qwen3-14b-awq-local", "qwen3-32b-awq-local"]
K_VALUES = [2, 4, 8, 16]
ROUTECODE_PREDICTOR_K_VALUES = [4, 8, 16]
COMBINED_K_VALUES = [8, 16]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
CS = [1.0]
KMEANS_N_INIT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark-agnostic ProbeCode / probe-state RouteCode over cached Broad100 outputs."
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
        "--self-consistency",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "table_vllm_self_consistency_probe.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode"),
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
    features = build_probe_state_features(outputs, args.self_consistency)
    matrix = build_utility_matrix(outputs)

    features.to_csv(args.output_dir / "table_probe_state_features.csv", index=False)
    standard = run_standard_eval(features, matrix, args)
    standard_all = standard["all"]
    standard_selected = standard["selected"]
    transfer = run_benchmark_heldout_eval(features, matrix, args)
    transfer_all = transfer["all"]
    transfer_selected = transfer["selected"]
    transfer_summary = summarize_transfer(transfer_selected)
    cards = build_code_cards(features, matrix, standard["selected_probe_state"], args)

    standard_all.to_csv(args.output_dir / "table_probe_state_policy_all.csv", index=False)
    standard_selected.to_csv(args.output_dir / "table_probe_state_policy_selected.csv", index=False)
    transfer_all.to_csv(args.output_dir / "table_probe_state_benchmark_heldout_all.csv", index=False)
    transfer_selected.to_csv(args.output_dir / "table_probe_state_benchmark_heldout_selected.csv", index=False)
    transfer_summary.to_csv(args.output_dir / "table_probe_state_benchmark_heldout_summary.csv", index=False)
    # Backward-compatible selected table for downstream scripts and older memos.
    transfer_selected.to_csv(args.output_dir / "table_probe_state_benchmark_heldout.csv", index=False)
    cards.to_csv(args.output_dir / "table_probe_state_code_cards.csv", index=False)
    write_code_cards_md(args.output_dir / "probe_state_code_cards.md", cards)
    write_memo(
        args.output_dir / "PROBE_STATE_ROUTECODE_MEMO.md",
        args,
        features,
        standard_selected,
        transfer_selected,
        transfer_summary,
        cards,
    )
    print(f"Wrote benchmark-agnostic probe-state RouteCode results to {args.output_dir}")


def build_probe_state_features(outputs: pd.DataFrame, self_consistency_path: Path) -> pd.DataFrame:
    meta = (
        outputs.sort_values(["query_id", "model_id"])
        .groupby("query_id", as_index=False)
        .first()[["query_id", "query_text", "split", "benchmark", "domain", "metric"]]
    )
    by_query = {str(query_id): group.copy() for query_id, group in outputs.groupby("query_id", sort=False)}
    rows: list[dict[str, Any]] = []
    for item in meta.itertuples(index=False):
        query_id = str(item.query_id)
        group = by_query[query_id]
        action_rows = {str(row.model_id): row for row in group.itertuples(index=False)}
        local_answers = {
            model_id: normalize_answer(getattr(action_rows[model_id], "parsed_answer", ""))
            for model_id in LOCAL_MODELS
            if model_id in action_rows
        }
        row: dict[str, Any] = {
            "query_id": query_id,
            "query_text": str(item.query_text),
            "split": str(item.split),
            "benchmark": str(item.benchmark),
            "domain": str(item.domain),
            "metric": str(item.metric),
            "query_chars": float(len(str(item.query_text))),
            "query_words": float(len(str(item.query_text).split())),
            "is_multiple_choice_prompt": float("A)" in str(item.query_text) and "B)" in str(item.query_text)),
            "is_exact_answer_prompt": float(str(item.metric) == "exact_final_answer"),
        }
        row.update(local_agreement_features(action_rows, local_answers))
        for model_id in LOCAL_MODELS:
            prefix = short_model_name(model_id)
            model_row = action_rows.get(model_id)
            answer = local_answers.get(model_id, "")
            row[f"{prefix}_valid"] = float(bool(answer))
            row[f"{prefix}_answer_chars"] = float(len(answer))
            row[f"{prefix}_status_success"] = float(bool(model_row is not None and str(model_row.status) == "success"))
            row[f"{prefix}_output_tokens"] = float(getattr(model_row, "output_tokens", 0.0) or 0.0) if model_row is not None else 0.0
            row[f"{prefix}_latency_s"] = float(getattr(model_row, "latency_s", 0.0) or 0.0) if model_row is not None else 0.0
        rows.append(row)
    features = pd.DataFrame(rows)
    features = merge_self_consistency(features, self_consistency_path)
    features = merge_optional_logprob_features(features)
    numeric_cols = feature_columns(features)
    features[numeric_cols] = features[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return features.sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def local_agreement_features(action_rows: dict[str, Any], local_answers: dict[str, str]) -> dict[str, float]:
    valid_answers = [answer for answer in local_answers.values() if answer]
    counts = Counter(valid_answers)
    top_counts = counts.most_common()
    top = top_counts[0][1] if top_counts else 0
    second = top_counts[1][1] if len(top_counts) > 1 else 0
    valid = len(valid_answers)
    entropy = answer_entropy(counts)
    small_answers = [local_answers.get(model, "") for model in SMALL_MODELS if local_answers.get(model, "")]
    medium_answers = [local_answers.get(model, "") for model in MEDIUM_MODELS if local_answers.get(model, "")]
    small_top = Counter(small_answers).most_common(1)[0][0] if small_answers else ""
    medium_top = Counter(medium_answers).most_common(1)[0][0] if medium_answers else ""
    answer_lengths = [len(answer) for answer in valid_answers]
    output_tokens = [
        float(getattr(action_rows[model], "output_tokens", 0.0) or 0.0)
        for model in LOCAL_MODELS
        if model in action_rows
    ]
    return {
        "local_valid_count": float(valid),
        "local_missing_count": float(max(0, len(LOCAL_MODELS) - valid)),
        "local_unique_answer_count": float(len(counts)),
        "local_top_vote_count": float(top),
        "local_vote_frac": float(top / valid) if valid else 0.0,
        "local_vote_margin": float((top - second) / valid) if valid else 0.0,
        "local_vote_entropy": entropy,
        "local_all_agree": float(valid > 0 and len(counts) == 1),
        "small_valid_count": float(len(small_answers)),
        "small_unique_answer_count": float(len(set(small_answers))),
        "medium_valid_count": float(len(medium_answers)),
        "medium_unique_answer_count": float(len(set(medium_answers))),
        "small_medium_agree": float(bool(small_top and medium_top and small_top == medium_top)),
        "small_medium_disagree": float(bool(small_top and medium_top and small_top != medium_top)),
        "q4_q8_agree": agree(local_answers, "qwen3-4b-local", "qwen3-8b-local"),
        "q4_q14_agree": agree(local_answers, "qwen3-4b-local", "qwen3-14b-awq-local"),
        "q8_q14_agree": agree(local_answers, "qwen3-8b-local", "qwen3-14b-awq-local"),
        "q14_q32_agree": agree(local_answers, "qwen3-14b-awq-local", "qwen3-32b-awq-local"),
        "q32_sc_agree": agree(local_answers, "qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local"),
        "answer_chars_mean": float(np.mean(answer_lengths)) if answer_lengths else 0.0,
        "answer_chars_std": float(np.std(answer_lengths)) if answer_lengths else 0.0,
        "output_tokens_mean": float(np.mean(output_tokens)) if output_tokens else 0.0,
        "output_tokens_std": float(np.std(output_tokens)) if output_tokens else 0.0,
    }


def merge_self_consistency(features: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        features["sc_available"] = 0.0
        return features
    probe = pd.read_csv(path)
    cols = [
        "query_id",
        "n_samples",
        "valid_count",
        "top_vote_count",
        "vote_frac",
        "vote_margin",
        "vote_entropy",
        "all_samples_agree",
        "latency_s",
        "input_tokens",
        "output_tokens",
    ]
    probe = probe[[col for col in cols if col in probe.columns]].copy()
    rename = {col: f"sc_{col}" for col in probe.columns if col != "query_id"}
    probe = probe.rename(columns=rename)
    probe["sc_available"] = 1.0
    merged = features.merge(probe, on="query_id", how="left")
    for col in merged.columns:
        if col.startswith("sc_"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def merge_optional_logprob_features(features: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        (
            Path("results/controlled/broad100_qwen4_logprob_probe/table_vllm_logprob_probe.csv"),
            "q4_lp",
            ["logprob_mean", "logprob_min", "logprob_margin_mean", "logprob_margin_min", "logprob_first_token_margin"],
        ),
        (
            Path("results/controlled/broad100_qwen4_choice_logprob_probe/table_vllm_choice_logprob_probe.csv"),
            "q4_choice",
            ["choice_logprob_margin", "choice_entropy", "choice_seen_count", "choice_missing_count"],
        ),
        (
            Path("results/controlled/broad100_qwen32_choice_logprob_probe/table_vllm_choice_logprob_probe.csv"),
            "q32_choice",
            ["choice_logprob_margin", "choice_entropy", "choice_seen_count", "choice_missing_count"],
        ),
    ]
    merged = features.copy()
    for path, prefix, cols in candidates:
        if not path.exists():
            merged[f"{prefix}_available"] = 0.0
            continue
        probe = pd.read_csv(path)
        present = ["query_id", *[col for col in cols if col in probe.columns]]
        if len(present) <= 1:
            merged[f"{prefix}_available"] = 0.0
            continue
        probe = probe[present].copy().rename(columns={col: f"{prefix}_{col}" for col in present if col != "query_id"})
        probe[f"{prefix}_available"] = 1.0
        merged = merged.merge(probe, on="query_id", how="left")
        for col in merged.columns:
            if col.startswith(prefix):
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def build_utility_matrix(outputs: pd.DataFrame) -> dict[str, Any]:
    model_ids = sorted(outputs["model_id"].astype(str).unique().tolist())
    utility = outputs.pivot(index="query_id", columns="model_id", values="utility").reindex(columns=model_ids)
    quality = outputs.pivot(index="query_id", columns="model_id", values="quality_score").reindex(columns=model_ids)
    norm_cost = outputs.pivot(index="query_id", columns="model_id", values="normalized_remote_cost").reindex(columns=model_ids)
    frontier = outputs.pivot(index="query_id", columns="model_id", values="is_frontier").reindex(columns=model_ids)
    utility = utility.fillna(-1e9)
    quality = quality.fillna(0.0)
    norm_cost = norm_cost.fillna(0.0)
    frontier = frontier.fillna(False).astype(bool)
    oracle_idx = utility.to_numpy().argmax(axis=1)
    oracle_action = pd.Series([model_ids[i] for i in oracle_idx], index=utility.index, name="oracle_action")
    oracle_utility = pd.Series(utility.to_numpy()[np.arange(len(utility)), oracle_idx], index=utility.index, name="oracle_utility")
    oracle_quality = pd.Series(quality.to_numpy()[np.arange(len(quality)), oracle_idx], index=quality.index, name="oracle_quality")
    return {
        "model_ids": model_ids,
        "utility": utility,
        "quality": quality,
        "cost": norm_cost,
        "frontier": frontier,
        "oracle_action": oracle_action,
        "oracle_utility": oracle_utility,
        "oracle_quality": oracle_quality,
    }


def run_standard_eval(features: pd.DataFrame, matrix: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_ids = features.loc[features["split"].eq("train"), "query_id"].astype(str).tolist()
    val_ids = features.loc[features["split"].eq("val"), "query_id"].astype(str).tolist()
    test_ids = features.loc[features["split"].eq("test"), "query_id"].astype(str).tolist()
    candidates = run_candidates(features, matrix, train_ids, val_ids, test_ids, args, scenario="standard")
    selected = select_by_validation(candidates)
    probe_rows = selected[
        selected["family"].eq("probe_state_kmeans") & selected["eval_split"].eq("test")
    ].copy()
    selected_probe_state = None
    if not probe_rows.empty:
        selected_probe_state = {
            "method": str(probe_rows.iloc[0]["method"]),
            "k": int(probe_rows.iloc[0]["k"]),
            "include_benchmark_id": False,
            "train_ids": train_ids,
        }
    return {"all": candidates, "selected": selected, "selected_probe_state": selected_probe_state}


def run_benchmark_heldout_eval(features: pd.DataFrame, matrix: dict[str, Any], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    all_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    benchmarks = sorted(features["benchmark"].astype(str).unique().tolist())
    for heldout in benchmarks:
        train_ids = features.loc[
            features["split"].eq("train") & features["benchmark"].astype(str).ne(heldout), "query_id"
        ].astype(str).tolist()
        val_ids = features.loc[
            features["split"].eq("val") & features["benchmark"].astype(str).ne(heldout), "query_id"
        ].astype(str).tolist()
        test_ids = features.loc[
            features["split"].eq("test") & features["benchmark"].astype(str).eq(heldout), "query_id"
        ].astype(str).tolist()
        candidates = run_candidates(
            features,
            matrix,
            train_ids,
            val_ids,
            test_ids,
            args,
            scenario="benchmark_heldout",
            heldout_benchmark=heldout,
        )
        all_rows.append(candidates)
        selected_rows.append(select_by_validation(candidates))
    return {
        "all": pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(),
        "selected": pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(),
    }


def summarize_transfer(transfer_selected: pd.DataFrame) -> pd.DataFrame:
    if transfer_selected.empty:
        return pd.DataFrame()
    transfer_test = transfer_selected[transfer_selected["eval_split"].eq("test")].copy()
    if transfer_test.empty:
        return pd.DataFrame()
    return (
        transfer_test.groupby("family", as_index=False)
        .agg(
            mean_heldout_quality=("mean_quality", "mean"),
            mean_heldout_utility=("mean_utility", "mean"),
            mean_heldout_oracle_ratio=("oracle_utility_ratio", "mean"),
            mean_normalized_cost=("mean_normalized_cost", "mean"),
            mean_frontier_call_rate=("frontier_call_rate", "mean"),
            n_heldout_benchmarks=("heldout_benchmark", "nunique"),
        )
        .sort_values("mean_heldout_utility", ascending=False)
        .reset_index(drop=True)
    )


def run_candidates(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    *,
    scenario: str,
    heldout_benchmark: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(global_best_candidate(matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark))
    rows.extend(benchmark_lookup_candidate(features, matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark))
    rows.extend(text_utility_candidates(features, matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark))
    rows.extend(probe_utility_candidates(features, matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark))
    rows.extend(probe_state_candidates(features, matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark, include_benchmark_id=False))
    rows.extend(probe_state_candidates(features, matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark, include_benchmark_id=True))
    rows.extend(routecode_oracle_label_candidates(matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark))
    rows.extend(oracle_local_vs_large_gate_candidates(matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark))
    rows.extend(routecode_predictor_candidates(features, matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark, view="probe"))
    rows.extend(routecode_predictor_candidates(features, matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark, view="text"))
    rows.extend(routecode_plus_probe_state_candidates(features, matrix, train_ids, val_ids, test_ids, args, scenario, heldout_benchmark))
    return pd.DataFrame(rows)


def global_best_candidate(
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    model = best_action_for_ids(matrix, train_ids)
    return [
        metric_row(matrix, ids, pd.Series(model, index=ids), "global_best_single", "global_best_single", split, scenario, heldout, args)
        for split, ids in [("val", val_ids), ("test", test_ids)]
    ]


def benchmark_lookup_candidate(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    global_model = best_action_for_ids(matrix, train_ids)
    train_meta = features.set_index("query_id").loc[train_ids]
    mapping = {
        benchmark: best_action_for_ids(matrix, ids.index.astype(str).tolist())
        for benchmark, ids in train_meta.groupby("benchmark", sort=False)
    }
    rows = []
    for split, ids in [("val", val_ids), ("test", test_ids)]:
        meta = features.set_index("query_id").loc[ids]
        selected = pd.Series(
            [mapping.get(str(row.benchmark), global_model) for row in meta.itertuples()],
            index=ids,
        )
        rows.append(metric_row(matrix, ids, selected, "benchmark_lookup_train", "benchmark_lookup", split, scenario, heldout, args))
    return rows


def text_utility_candidates(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    rows = []
    y_train = matrix["utility"].loc[train_ids].to_numpy()
    texts = features.set_index("query_id")["query_text"]
    for alpha in ALPHAS:
        model = make_pipeline(
            TfidfVectorizer(max_features=4096, min_df=2, ngram_range=(1, 2)),
            Ridge(alpha=float(alpha)),
        )
        model.fit(texts.loc[train_ids].tolist(), y_train)
        for split, ids in [("val", val_ids), ("test", test_ids)]:
            pred = model.predict(texts.loc[ids].tolist())
            selected = actions_from_scores(pred, matrix["model_ids"], ids)
            rows.append(
                metric_row(
                    matrix,
                    ids,
                    selected,
                    f"text_utility_ridge_alpha{alpha:g}",
                    "text_only_utility_router",
                    split,
                    scenario,
                    heldout,
                    args,
                    alpha=float(alpha),
                )
            )
    return rows


def probe_utility_candidates(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    rows = []
    cols = feature_columns(features)
    x_train = features.set_index("query_id").loc[train_ids, cols].to_numpy()
    y_train = matrix["utility"].loc[train_ids].to_numpy()
    for alpha in ALPHAS:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(x_train, y_train)
        for split, ids in [("val", val_ids), ("test", test_ids)]:
            x_eval = features.set_index("query_id").loc[ids, cols].to_numpy()
            selected = actions_from_scores(model.predict(x_eval), matrix["model_ids"], ids)
            rows.append(
                metric_row(
                    matrix,
                    ids,
                    selected,
                    f"probe_utility_ridge_alpha{alpha:g}",
                    "direct_probe_utility_router",
                    split,
                    scenario,
                    heldout,
                    args,
                    alpha=float(alpha),
                )
            )
    return rows


def probe_state_candidates(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
    *,
    include_benchmark_id: bool,
) -> list[dict[str, Any]]:
    rows = []
    cols = feature_columns(features)
    x_train, x_val, x_test = probe_matrix(features, cols, train_ids, val_ids, test_ids, include_benchmark_id=include_benchmark_id)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
    x_val_scaled = scaler.transform(imputer.transform(x_val))
    x_test_scaled = scaler.transform(imputer.transform(x_test))
    family = "probe_state_with_benchmark_id_diagnostic" if include_benchmark_id else "probe_state_kmeans"
    suffix = "_with_benchmark_id" if include_benchmark_id else ""
    for k in K_VALUES:
        kmeans = KMeans(n_clusters=int(k), random_state=int(args.seed), n_init=KMEANS_N_INIT)
        train_labels = kmeans.fit_predict(x_train_scaled)
        action_by_label = best_action_by_label(matrix, train_ids, train_labels)
        fallback = best_action_for_ids(matrix, train_ids)
        for split, ids, labels in [
            ("val", val_ids, kmeans.predict(x_val_scaled)),
            ("test", test_ids, kmeans.predict(x_test_scaled)),
        ]:
            selected = pd.Series([action_by_label.get(int(label), fallback) for label in labels], index=ids)
            rows.append(
                metric_row(
                    matrix,
                    ids,
                    selected,
                    f"probe_state_k{k}{suffix}",
                    family,
                    split,
                    scenario,
                    heldout,
                    args,
                    k=int(k),
                )
            )
    return rows


def routecode_oracle_label_candidates(
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    rows = []
    y_train = matrix["utility"].loc[train_ids].to_numpy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(imputer.fit_transform(y_train))
    for k in K_VALUES:
        kmeans = KMeans(n_clusters=int(k), random_state=int(args.seed), n_init=KMEANS_N_INIT)
        train_labels = kmeans.fit_predict(y_train_scaled)
        action_by_label = best_action_by_label(matrix, train_ids, train_labels)
        fallback = best_action_for_ids(matrix, train_ids)
        for split, ids in [("val", val_ids), ("test", test_ids)]:
            y_eval = matrix["utility"].loc[ids].to_numpy()
            labels = pairwise_distances_argmin(scaler.transform(imputer.transform(y_eval)), kmeans.cluster_centers_)
            selected = pd.Series([action_by_label.get(int(label), fallback) for label in labels], index=ids)
            rows.append(
                metric_row(
                    matrix,
                    ids,
                    selected,
                    f"oracle_routecode_label_k{k}",
                    "oracle_routecode_label_upper_bound",
                    split,
                    scenario,
                    heldout,
                    args,
                    k=int(k),
                    diagnostic=True,
                )
            )
    return rows


def oracle_local_vs_large_gate_candidates(
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    local_models = [model for model in matrix["model_ids"] if not bool(matrix["frontier"][model].any())]
    large_models = [model for model in matrix["model_ids"] if bool(matrix["frontier"][model].any())]
    if not local_models or not large_models:
        return []
    best_local = best_action_for_ids(matrix, train_ids, candidate_models=local_models)
    best_large = best_action_for_ids(matrix, train_ids, candidate_models=large_models)
    rows = []
    for split, ids in [("val", val_ids), ("test", test_ids)]:
        util = matrix["utility"].loc[ids]
        selected = pd.Series(
            np.where(
                util[best_large].to_numpy() > util[best_local].to_numpy(),
                best_large,
                best_local,
            ),
            index=ids,
        )
        rows.append(
            metric_row(
                matrix,
                ids,
                selected,
                f"oracle_fixed_local_vs_large_gate:{best_local}:vs:{best_large}",
                "oracle_local_vs_large_gate_upper_bound",
                split,
                scenario,
                heldout,
                args,
                diagnostic=True,
            )
        )
    return rows


def routecode_predictor_candidates(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
    *,
    view: str,
) -> list[dict[str, Any]]:
    rows = []
    y_train_utility = matrix["utility"].loc[train_ids].to_numpy()
    util_imputer = SimpleImputer(strategy="median")
    util_scaler = StandardScaler()
    y_train_scaled = util_scaler.fit_transform(util_imputer.fit_transform(y_train_utility))
    cols = feature_columns(features)
    feature_index = features.set_index("query_id")
    for k in ROUTECODE_PREDICTOR_K_VALUES:
        codebook = KMeans(n_clusters=int(k), random_state=int(args.seed), n_init=KMEANS_N_INIT)
        train_labels = codebook.fit_predict(y_train_scaled)
        if len(set(train_labels)) < 2:
            continue
        action_by_label = best_action_by_label(matrix, train_ids, train_labels)
        fallback = best_action_for_ids(matrix, train_ids)
        for c_value in CS:
            if view == "probe":
                x_train = feature_index.loc[train_ids, cols].to_numpy()
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    OneVsRestClassifier(
                        LogisticRegression(C=float(c_value), max_iter=300, class_weight="balanced", solver="liblinear")
                    ),
                )
                model.fit(x_train, train_labels)
                eval_inputs = {
                    "val": feature_index.loc[val_ids, cols].to_numpy(),
                    "test": feature_index.loc[test_ids, cols].to_numpy(),
                }
                family = "probe_to_routecode_label"
                method = f"probe_to_routecode_k{k}_c{c_value:g}"
            else:
                texts = feature_index["query_text"]
                model = make_pipeline(
                    TfidfVectorizer(max_features=4096, min_df=2, ngram_range=(1, 2)),
                    OneVsRestClassifier(
                        LogisticRegression(C=float(c_value), max_iter=300, class_weight="balanced", solver="liblinear")
                    ),
                )
                model.fit(texts.loc[train_ids].tolist(), train_labels)
                eval_inputs = {"val": texts.loc[val_ids].tolist(), "test": texts.loc[test_ids].tolist()}
                family = "text_to_routecode_label"
                method = f"text_to_routecode_k{k}_c{c_value:g}"
            for split, ids in [("val", val_ids), ("test", test_ids)]:
                labels = model.predict(eval_inputs[split])
                selected = pd.Series([action_by_label.get(int(label), fallback) for label in labels], index=ids)
                rows.append(
                    metric_row(
                        matrix,
                        ids,
                        selected,
                        method,
                        family,
                        split,
                        scenario,
                        heldout,
                        args,
                        k=int(k),
                        c=float(c_value),
                    )
                )
    return rows


def routecode_plus_probe_state_candidates(
    features: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    rows = []
    feature_index = features.set_index("query_id")
    cols = feature_columns(features)
    texts = feature_index["query_text"]

    probe_x_train, probe_x_val, probe_x_test = probe_matrix(
        features, cols, train_ids, val_ids, test_ids, include_benchmark_id=False
    )
    probe_imputer = SimpleImputer(strategy="median")
    probe_scaler = StandardScaler()
    probe_x_train = probe_scaler.fit_transform(probe_imputer.fit_transform(probe_x_train))
    probe_x_val = probe_scaler.transform(probe_imputer.transform(probe_x_val))
    probe_x_test = probe_scaler.transform(probe_imputer.transform(probe_x_test))

    util_train = matrix["utility"].loc[train_ids].to_numpy()
    util_imputer = SimpleImputer(strategy="median")
    util_scaler = StandardScaler()
    util_train_scaled = util_scaler.fit_transform(util_imputer.fit_transform(util_train))

    for k in COMBINED_K_VALUES:
        route_codebook = KMeans(n_clusters=int(k), random_state=int(args.seed), n_init=KMEANS_N_INIT)
        train_route_labels = route_codebook.fit_predict(util_train_scaled)
        if len(set(train_route_labels)) < 2:
            continue

        route_predictor = make_pipeline(
            TfidfVectorizer(max_features=4096, min_df=2, ngram_range=(1, 2)),
            OneVsRestClassifier(
                LogisticRegression(C=1.0, max_iter=300, class_weight="balanced", solver="liblinear")
            ),
        )
        route_predictor.fit(texts.loc[train_ids].tolist(), train_route_labels)

        probe_codebook = KMeans(n_clusters=int(k), random_state=int(args.seed), n_init=KMEANS_N_INIT)
        train_probe_labels = probe_codebook.fit_predict(probe_x_train)
        pair_to_action = best_action_by_pair(matrix, train_ids, train_route_labels, train_probe_labels)
        fallback = best_action_for_ids(matrix, train_ids)

        route_labels_by_split = {
            "val": route_predictor.predict(texts.loc[val_ids].tolist()),
            "test": route_predictor.predict(texts.loc[test_ids].tolist()),
        }
        probe_labels_by_split = {
            "val": probe_codebook.predict(probe_x_val),
            "test": probe_codebook.predict(probe_x_test),
        }
        for split, ids in [("val", val_ids), ("test", test_ids)]:
            pairs = zip(route_labels_by_split[split], probe_labels_by_split[split])
            selected = pd.Series(
                [pair_to_action.get((int(route_label), int(probe_label)), fallback) for route_label, probe_label in pairs],
                index=ids,
            )
            rows.append(
                metric_row(
                    matrix,
                    ids,
                    selected,
                    f"text_routecode_plus_probe_state_k{k}",
                    "text_routecode_plus_probe_state",
                    split,
                    scenario,
                    heldout,
                    args,
                    k=int(k),
                )
            )
    return rows


def select_by_validation(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for (scenario, heldout, family), group in candidates.groupby(["scenario", "heldout_benchmark", "family"], dropna=False):
        val = group[group["eval_split"].eq("val")].copy()
        test = group[group["eval_split"].eq("test")].copy()
        if val.empty:
            continue
        best = val.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).iloc[0].copy()
        best["selection_rule"] = "val_best_mean_utility"
        rows.append(best)
        match = test[test["method"].astype(str).eq(str(best["method"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = "val_best_mean_utility_test"
            rows.append(test_row)
    return pd.DataFrame(rows).sort_values(["scenario", "heldout_benchmark", "family", "eval_split"])


def metric_row(
    matrix: dict[str, Any],
    ids: list[str],
    selected: pd.Series,
    method: str,
    family: str,
    eval_split: str,
    scenario: str,
    heldout: str,
    args: argparse.Namespace,
    **extra: Any,
) -> dict[str, Any]:
    ids = [str(item) for item in ids]
    selected = selected.reindex(ids).astype(str)
    utility = matrix["utility"].reindex(ids)
    quality = matrix["quality"].reindex(ids)
    cost = matrix["cost"].reindex(ids)
    frontier = matrix["frontier"].reindex(ids)
    selected_utility = np.array([float(utility.loc[qid, model]) for qid, model in selected.items()])
    selected_quality = np.array([float(quality.loc[qid, model]) for qid, model in selected.items()])
    selected_cost = np.array([float(cost.loc[qid, model]) for qid, model in selected.items()])
    selected_frontier = np.array([bool(frontier.loc[qid, model]) for qid, model in selected.items()])
    oracle_utility = matrix["oracle_utility"].reindex(ids).astype(float).to_numpy()
    oracle_quality = matrix["oracle_quality"].reindex(ids).astype(float).to_numpy()
    ci_low, ci_high = bootstrap_ci(selected_utility, int(args.bootstrap_samples), int(args.seed))
    return {
        "scenario": scenario,
        "heldout_benchmark": heldout,
        "method": method,
        "family": family,
        "eval_split": eval_split,
        "n_queries": int(len(ids)),
        "mean_quality": float(selected_quality.mean()) if len(ids) else float("nan"),
        "mean_utility": float(selected_utility.mean()) if len(ids) else float("nan"),
        "mean_utility_ci_low": ci_low,
        "mean_utility_ci_high": ci_high,
        "mean_normalized_cost": float(selected_cost.mean()) if len(ids) else float("nan"),
        "oracle_mean_quality": float(oracle_quality.mean()) if len(ids) else float("nan"),
        "oracle_mean_utility": float(oracle_utility.mean()) if len(ids) else float("nan"),
        "oracle_utility_ratio": float(selected_utility.mean() / max(float(oracle_utility.mean()), 1e-12)) if len(ids) else float("nan"),
        "utility_gap_to_oracle": float(oracle_utility.mean() - selected_utility.mean()) if len(ids) else float("nan"),
        "quality_gap_to_oracle": float(oracle_quality.mean() - selected_quality.mean()) if len(ids) else float("nan"),
        "frontier_call_rate": float(selected_frontier.mean()) if len(ids) else float("nan"),
        "selected_models_json": json.dumps(selected.value_counts().sort_index().to_dict(), sort_keys=True),
        **extra,
    }


def build_code_cards(features: pd.DataFrame, matrix: dict[str, Any], selected_probe_state: dict[str, Any] | None, args: argparse.Namespace) -> pd.DataFrame:
    if not selected_probe_state:
        return pd.DataFrame()
    k = int(selected_probe_state["k"])
    train_ids = selected_probe_state["train_ids"]
    cols = feature_columns(features)
    feature_index = features.set_index("query_id")
    x_train = feature_index.loc[train_ids, cols].to_numpy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
    kmeans = KMeans(n_clusters=k, random_state=int(args.seed), n_init=KMEANS_N_INIT)
    train_labels = kmeans.fit_predict(x_train_scaled)
    all_x = scaler.transform(imputer.transform(feature_index[cols].to_numpy()))
    all_labels = kmeans.predict(all_x)
    label_by_query = pd.Series(all_labels, index=feature_index.index)
    action_by_label = best_action_by_label(matrix, train_ids, train_labels)
    global_means = pd.Series(x_train_scaled.mean(axis=0), index=cols)
    rows: list[dict[str, Any]] = []
    for label in range(k):
        ids = label_by_query[label_by_query.eq(label)].index.astype(str).tolist()
        train_label_ids = [qid for qid, z in zip(train_ids, train_labels) if int(z) == label]
        if train_label_ids:
            label_x = pd.DataFrame(x_train_scaled[[i for i, z in enumerate(train_labels) if int(z) == label]], columns=cols)
            top_features = (label_x.mean(axis=0) - global_means).abs().sort_values(ascending=False).head(8)
        else:
            top_features = pd.Series(dtype=float)
        train_utility = matrix["utility"].loc[train_label_ids] if train_label_ids else pd.DataFrame(columns=matrix["model_ids"])
        best_action = action_by_label.get(label, best_action_for_ids(matrix, train_ids))
        rows.append(
            {
                "probe_state": int(label),
                "n_queries_all_splits": int(len(ids)),
                "n_train_queries": int(len(train_label_ids)),
                "selected_action": best_action,
                "train_mean_selected_utility": float(train_utility[best_action].mean()) if train_label_ids else float("nan"),
                "train_frontier_rate_if_selected": float(matrix["frontier"].loc[train_label_ids, best_action].mean()) if train_label_ids else 0.0,
                "top_features_json": json.dumps(top_features.round(3).to_dict(), sort_keys=True),
                "benchmark_mix_json": json.dumps(features.set_index("query_id").loc[ids, "benchmark"].value_counts().sort_index().to_dict(), sort_keys=True),
                "description": describe_state(top_features.index.tolist(), best_action),
            }
        )
    return pd.DataFrame(rows)


def describe_state(top_features: list[str], action: str) -> str:
    text = " / ".join(top_features[:4]) if top_features else "low-signal"
    return f"{text} -> {action}"


def best_action_by_label(matrix: dict[str, Any], ids: list[str], labels: np.ndarray) -> dict[int, str]:
    frame = matrix["utility"].loc[ids].copy()
    frame["_label"] = labels
    mapping: dict[int, str] = {}
    for label, group in frame.groupby("_label"):
        means = group.drop(columns=["_label"]).mean(axis=0)
        mapping[int(label)] = str(means.idxmax())
    return mapping


def best_action_by_pair(matrix: dict[str, Any], ids: list[str], left_labels: np.ndarray, right_labels: np.ndarray) -> dict[tuple[int, int], str]:
    frame = matrix["utility"].loc[ids].copy()
    frame["_left_label"] = left_labels
    frame["_right_label"] = right_labels
    mapping: dict[tuple[int, int], str] = {}
    for pair, group in frame.groupby(["_left_label", "_right_label"]):
        means = group.drop(columns=["_left_label", "_right_label"]).mean(axis=0)
        mapping[(int(pair[0]), int(pair[1]))] = str(means.idxmax())
    return mapping


def best_action_for_ids(matrix: dict[str, Any], ids: list[str], candidate_models: list[str] | None = None) -> str:
    frame = matrix["utility"].loc[ids]
    if candidate_models is not None:
        frame = frame[candidate_models]
    return str(frame.mean(axis=0).idxmax())


def actions_from_scores(scores: np.ndarray, model_ids: list[str], ids: list[str]) -> pd.Series:
    idx = np.asarray(scores).argmax(axis=1)
    return pd.Series([model_ids[int(i)] for i in idx], index=ids)


def probe_matrix(
    features: pd.DataFrame,
    cols: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    *,
    include_benchmark_id: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = features.set_index("query_id")
    train = index.loc[train_ids, cols].copy()
    val = index.loc[val_ids, cols].copy()
    test = index.loc[test_ids, cols].copy()
    if include_benchmark_id:
        train_benchmarks = sorted(index.loc[train_ids, "benchmark"].astype(str).unique().tolist())
        for benchmark in train_benchmarks:
            col = f"benchmark_id_{benchmark}"
            train[col] = (index.loc[train_ids, "benchmark"].astype(str) == benchmark).astype(float).to_numpy()
            val[col] = (index.loc[val_ids, "benchmark"].astype(str) == benchmark).astype(float).to_numpy()
            test[col] = (index.loc[test_ids, "benchmark"].astype(str) == benchmark).astype(float).to_numpy()
    return train.to_numpy(), val.to_numpy(), test.to_numpy()


def feature_columns(features: pd.DataFrame) -> list[str]:
    blocked = {"query_id", "query_text", "split", "benchmark", "domain", "metric"}
    return [
        col
        for col in features.columns
        if col not in blocked and pd.api.types.is_numeric_dtype(features[col])
    ]


def normalize_answer(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    return text.removeprefix("answer:").strip().strip("$").strip()


def short_model_name(model_id: str) -> str:
    return (
        str(model_id)
        .replace("deterministic_math_tool", "tool")
        .replace("qwen3-", "qwen")
        .replace("-awq", "")
        .replace("-local", "")
        .replace("-selfconsistency-n3", "_sc")
        .replace("-", "_")
    )


def agree(answers: dict[str, str], left: str, right: str) -> float:
    a = answers.get(left, "")
    b = answers.get(right, "")
    return float(bool(a and b and a == b))


def answer_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_code_cards_md(path: Path, cards: pd.DataFrame) -> None:
    if cards.empty:
        path.write_text("# Probe-State Code Cards\n\nNo selected probe-state cards were produced.\n", encoding="utf-8")
        return
    lines = ["# Probe-State Code Cards", ""]
    for row in cards.itertuples(index=False):
        lines.extend(
            [
                f"## Probe State {row.probe_state}",
                "",
                f"- Selected action: `{row.selected_action}`",
                f"- Train queries: `{row.n_train_queries}`",
                f"- Description: {row.description}",
                f"- Top standardized feature deviations: `{row.top_features_json}`",
                f"- Benchmark mix: `{row.benchmark_mix_json}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_memo(
    path: Path,
    args: argparse.Namespace,
    features: pd.DataFrame,
    selected: pd.DataFrame,
    transfer_selected: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    cards: pd.DataFrame,
) -> None:
    selected_cols = [
        "family",
        "method",
        "eval_split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "mean_normalized_cost",
        "frontier_call_rate",
        "selection_rule",
    ]
    transfer_cols = [
        "heldout_benchmark",
        "family",
        "method",
        "eval_split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "selection_rule",
    ]
    transfer_test = (
        transfer_selected[transfer_selected["eval_split"].eq("test")].copy()
        if not transfer_selected.empty
        else transfer_selected
    )
    transfer_takeaway = transfer_comparison_takeaway(transfer_summary)
    lines = [
        "# Probe-State RouteCode",
        "",
        "This cached Broad100 experiment tests benchmark-agnostic probe states built from cheap local behavior.",
        "No provider, local generation, vLLM serving, or benchmark-specific checker calls are made.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/201_benchmark_agnostic_probe_state_routecode.py",
        "```",
        "",
        "## Probe Features",
        "",
        f"- Rows: `{len(features)}`",
        f"- Numeric feature count: `{len(feature_columns(features))}`",
        "- Feature families: local answer agreement, small-vs-medium disagreement, validity/malformed proxies,",
        "  self-consistency entropy/margins, output length/latency, and cached logprob margins when available.",
        "",
        "## Standard Split Selected Rows",
        "",
        markdown_table(selected[selected["eval_split"].isin(["val", "test"])][selected_cols]),
        "",
        "## Benchmark-Heldout Transfer Summary",
        "",
        markdown_table(transfer_summary) if not transfer_summary.empty else "No transfer rows.",
        "",
        transfer_takeaway,
        "",
        "## Benchmark-Heldout Selected Test Rows",
        "",
        markdown_table(transfer_test[transfer_cols]) if not transfer_test.empty else "No transfer rows.",
        "",
        "## Interpretation",
        "",
        "- Main probe-state rows exclude benchmark ID; benchmark-ID rows are diagnostic.",
        "- `text_routecode_plus_probe_state` combines a train-fit utility RouteCode label predicted from text with a train-fit probe-state label.",
        "- `oracle_local_vs_large_gate_upper_bound` is a diagnostic fixed-action gate: it chooses between the train-best local action and train-best frontier action using held-out true utility.",
        "- Benchmark-heldout rows fit state/action tables on other benchmarks only, select hyperparameters on their validation rows, and evaluate on the held-out benchmark test split.",
        "- The important comparison is probe-state transfer against benchmark lookup and text-only routing, not repairing one benchmark with custom checkers.",
        "",
        "## Artifacts",
        "",
        f"- Probe features: `{path.parent / 'table_probe_state_features.csv'}`",
        f"- All standard policies: `{path.parent / 'table_probe_state_policy_all.csv'}`",
        f"- Selected standard policies: `{path.parent / 'table_probe_state_policy_selected.csv'}`",
        f"- Benchmark-heldout all rows: `{path.parent / 'table_probe_state_benchmark_heldout_all.csv'}`",
        f"- Benchmark-heldout selected rows: `{path.parent / 'table_probe_state_benchmark_heldout_selected.csv'}`",
        f"- Benchmark-heldout summary: `{path.parent / 'table_probe_state_benchmark_heldout_summary.csv'}`",
        f"- Backward-compatible benchmark-heldout selected table: `{path.parent / 'table_probe_state_benchmark_heldout.csv'}`",
        f"- Probe-state code cards: `{path.parent / 'probe_state_code_cards.md'}`",
        f"- Code-card table: `{path.parent / 'table_probe_state_code_cards.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def transfer_comparison_takeaway(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No benchmark-heldout transfer comparison was produced."
    utility_by_family = summary.set_index("family")["mean_heldout_utility"].to_dict()
    probe = utility_by_family.get("probe_state_kmeans")
    text = utility_by_family.get("text_only_utility_router")
    benchmark = utility_by_family.get("benchmark_lookup")
    direct_probe = utility_by_family.get("direct_probe_utility_router")
    parts = []
    if probe is not None and text is not None:
        parts.append(f"probe-state KMeans beats text-only by {probe - text:+.4f} mean utility")
    if probe is not None and benchmark is not None:
        parts.append(f"probe-state KMeans beats benchmark lookup by {probe - benchmark:+.4f}")
    if direct_probe is not None and text is not None:
        parts.append(f"direct probe utility beats text-only by {direct_probe - text:+.4f}")
    if not parts:
        return "Benchmark-heldout transfer comparison could not find the expected family rows."
    return "Transfer comparison: " + "; ".join(parts) + "."


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
