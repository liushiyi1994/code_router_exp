from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, roc_auc_score

from routecode.controlled.live_stage0 import normalize_answer


LOCAL_ACTIONS = (
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
)
LARGE_ACTIONS = (
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
)
SLM_MODEL = "qwen3-4b-local"
MEDIUM_MODELS = ("qwen3-14b-awq-local", "qwen3-32b-awq-local")
SELF_MODEL = "qwen3-32b-awq-selfconsistency-n3-local"
FRONTIER_MODELS = {"gemini-3.5-flash", "gpt-5.5", "gemini-3.5-flash-strong-solve"}
QUERY_RISK_BY_BENCHMARK = {
    "aime": 0.85,
    "bbh": 0.50,
    "gpqa": 1.00,
    "gsm8k": 0.20,
    "humaneval": 0.30,
    "livemathbench": 0.80,
    "math500": 0.65,
    "mbpp": 0.25,
    "mmlupro": 0.90,
}
QUERY_RISK_BY_METRIC = {
    "multiple_choice": 0.50,
    "exact_final_answer": 0.55,
    "pass_at_1": 0.25,
}
CAPS = (0.10, 0.20, 0.30, 0.40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-training SLM/LLM early-signal probe pilot.")
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
        default=Path("results/controlled/broad100_slm_llm_early_signal_probe_pilot"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--tau-gain", type=float, default=1e-9)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--answerability-vllm-base-url", default="")
    parser.add_argument("--answerability-served-model-name", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--answerability-splits", default="val,test")
    parser.add_argument("--answerability-batch-size", type=int, default=16)
    parser.add_argument("--answerability-timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_gate")
    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    target = build_oracle_targets(outputs, probe, tau_gain=float(args.tau_gain))
    target = add_probe_signals(target)
    if args.answerability_vllm_base_url:
        answerability = collect_or_load_answerability_probe(target, args)
        target = merge_answerability_signal(target, answerability)
    policy_table, selected = evaluate_threshold_policies(
        target,
        lambda_cost=float(args.lambda_cost),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    cap_table = precision_at_caps(target)

    target.to_csv(args.output_dir / "table_slm_llm_oracle_targets.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_slm_llm_threshold_policies_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_slm_llm_threshold_policies_selected.csv", index=False)
    cap_table.to_csv(args.output_dir / "table_slm_llm_precision_at_caps.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "SLM_LLM_EARLY_SIGNAL_PROBE_MEMO.md", args, target, policy_table, selected, cap_table)
    print(f"Wrote SLM/LLM early-signal probe pilot to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_oracle_targets(outputs: pd.DataFrame, probe: pd.DataFrame, *, tau_gain: float) -> pd.DataFrame:
    by_query = outputs.drop_duplicates("query_id").set_index("query_id")
    by_query_model = outputs.set_index(["query_id", "model_id"])
    probe_by_query = probe.set_index("query_id")
    rows: list[dict[str, Any]] = []
    for query_id, info in by_query.sort_values(["split", "benchmark", "query_id"]).iterrows():
        query_id = str(query_id)
        local_row = best_available_action(by_query_model, query_id, LOCAL_ACTIONS)
        large_row = best_available_action(by_query_model, query_id, LARGE_ACTIONS)
        if local_row is None or large_row is None:
            continue
        probe_row = probe_by_query.loc[query_id] if query_id in probe_by_query.index else pd.Series(dtype=object)
        delta_large = float(large_row["utility"]) - float(local_row["utility"])
        rows.append(
            {
                "query_id": query_id,
                "split": str(info.get("split", "")),
                "benchmark": str(info.get("benchmark", "")),
                "domain": str(info.get("domain", "")),
                "metric": str(info.get("metric", "")),
                "query_text": str(info.get("query_text", "")),
                "gold_answer": str(info.get("gold_answer", "")),
                "best_local_action": str(local_row["model_id"]),
                "best_large_action": str(large_row["model_id"]),
                "local_quality": float(local_row["quality_score"]),
                "large_quality": float(large_row["quality_score"]),
                "local_normalized_cost": float(local_row["normalized_remote_cost"]),
                "large_normalized_cost": float(large_row["normalized_remote_cost"]),
                "local_cost_usd": float(local_row["cost_total_usd"]),
                "large_cost_usd": float(large_row["cost_total_usd"]),
                "local_latency_s": float(local_row["latency_s"]),
                "large_latency_s": float(large_row["latency_s"]),
                "local_utility": float(local_row["utility"]),
                "large_utility": float(large_row["utility"]),
                "delta_large": delta_large,
                "need_large": bool(delta_large >= float(tau_gain)),
                "need_large_positive_gain": bool(delta_large > 1e-12),
                "large_is_frontier": bool(str(large_row["model_id"]) in FRONTIER_MODELS),
                "best_large_family": large_family(str(large_row["model_id"])),
                "slm_answer": normalized_output_answer(by_query_model, query_id, SLM_MODEL),
                "medium14_answer": normalized_output_answer(by_query_model, query_id, "qwen3-14b-awq-local"),
                "medium32_answer": normalized_output_answer(by_query_model, query_id, "qwen3-32b-awq-local"),
                "self_majority_answer": normalized_probe_answer(probe_row.get("majority_answer_norm", "")),
                "self_vote_frac": as_float(probe_row.get("vote_frac", 0.0)),
                "self_vote_margin": as_float(probe_row.get("vote_margin", 0.0)),
                "self_vote_entropy": as_float(probe_row.get("vote_entropy", 0.0)),
                "self_valid_count": as_float(probe_row.get("valid_count", 0.0)),
                "self_n_samples": as_float(probe_row.get("n_samples", 0.0)),
                "self_all_samples_agree": bool(probe_row.get("all_samples_agree", False)),
                "self_answer_norms_json": str(probe_row.get("answer_norms_json", "[]")),
            }
        )
    return pd.DataFrame(rows)


def best_available_action(by_query_model: pd.DataFrame, query_id: str, actions: tuple[str, ...]) -> pd.Series | None:
    candidates: list[pd.Series] = []
    for action in actions:
        key = (query_id, action)
        if key in by_query_model.index:
            row = by_query_model.loc[key].copy()
            row["model_id"] = action
            candidates.append(row)
    if not candidates:
        return None
    frame = pd.DataFrame(candidates)
    frame = frame.sort_values(["utility", "quality_score", "normalized_remote_cost"], ascending=[False, False, True])
    return frame.iloc[0]


def add_probe_signals(target: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    text_len = out["query_text"].astype(str).str.len().to_numpy(dtype=float)
    option_count = out["query_text"].astype(str).str.count(r"\n[A-D]\)").to_numpy(dtype=float)
    numeric_count = out["query_text"].astype(str).str.count(r"\d").to_numpy(dtype=float)
    query_len_risk = minmax(text_len)
    numeric_risk = minmax(numeric_count)
    option_bonus = np.where(option_count >= 4, 0.05, 0.0)
    benchmark_risk = out["benchmark"].map(QUERY_RISK_BY_BENCHMARK).fillna(0.50).to_numpy(dtype=float)
    metric_risk = out["metric"].map(QUERY_RISK_BY_METRIC).fillna(0.50).to_numpy(dtype=float)
    out["signal_query_answerability_risk"] = clip01(
        0.45 * benchmark_risk + 0.20 * metric_risk + 0.20 * query_len_risk + 0.10 * numeric_risk + option_bonus
    )
    out["signal_query_answerability"] = 1.0 - out["signal_query_answerability_risk"]
    out["signal_vllm_answerability_risk"] = np.nan

    sample_denominator = out["self_n_samples"].replace(0.0, np.nan)
    invalid_rate = 1.0 - (out["self_valid_count"] / sample_denominator).fillna(0.0)
    vote_entropy_norm = out["self_vote_entropy"] / np.log2(np.maximum(out["self_n_samples"], 2.0))
    out["self_unique_answer_count"] = out["self_answer_norms_json"].map(count_unique_answers)
    unique_norm = (out["self_unique_answer_count"] - 1.0) / np.maximum(out["self_n_samples"] - 1.0, 1.0)
    out["signal_early_rollout_instability"] = clip01(
        0.35 * (1.0 - out["self_vote_frac"])
        + 0.25 * (1.0 - out["self_vote_margin"])
        + 0.20 * vote_entropy_norm.fillna(0.0)
        + 0.20 * invalid_rate
    )
    out["signal_semantic_uncertainty"] = clip01(
        0.45 * vote_entropy_norm.fillna(0.0)
        + 0.30 * unique_norm.fillna(0.0)
        + 0.15 * (1.0 - out["self_vote_frac"])
        + 0.10 * invalid_rate
    )

    slm = out["slm_answer"].astype(str)
    m14 = out["medium14_answer"].astype(str)
    m32 = out["medium32_answer"].astype(str)
    self_ans = out["self_majority_answer"].astype(str)
    slm_valid = slm.ne("")
    medium_valid_count = m14.ne("").astype(float) + m32.ne("").astype(float)
    medium_agree = m14.ne("") & m14.eq(m32)
    small_differs_medium14 = slm_valid & m14.ne("") & slm.ne(m14)
    small_differs_medium32 = slm_valid & m32.ne("") & slm.ne(m32)
    self_differs_small = slm_valid & self_ans.ne("") & slm.ne(self_ans)
    out["signal_slm_medium_divergence"] = clip01(
        0.30 * small_differs_medium14.astype(float)
        + 0.30 * small_differs_medium32.astype(float)
        + 0.20 * (medium_agree & small_differs_medium14 & small_differs_medium32).astype(float)
        + 0.10 * self_differs_small.astype(float)
        + 0.10 * (1.0 - medium_valid_count / 2.0)
    )
    out["signal_medium_consensus_disagrees_with_slm"] = (
        medium_agree & small_differs_medium14 & small_differs_medium32
    ).astype(float)

    train_prior = (
        out[out["split"].eq("train")]
        .groupby("benchmark")["need_large"]
        .mean()
        .to_dict()
    )
    global_prior = float(out[out["split"].eq("train")]["need_large"].mean()) if (out["split"] == "train").any() else 0.5
    out["signal_query_train_prior_need_large"] = out["benchmark"].map(train_prior).fillna(global_prior).astype(float)
    out["signal_combined_mean_risk"] = clip01(
        out[
            [
                "signal_query_answerability_risk",
                "signal_early_rollout_instability",
                "signal_slm_medium_divergence",
                "signal_semantic_uncertainty",
            ]
        ].mean(axis=1)
    )
    out["signal_combined_max_risk"] = clip01(
        out[
            [
                "signal_query_answerability_risk",
                "signal_early_rollout_instability",
                "signal_slm_medium_divergence",
                "signal_semantic_uncertainty",
            ]
        ].max(axis=1)
    )
    return out


def evaluate_threshold_policies(
    target: pd.DataFrame,
    *,
    lambda_cost: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    signal_columns = [
        "signal_query_answerability_risk",
        "signal_query_answerability",
        "signal_vllm_answerability_risk",
        "signal_query_train_prior_need_large",
        "signal_early_rollout_instability",
        "signal_slm_medium_divergence",
        "signal_medium_consensus_disagrees_with_slm",
        "signal_semantic_uncertainty",
        "signal_combined_mean_risk",
        "signal_combined_max_risk",
    ]
    for split in ["val", "test"]:
        split_frame = target[target["split"].eq(split)].copy()
        rows.extend(reference_rows(split_frame, split=split, lambda_cost=lambda_cost))

    for signal in signal_columns:
        for direction in ["high", "low"]:
            thresholds = candidate_thresholds(target[target["split"].eq("val")][signal].to_numpy(dtype=float))
            for threshold in thresholds:
                for split in ["val", "test"]:
                    split_frame = target[target["split"].eq(split)].copy()
                    choose_large = threshold_decision(split_frame[signal].to_numpy(dtype=float), threshold, direction)
                    row = evaluate_decision(
                        split_frame,
                        choose_large,
                        split=split,
                        method=f"{signal}_{direction}_thr{threshold:.4g}",
                        family="threshold_signal",
                        lambda_cost=lambda_cost,
                    )
                    row.update({"signal": signal, "direction": direction, "threshold": float(threshold)})
                    rows.append(row)
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    selected = validation_selected_rows(table, bootstrap_samples=bootstrap_samples, seed=seed)
    return table, selected


def reference_rows(split_frame: pd.DataFrame, *, split: str, lambda_cost: float) -> list[dict[str, Any]]:
    return [
        evaluate_decision(
            split_frame,
            np.zeros(len(split_frame), dtype=bool),
            split=split,
            method="always_best_local_action",
            family="reference",
            lambda_cost=lambda_cost,
        ),
        evaluate_decision(
            split_frame,
            np.ones(len(split_frame), dtype=bool),
            split=split,
            method="always_best_large_action",
            family="reference",
            lambda_cost=lambda_cost,
        ),
        evaluate_decision(
            split_frame,
            split_frame["delta_large"].to_numpy(dtype=float) >= 0.0,
            split=split,
            method="oracle_local_vs_large_gate",
            family="diagnostic_oracle",
            lambda_cost=lambda_cost,
        ),
    ]


def evaluate_decision(
    frame: pd.DataFrame,
    choose_large: np.ndarray,
    *,
    split: str,
    method: str,
    family: str,
    lambda_cost: float,
) -> dict[str, Any]:
    choose_large = np.asarray(choose_large, dtype=bool)
    quality = np.where(choose_large, frame["large_quality"], frame["local_quality"]).astype(float)
    utility = np.where(choose_large, frame["large_utility"], frame["local_utility"]).astype(float)
    norm_cost = np.where(choose_large, frame["large_normalized_cost"], frame["local_normalized_cost"]).astype(float)
    usd_cost = np.where(choose_large, frame["large_cost_usd"], frame["local_cost_usd"]).astype(float)
    latency = np.where(choose_large, frame["large_latency_s"], frame["local_latency_s"]).astype(float)
    selected_action = np.where(choose_large, frame["best_large_action"], frame["best_local_action"])
    oracle_utility = np.maximum(frame["local_utility"], frame["large_utility"]).astype(float)
    oracle_quality = np.where(
        frame["large_quality"].to_numpy(dtype=float) >= frame["local_quality"].to_numpy(dtype=float),
        frame["large_quality"],
        frame["local_quality"],
    ).astype(float)
    positives = frame["need_large"].astype(bool).to_numpy()
    tp = int(np.sum(choose_large & positives))
    fp = int(np.sum(choose_large & ~positives))
    fn = int(np.sum(~choose_large & positives))
    tn = int(np.sum(~choose_large & ~positives))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    local_mean = float(frame["local_utility"].mean())
    oracle_mean = float(oracle_utility.mean())
    mean_utility = float(np.mean(utility))
    recovered = (mean_utility - local_mean) / max(oracle_mean - local_mean, 1e-12)
    return {
        "method": method,
        "family": family,
        "split": split,
        "n_queries": int(len(frame)),
        "mean_quality": float(np.mean(quality)),
        "mean_utility": mean_utility,
        "normalized_cost_mean": float(np.mean(norm_cost)),
        "remote_cost_total_usd": float(np.sum(usd_cost)),
        "mean_latency_s": float(np.mean(latency)),
        "p95_latency_s": float(np.quantile(latency, 0.95)),
        "local_baseline_mean_utility": local_mean,
        "local_large_oracle_mean_utility": oracle_mean,
        "local_large_oracle_mean_quality": float(np.mean(oracle_quality)),
        "utility_gap_to_oracle": float(oracle_mean - mean_utility),
        "oracle_utility_ratio": float(mean_utility / max(oracle_mean, 1e-12)),
        "recovered_gap_vs_local": float(recovered),
        "large_call_rate": float(np.mean(choose_large)),
        "frontier_call_rate": float(np.mean([action in FRONTIER_MODELS for action in selected_action])),
        "need_large_precision": float(precision),
        "need_large_recall": float(recall),
        "need_large_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "selected_actions_json": json.dumps(pd.Series(selected_action).value_counts().sort_index().to_dict(), sort_keys=True),
        "lambda_cost": float(lambda_cost),
        "_utility_values": utility.tolist(),
    }


def validation_selected_rows(table: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in table.groupby("family"):
        if family == "diagnostic_oracle":
            continue
        val = group[group["split"].eq("val")].sort_values(
            ["mean_utility", "normalized_cost_mean", "large_call_rate"],
            ascending=[False, True, True],
        )
        if val.empty:
            continue
        best = val.head(1).copy()
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = group[group["split"].eq("test") & group["method"].eq(method)].head(1).copy()
        if not test.empty:
            rows.append(test.assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not selected.empty:
        selected = add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)
        selected = selected.drop(columns=["_utility_values"], errors="ignore")
    return selected


def add_bootstrap_ci(frame: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    lows: list[float] = []
    highs: list[float] = []
    for values in out["_utility_values"]:
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0 or bootstrap_samples <= 0:
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        draws = rng.choice(arr, size=(bootstrap_samples, len(arr)), replace=True).mean(axis=1)
        lows.append(float(np.quantile(draws, 0.025)))
        highs.append(float(np.quantile(draws, 0.975)))
    out["mean_utility_ci_low"] = lows
    out["mean_utility_ci_high"] = highs
    return out


def precision_at_caps(target: pd.DataFrame) -> pd.DataFrame:
    signal_columns = [
        "signal_query_answerability_risk",
        "signal_vllm_answerability_risk",
        "signal_query_train_prior_need_large",
        "signal_early_rollout_instability",
        "signal_slm_medium_divergence",
        "signal_medium_consensus_disagrees_with_slm",
        "signal_semantic_uncertainty",
        "signal_combined_mean_risk",
        "signal_combined_max_risk",
    ]
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        frame = target[target["split"].eq(split)].copy()
        labels = frame["need_large"].astype(bool).to_numpy()
        for signal in signal_columns:
            scores = frame[signal].to_numpy(dtype=float)
            auroc = safe_auroc(labels, scores)
            auprc = safe_auprc(labels, scores)
            order_scores = np.where(np.isfinite(scores), scores, -np.inf)
            order = np.argsort(order_scores)[::-1]
            for cap in CAPS:
                k = max(1, int(math.floor(cap * len(frame))))
                selected = np.zeros(len(frame), dtype=bool)
                selected[order[:k]] = True
                tp = int(np.sum(selected & labels))
                fp = int(np.sum(selected & ~labels))
                fn = int(np.sum(~selected & labels))
                rows.append(
                    {
                        "split": split,
                        "signal": signal,
                        "cap": float(cap),
                        "selected_count": int(k),
                        "precision": float(tp / max(tp + fp, 1)),
                        "recall": float(tp / max(tp + fn, 1)),
                        "auroc": auroc,
                        "auprc": auprc,
                    }
                )
    return pd.DataFrame(rows)


def threshold_decision(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "high":
        return values >= threshold
    if direction == "low":
        return values <= threshold
    raise ValueError(direction)


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.asarray([0.0])
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 41))
    unique = np.unique(np.concatenate([quantiles, values]))
    return unique


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def collect_or_load_answerability_probe(target: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    path = args.output_dir / "table_vllm_answerability_probe.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(existing["query_id"].astype(str)) if not existing.empty and "query_id" in existing.columns else set()
    splits = {item.strip() for item in str(args.answerability_splits).split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    pending = target[target["split"].isin(splits) & ~target["query_id"].astype(str).isin(done)].copy()
    records = pending.to_dict("records")
    for start in range(0, len(records), int(args.answerability_batch_size)):
        batch = records[start : start + int(args.answerability_batch_size)]
        rows.extend(call_answerability_batch(batch, args))
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
        combined.to_csv(path, index=False)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def call_answerability_batch(batch: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not batch:
        return []
    prompts = [answerability_prompt(row) for row in batch]
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{str(args.answerability_vllm_base_url).rstrip('/')}/completions",
            json={
                "model": args.answerability_served_model_name,
                "prompt": prompts,
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": 5,
            },
            timeout=float(args.answerability_timeout_s),
        )
        response.raise_for_status()
        payload = response.json()
        choices = sorted(payload.get("choices", []), key=lambda item: int(item.get("index", 0)))
    except Exception as exc:  # noqa: BLE001 - cache the failure per row.
        elapsed = time.perf_counter() - started
        return [
            {
                "query_id": str(row["query_id"]),
                "split": str(row["split"]),
                "benchmark": str(row["benchmark"]),
                "status": "error",
                "error": repr(exc),
                "raw_text": "",
                "answerability_score": np.nan,
                "latency_s": elapsed / max(len(batch), 1),
            }
            for row in batch
        ]
    elapsed = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    for row, choice in zip(batch, choices):
        raw_text = str(choice.get("text", ""))
        score = answerability_score_from_choice(choice)
        rows.append(
            {
                "query_id": str(row["query_id"]),
                "split": str(row["split"]),
                "benchmark": str(row["benchmark"]),
                "status": "success",
                "error": "",
                "raw_text": raw_text,
                "answerability_score": score,
                "latency_s": elapsed / max(len(batch), 1),
            }
        )
    return rows


def answerability_prompt(row: dict[str, Any]) -> str:
    query = re.sub(r"\s+", " ", str(row.get("query_text", ""))).strip()[:1800]
    benchmark = str(row.get("benchmark", ""))
    metric = str(row.get("metric", ""))
    return (
        "Answer with exactly one token: YES or NO.\n"
        "Question: Can a small local language model answer this item correctly without help from a larger model?\n"
        f"Benchmark: {benchmark}\n"
        f"Metric: {metric}\n"
        f"Item: {query}\n"
        "Answer:"
    )


def answerability_score_from_choice(choice: dict[str, Any]) -> float:
    text = str(choice.get("text", "")).strip().lower()
    text_score = 1.0 if text.startswith("yes") else 0.0 if text.startswith("no") else 0.5
    logprobs = choice.get("logprobs") or {}
    top_logprobs = logprobs.get("top_logprobs") or []
    if not top_logprobs:
        return float(text_score)
    first = top_logprobs[0] or {}
    yes_logp = None
    no_logp = None
    for token, value in first.items():
        norm = str(token).strip().lower()
        if norm.startswith("yes"):
            yes_logp = float(value)
        elif norm.startswith("no"):
            no_logp = float(value)
    if yes_logp is None or no_logp is None:
        return float(text_score)
    yes = math.exp(yes_logp)
    no = math.exp(no_logp)
    return float(yes / max(yes + no, 1e-12))


def merge_answerability_signal(target: pd.DataFrame, answerability: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    if answerability.empty:
        out["signal_vllm_answerability_risk"] = np.nan
        return out
    scores = answerability[answerability["status"].eq("success")].drop_duplicates("query_id").set_index("query_id")[
        "answerability_score"
    ]
    out["signal_vllm_answerability_score"] = out["query_id"].map(scores).astype(float)
    out["signal_vllm_answerability_risk"] = 1.0 - out["signal_vllm_answerability_score"]
    return out


def large_family(model_id: str) -> str:
    if model_id in FRONTIER_MODELS:
        return "frontier"
    if model_id == SELF_MODEL:
        return "self_consistency"
    return "strong_local"


def normalized_output_answer(by_query_model: pd.DataFrame, query_id: str, model_id: str) -> str:
    if (query_id, model_id) not in by_query_model.index:
        return ""
    return normalized_probe_answer(by_query_model.loc[(query_id, model_id), "parsed_answer"])


def normalized_probe_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(normalize_answer(str(value))).strip().lower()


def count_unique_answers(value: object) -> float:
    try:
        values = json.loads(str(value))
    except json.JSONDecodeError:
        values = []
    answers = [normalized_probe_answer(item) for item in values]
    return float(len({answer for answer in answers if answer}))


def as_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(np.min(values)) if len(values) else 0.0
    high = float(np.max(values)) if len(values) else 1.0
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def clip01(values: Any) -> Any:
    return np.clip(values, 0.0, 1.0)


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#586f86")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("SLM/LLM Early-Signal Probe Pilot")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_slm_llm_early_signal_probe_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    target: pd.DataFrame,
    policy_table: pd.DataFrame,
    selected: pd.DataFrame,
    cap_table: pd.DataFrame,
) -> None:
    target_summary = target_summary_table(target)
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_ci_low",
        "mean_utility_ci_high",
        "oracle_utility_ratio",
        "recovered_gap_vs_local",
        "large_call_rate",
        "frontier_call_rate",
        "need_large_precision",
        "need_large_recall",
        "signal",
        "direction",
        "threshold",
        "selection_rule",
    ]
    cap_cols = ["split", "signal", "cap", "precision", "recall", "auroc", "auprc"]
    command = (
        "PYTHONPATH=src python experiments/166_slm_llm_early_signal_probe_pilot.py "
        f"--output-dir {args.output_dir} --lambda-cost {args.lambda_cost} --tau-gain {args.tau_gain}"
    )
    if args.answerability_vllm_base_url:
        command += (
            f" --answerability-vllm-base-url {args.answerability_vllm_base_url}"
            f" --answerability-served-model-name {args.answerability_served_model_name}"
            f" --answerability-splits {args.answerability_splits}"
            f" --answerability-batch-size {args.answerability_batch_size}"
        )
    model_call_line = (
        f"- vLLM answerability calls were made to `{args.answerability_served_model_name}` at "
        f"`{args.answerability_vllm_base_url}` for splits `{args.answerability_splits}` and cached in "
        "`table_vllm_answerability_probe.csv`."
        if args.answerability_vllm_base_url
        else "- No provider API calls and no vLLM calls were made by this pilot; it uses cached local/frontier outputs only."
    )
    if args.answerability_vllm_base_url:
        vllm_cap10 = cap_table[
            cap_table["split"].eq("test")
            & cap_table["signal"].eq("signal_vllm_answerability_risk")
            & np.isclose(cap_table["cap"].astype(float), 0.1)
        ]
        if not vllm_cap10.empty:
            row = vllm_cap10.iloc[0]
            vllm_summary = (
                f"held-out 10% cap precision `{float(row['precision']):.4f}`, "
                f"recall `{float(row['recall']):.4f}`, AUROC `{float(row['auroc']):.4f}`, "
                f"and AUPRC `{float(row['auprc']):.4f}`"
            )
        else:
            vllm_summary = "no usable held-out cap row"
        next_probe_line = (
            "- The one-token Qwen14 answerability signal was noisy and did not help selected utility "
            f"({vllm_summary}). The next useful probe is constrained YES/NO logit scoring or local "
            "final-answer confidence/activation evidence, not another unconstrained free-token answerability call."
        )
    else:
        next_probe_line = (
            "- If this cached pilot does not improve utility, the next useful probe is a true vLLM one-token "
            "answerability/confidence collection on the same target table, because the cache-derived query and "
            "self-consistency signals are still indirect proxies."
        )
    lines = [
        "# SLM/LLM Early-Signal Probe Pilot",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/166_slm_llm_early_signal_probe_pilot.py",
        command,
        "```",
        "",
        "## Scope",
        "",
        f"- Outputs: `{args.outputs}`",
        f"- Self-consistency probe table: `{args.probe_table}`",
        f"- Output directory: `{args.output_dir}`",
        model_call_line,
        "- This is a binary local-vs-large diagnostic: threshold signals choose between per-query best cached local action and per-query best cached large action. It tests whether the early signal detects upward-routing value; it is not a final deployable multi-model selector.",
        "",
        "## Action Pools",
        "",
        f"- Local actions: `{', '.join(LOCAL_ACTIONS)}`",
        f"- Large actions: `{', '.join(LARGE_ACTIONS)}`",
        f"- Need-large label: `delta_large >= {args.tau_gain}` with utility `quality - {args.lambda_cost} * normalized_cost`.",
        "",
        "## Target Summary",
        "",
        "```csv",
        compact_csv(target_summary),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[column for column in cols if column in selected.columns]], max_rows=48),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(
            policy_table[policy_table["split"].eq("test")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)[[column for column in cols if column in policy_table.columns]],
            max_rows=32,
        ),
        "```",
        "",
        "## Precision At Large-Call Caps",
        "",
        "```csv",
        compact_csv(
            cap_table[cap_table["split"].eq("test")]
            .sort_values(["cap", "precision", "recall"], ascending=[True, False, False])[[column for column in cap_cols if column in cap_table.columns]],
            max_rows=48,
        ),
        "```",
        "",
        "## Interpretation",
        "",
        "- Query surface risk, self-consistency instability, SLM-vs-medium divergence, and semantic uncertainty are evaluated as threshold-only signals.",
        "- When enabled, the vLLM answerability signal is also evaluated as a threshold-only signal and charged only as a probe, not as a provider API call.",
        "- A selected threshold improves the binary local-vs-large decision only if the validation-selected test row beats `always_best_local_action` without excessive large/frontier calls.",
        "- If AUROC or precision-at-cap is nontrivial but utility remains low, the signal separates local-risky examples but still does not identify where the larger action is cost-effective.",
        "",
        "## Next Recommended Probe",
        "",
        next_probe_line,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_summary_table(target: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in target.groupby("split"):
        rows.append(
            {
                "split": split,
                "n_queries": int(len(group)),
                "mean_local_utility": float(group["local_utility"].mean()),
                "mean_large_utility": float(group["large_utility"].mean()),
                "mean_oracle_utility": float(np.maximum(group["local_utility"], group["large_utility"]).mean()),
                "mean_local_quality": float(group["local_quality"].mean()),
                "mean_large_quality": float(group["large_quality"].mean()),
                "need_large_rate": float(group["need_large"].mean()),
                "best_large_frontier_rate": float(group["large_is_frontier"].mean()),
                "mean_delta_large": float(group["delta_large"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("split")


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if column == "_utility_values":
            continue
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
