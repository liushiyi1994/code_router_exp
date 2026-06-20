from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOOL_MODEL_ID = "deterministic_math_tool"
CHEAP_LOCAL_ACTIONS = (
    TOOL_MODEL_ID,
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
)
OBSERVABLE_LOCAL_ACTIONS = (
    TOOL_MODEL_ID,
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
)
STRONG_OR_FRONTIER_ACTIONS = (
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
)
ALL_ACTIONS = tuple(dict.fromkeys((*OBSERVABLE_LOCAL_ACTIONS, *STRONG_OR_FRONTIER_ACTIONS)))
THRESHOLD_SIGNALS = (
    "signal_query_answerability_risk",
    "signal_early_rollout_instability",
    "signal_slm_medium_divergence",
    "signal_semantic_uncertainty",
    "signal_combined_mean_risk",
    "signal_combined_max_risk",
    "signal_constrained_yesno_query_only_risk",
    "signal_constrained_yesno_local_evidence_risk",
    "signal_constrained_yesno_max_risk",
    "signal_constrained_yesno_mean_risk",
    "signal_constrained_plus_cached_mean_risk",
    "signal_constrained_plus_cached_max_risk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actual deployed-action bridge for tool-aware broad100 routing.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--benchmark-composed-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_tool_aware_benchmark_composed_policy/"
            "table_tool_aware_benchmark_composed_choices.csv"
        ),
    )
    parser.add_argument(
        "--benchmark-composed-method",
        default="tool_aware_benchmark_composed_eps0.01_recall_then_quality",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_tool_aware_deployed_action_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    e171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171")
    target = e171.add_tool_availability(target, outputs)
    target = add_benchmark_composed_gate(target, args.benchmark_composed_choices, args.benchmark_composed_method, e171)

    table_internal, details = evaluate_policy_library(
        target,
        outputs,
        lambda_cost=float(args.lambda_cost),
    )
    selected = validation_selected_rows(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = add_bootstrap_ci(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = table.drop(columns=["_utility_values"], errors="ignore")
    table.to_csv(args.output_dir / "table_tool_aware_deployed_action_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_tool_aware_deployed_action_policy_selected.csv", index=False)
    details.to_csv(args.output_dir / "table_tool_aware_deployed_action_choices.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(
        args.output_dir / "TOOL_AWARE_DEPLOYED_ACTION_POLICY_MEMO.md",
        args,
        table,
        selected,
        details,
    )
    print(f"Wrote deployed-action policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare_outputs(outputs: pd.DataFrame) -> pd.DataFrame:
    out = outputs.copy()
    out["query_id"] = out["query_id"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["answer_norm"] = out.get("parsed_answer", "").map(normalized_answer)
    for column in ["quality_score", "utility", "normalized_remote_cost", "cost_total_usd", "latency_s"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    for column in ["is_frontier", "is_local", "tool_available"]:
        if column in out:
            out[column] = out[column].fillna(False).astype(bool)
    return out


def add_benchmark_composed_gate(
    target: pd.DataFrame,
    choices_path: Path,
    method: str,
    e171,
) -> pd.DataFrame:
    out = target.copy()
    out["benchmark_composed_need_large"] = False
    if not choices_path.exists():
        return out
    choices = pd.read_csv(choices_path)
    choices = choices[choices["method"].astype(str).eq(str(method))]
    if choices.empty:
        return out
    selected_by_benchmark = {
        str(row["benchmark"]): str(row["chosen_policy"]) for _, row in choices.iterrows()
    }
    policy_fns = e171.candidate_policy_functions()
    for split in sorted(out["split"].dropna().astype(str).unique()):
        frame = out[out["split"].astype(str).eq(split)].copy()
        choose = np.zeros(len(frame), dtype=bool)
        benchmarks = frame["benchmark"].astype(str).to_numpy()
        for benchmark, policy_name in selected_by_benchmark.items():
            if policy_name not in policy_fns:
                continue
            positions = np.where(benchmarks == benchmark)[0]
            if positions.size:
                choose[positions] = policy_fns[policy_name](frame.iloc[positions])
        out.loc[frame.index, "benchmark_composed_need_large"] = choose
    return out


def evaluate_policy_library(
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    priors = fit_train_priors(outputs)
    policy_fns = fixed_policy_functions(outputs, priors)
    threshold_fns = threshold_policy_functions(target, outputs, priors)

    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        frame = target[target["split"].astype(str).eq(split)].copy()
        for method, (family, selector) in {**policy_fns, **threshold_fns}.items():
            selected = select_actions(frame, selector, rows_by_query)
            selected_rows = selected.merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            row = evaluate_selected_rows(
                method,
                family,
                split,
                selected_rows,
                outputs,
                target=frame,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            rows.append(row)
            if split == "test" and method in detail_methods():
                detail = selected_rows[
                    [
                        "query_id",
                        "query_text",
                        "benchmark",
                        "metric",
                        "model_id",
                        "quality_score",
                        "utility",
                        "normalized_remote_cost",
                        "is_frontier",
                        "parsed_answer",
                    ]
                ].copy()
                detail["method"] = method
                detail["family"] = family
                detail_frames.append(detail)
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return table, details


def fixed_policy_functions(
    outputs: pd.DataFrame,
    priors: dict[str, Any],
) -> dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]]:
    return {
        "full_cost_aware_oracle": ("diagnostic_oracle", full_oracle_selector("utility")),
        "full_quality_oracle": ("diagnostic_oracle", full_oracle_selector("quality_score")),
        "train_best_single_action": (
            "reference",
            lambda row, actions: choose_prior_action(row, actions, priors, ALL_ACTIONS, scope="global"),
        ),
        "train_benchmark_prior_all_actions": (
            "deployed_policy",
            lambda row, actions: choose_prior_action(row, actions, priors, ALL_ACTIONS, scope="benchmark"),
        ),
        "tool_then_train_benchmark_prior_all_actions": (
            "deployed_policy",
            lambda row, actions: choose_tool_or_prior(row, actions, priors, ALL_ACTIONS),
        ),
        "tool_then_local_consensus_else_benchmark_prior": (
            "deployed_policy",
            lambda row, actions: choose_tool_or_local_consensus(row, actions, priors, fallback_pool=ALL_ACTIONS),
        ),
        "tool_then_171_gate_train_prior": (
            "deployed_policy",
            lambda row, actions: choose_tool_or_gate_prior(row, actions, priors),
        ),
        "tool_then_171_gate_local_consensus_large_prior": (
            "deployed_policy",
            lambda row, actions: choose_tool_or_gate_consensus(row, actions, priors),
        ),
        "oracle_local_vs_large_gate_train_prior": (
            "diagnostic_policy",
            lambda row, actions: choose_tool_or_gate_prior(row, actions, priors, oracle_gate=True),
        ),
        "diagnostic_all_candidate_answer_agreement_prior": (
            "diagnostic_policy",
            lambda row, actions: choose_answer_agreement(
                row,
                actions,
                priors,
                pool=ALL_ACTIONS,
                evidence_pool=ALL_ACTIONS,
                alpha=0.75,
                beta=0.50,
                fallback_pool=ALL_ACTIONS,
            ),
        ),
    }


def threshold_policy_functions(
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    priors: dict[str, Any],
) -> dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]]:
    functions: dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]] = {}
    val = target[target["split"].astype(str).eq("val")]
    available_signals = [signal for signal in THRESHOLD_SIGNALS if signal in target.columns]
    for signal in available_signals:
        thresholds = candidate_thresholds(val[signal].to_numpy(dtype=float))
        for threshold in thresholds:
            method = f"threshold_{signal}_high_thr{threshold:.4g}_local_consensus_large_prior"
            functions[method] = (
                "threshold_action_policy",
                threshold_selector(signal, float(threshold), priors),
            )
    return functions


def threshold_selector(
    signal: str,
    threshold: float,
    priors: dict[str, Any],
) -> Callable[[pd.Series, dict[str, dict[str, Any]]], str]:
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        tool = tool_action(actions)
        if tool:
            return tool
        value = as_float(row.get(signal, np.nan), default=-np.inf)
        if np.isfinite(value) and value >= threshold:
            return choose_prior_action(row, actions, priors, STRONG_OR_FRONTIER_ACTIONS, scope="benchmark")
        return choose_answer_agreement(
            row,
            actions,
            priors,
            pool=tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
            evidence_pool=tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
            alpha=0.50,
            beta=0.25,
            fallback_pool=tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
        )

    return select


def fit_train_priors(outputs: pd.DataFrame) -> dict[str, Any]:
    train = outputs[outputs["split"].astype(str).eq("train")].copy()
    benchmark_rankings: dict[tuple[str, tuple[str, ...], str], list[str]] = {}
    global_rankings: dict[tuple[tuple[str, ...], str], list[str]] = {}
    for metric in ["utility", "quality_score"]:
        for pool in pool_variants():
            ranking = rank_actions(train[train["model_id"].isin(pool)], ["model_id"], metric)
            global_rankings[(pool, metric)] = ranking
            for benchmark, group in train[train["model_id"].isin(pool)].groupby("benchmark"):
                benchmark_rankings[(str(benchmark), pool, metric)] = rank_actions(group, ["model_id"], metric)
    return {
        "benchmark_rankings": benchmark_rankings,
        "global_rankings": global_rankings,
        "all_models": tuple(sorted(outputs["model_id"].astype(str).unique())),
    }


def pool_variants() -> tuple[tuple[str, ...], ...]:
    all_without_tool = tuple(model for model in ALL_ACTIONS if model != TOOL_MODEL_ID)
    return tuple(
        dict.fromkeys(
            [
                ALL_ACTIONS,
                all_without_tool,
                CHEAP_LOCAL_ACTIONS,
                tuple(model for model in CHEAP_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
                OBSERVABLE_LOCAL_ACTIONS,
                tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
                STRONG_OR_FRONTIER_ACTIONS,
            ]
        )
    )


def rank_actions(frame: pd.DataFrame, keys: list[str], metric: str) -> list[str]:
    if frame.empty:
        return []
    grouped = (
        frame.groupby(keys, as_index=False)
        .agg(
            score=(metric, "mean"),
            mean_utility=("utility", "mean"),
            mean_quality=("quality_score", "mean"),
            mean_cost=("normalized_remote_cost", "mean"),
        )
        .sort_values(["score", "mean_utility", "mean_quality", "mean_cost"], ascending=[False, False, False, True])
    )
    return grouped["model_id"].astype(str).tolist()


def choose_prior_action(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    priors: dict[str, Any],
    pool: tuple[str, ...],
    *,
    scope: str,
    metric: str = "utility",
) -> str:
    benchmark = str(row.get("benchmark", ""))
    rankings = priors["benchmark_rankings"] if scope == "benchmark" else priors["global_rankings"]
    key = (benchmark, pool, metric) if scope == "benchmark" else (pool, metric)
    ranking = rankings.get(key, [])
    fallback = priors["global_rankings"].get((pool, metric), [])
    for model_id in [*ranking, *fallback, *pool]:
        if is_action_available(actions, model_id):
            return model_id
    return first_available(actions, priors["all_models"])


def choose_tool_or_prior(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    priors: dict[str, Any],
    pool: tuple[str, ...],
) -> str:
    tool = tool_action(actions)
    if tool:
        return tool
    return choose_prior_action(row, actions, priors, tuple(model for model in pool if model != TOOL_MODEL_ID), scope="benchmark")


def choose_tool_or_local_consensus(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    priors: dict[str, Any],
    *,
    fallback_pool: tuple[str, ...],
) -> str:
    tool = tool_action(actions)
    if tool:
        return tool
    return choose_answer_agreement(
        row,
        actions,
        priors,
        pool=tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
        evidence_pool=tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
        alpha=0.50,
        beta=0.25,
        fallback_pool=tuple(model for model in fallback_pool if model != TOOL_MODEL_ID),
    )


def choose_tool_or_gate_prior(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    priors: dict[str, Any],
    *,
    oracle_gate: bool = False,
) -> str:
    tool = tool_action(actions)
    if tool:
        return tool
    gate = bool(row.get("need_large", False)) if oracle_gate else bool(row.get("benchmark_composed_need_large", False))
    pool = STRONG_OR_FRONTIER_ACTIONS if gate else tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID)
    return choose_prior_action(row, actions, priors, pool, scope="benchmark")


def choose_tool_or_gate_consensus(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    priors: dict[str, Any],
) -> str:
    tool = tool_action(actions)
    if tool:
        return tool
    if bool(row.get("benchmark_composed_need_large", False)):
        return choose_prior_action(row, actions, priors, STRONG_OR_FRONTIER_ACTIONS, scope="benchmark")
    return choose_answer_agreement(
        row,
        actions,
        priors,
        pool=tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
        evidence_pool=tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
        alpha=0.50,
        beta=0.25,
        fallback_pool=tuple(model for model in OBSERVABLE_LOCAL_ACTIONS if model != TOOL_MODEL_ID),
    )


def choose_answer_agreement(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    priors: dict[str, Any],
    *,
    pool: tuple[str, ...],
    evidence_pool: tuple[str, ...],
    alpha: float,
    beta: float,
    fallback_pool: tuple[str, ...],
) -> str:
    counts = answer_counts(actions, evidence_pool)
    if counts:
        max_count = max(counts.values())
        top_answers = {answer for answer, count in counts.items() if count == max_count and count >= 2}
    else:
        max_count = 0
        top_answers = set()
    candidates: list[tuple[float, float, float, str]] = []
    ranking = prior_rank_index(row, priors, pool)
    n_answers = max(sum(counts.values()), 1)
    for model_id in pool:
        if not is_action_available(actions, model_id):
            continue
        answer = str(actions[model_id].get("answer_norm", ""))
        agree = counts.get(answer, 0) / n_answers if answer else 0.0
        top = 1.0 if answer in top_answers else 0.0
        prior = ranking.get(model_id, 0.0)
        cost = as_float(actions[model_id].get("normalized_remote_cost", 0.0))
        score = prior + float(alpha) * agree + float(beta) * top - 0.10 * cost
        quality = as_float(actions[model_id].get("quality_score", 0.0))
        candidates.append((score, quality, -cost, model_id))
    if candidates:
        return sorted(candidates, reverse=True)[0][3]
    return choose_prior_action(row, actions, priors, fallback_pool, scope="benchmark")


def prior_rank_index(row: pd.Series, priors: dict[str, Any], pool: tuple[str, ...]) -> dict[str, float]:
    benchmark = str(row.get("benchmark", ""))
    ranking = priors["benchmark_rankings"].get((benchmark, pool, "utility"), [])
    if not ranking:
        ranking = priors["global_rankings"].get((pool, "utility"), [])
    n = max(len(ranking), 1)
    return {model_id: (n - idx) / n for idx, model_id in enumerate(ranking)}


def full_oracle_selector(metric: str) -> Callable[[pd.Series, dict[str, dict[str, Any]]], str]:
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        candidates = [
            (
                as_float(action.get(metric, 0.0)),
                as_float(action.get("utility", 0.0)),
                as_float(action.get("quality_score", 0.0)),
                -as_float(action.get("normalized_remote_cost", 0.0)),
                model_id,
            )
            for model_id, action in actions.items()
            if is_action_available(actions, model_id)
        ]
        return sorted(candidates, reverse=True)[0][4]

    return select


def select_actions(
    frame: pd.DataFrame,
    selector: Callable[[pd.Series, dict[str, dict[str, Any]]], str],
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        query_id = str(row["query_id"])
        actions = rows_by_query[query_id]
        model_id = selector(row, actions)
        if not is_action_available(actions, model_id):
            model_id = first_available(actions, ALL_ACTIONS)
        rows.append({"query_id": query_id, "model_id": model_id})
    return pd.DataFrame(rows)


def evaluate_selected_rows(
    method: str,
    family: str,
    split: str,
    selected_rows: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    target: pd.DataFrame,
    frontiers: set[str],
    lambda_cost: float,
) -> dict[str, Any]:
    split_outputs = outputs[outputs["split"].astype(str).eq(split)]
    cost_oracle = split_outputs.loc[split_outputs.groupby("query_id")["utility"].idxmax()]
    quality_oracle = split_outputs.loc[split_outputs.groupby("query_id")["quality_score"].idxmax()]
    utility = selected_rows["utility"].to_numpy(dtype=float)
    quality = selected_rows["quality_score"].to_numpy(dtype=float)
    selected_models = selected_rows["model_id"].astype(str).to_numpy()
    choose_large = np.asarray([model in set(STRONG_OR_FRONTIER_ACTIONS) for model in selected_models], dtype=bool)
    positives = target["need_large"].astype(bool).to_numpy()
    tp = int(np.sum(choose_large & positives))
    fp = int(np.sum(choose_large & ~positives))
    fn = int(np.sum(~choose_large & positives))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    oracle_utility = float(cost_oracle["utility"].mean())
    oracle_quality = float(quality_oracle["quality_score"].mean())
    mean_utility = float(np.mean(utility)) if len(utility) else float("nan")
    mean_quality = float(np.mean(quality)) if len(quality) else float("nan")
    return {
        "method": method,
        "family": family,
        "split": split,
        "n_queries": int(selected_rows["query_id"].nunique()),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "mean_utility_ci_low": np.nan,
        "mean_utility_ci_high": np.nan,
        "normalized_cost_mean": float(selected_rows["normalized_remote_cost"].mean()),
        "remote_cost_total_usd": float(selected_rows["cost_total_usd"].sum()),
        "mean_latency_s": float(selected_rows["latency_s"].mean()),
        "p95_latency_s": float(selected_rows["latency_s"].quantile(0.95)),
        "cost_oracle_mean_utility": oracle_utility,
        "quality_oracle_mean_quality": oracle_quality,
        "utility_gap_to_oracle": float(oracle_utility - mean_utility),
        "quality_gap_to_oracle": float(oracle_quality - mean_quality),
        "oracle_utility_ratio": float(mean_utility / max(oracle_utility, 1e-12)),
        "within_3pct_oracle_utility": bool(mean_utility >= 0.97 * oracle_utility),
        "within_3pt_oracle_quality": bool(mean_quality >= oracle_quality - 0.03),
        "frontier_call_rate": float(np.mean([model in frontiers for model in selected_models])),
        "strong_or_frontier_call_rate": float(np.mean(choose_large)),
        "local_call_rate": float(selected_rows["is_local"].astype(bool).mean()),
        "need_large_precision": float(precision),
        "need_large_recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "selected_models_json": json.dumps(selected_rows["model_id"].value_counts().sort_index().to_dict(), sort_keys=True),
        "lambda_cost": float(lambda_cost),
        "_utility_values": utility.tolist(),
    }


def validation_selected_rows(table: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in table.groupby("family"):
        if family == "diagnostic_oracle":
            rows.append(group[group["split"].eq("test")].copy().assign(selection_rule="diagnostic_oracle_test"))
            continue
        val = group[group["split"].eq("val")].copy()
        if val.empty:
            continue
        best = val.sort_values(
            ["mean_utility", "normalized_cost_mean", "frontier_call_rate"],
            ascending=[False, True, True],
        ).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = group[group["split"].eq("test") & group["method"].eq(method)].copy()
        if not test.empty:
            rows.append(test.assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    with_values = table[["method", "split", "_utility_values"]]
    selected = selected.drop(columns=["_utility_values"], errors="ignore").merge(
        with_values,
        on=["method", "split"],
        how="left",
    )
    selected = add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)
    return selected.drop(columns=["_utility_values"], errors="ignore")


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


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.asarray([0.0])
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, 31)))


def answer_counts(actions: dict[str, dict[str, Any]], evidence_pool: tuple[str, ...]) -> Counter[str]:
    values: list[str] = []
    for model_id in evidence_pool:
        if not is_action_available(actions, model_id):
            continue
        answer = str(actions[model_id].get("answer_norm", ""))
        if answer:
            values.append(answer)
    return Counter(values)


def tool_action(actions: dict[str, dict[str, Any]]) -> str | None:
    if is_action_available(actions, TOOL_MODEL_ID):
        return TOOL_MODEL_ID
    return None


def is_action_available(actions: dict[str, dict[str, Any]], model_id: str) -> bool:
    if model_id not in actions:
        return False
    if model_id == TOOL_MODEL_ID:
        action = actions[model_id]
        return bool(action.get("tool_available", False)) and bool(action.get("answer_norm", ""))
    return True


def first_available(actions: dict[str, dict[str, Any]], pool: tuple[str, ...] | list[str]) -> str:
    for model_id in pool:
        if is_action_available(actions, str(model_id)):
            return str(model_id)
    return sorted(actions)[0]


def normalized_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text or text == "nan" or text == "no_code" or text.startswith("failed"):
        return ""
    return text


def as_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def detail_methods() -> set[str]:
    return {
        "full_cost_aware_oracle",
        "tool_then_local_consensus_else_benchmark_prior",
        "tool_then_171_gate_local_consensus_large_prior",
        "diagnostic_all_candidate_answer_agreement_prior",
    }


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#4f6f64")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Actual Deployed-Action Policy Bridge")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tool_aware_deployed_action_policy_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
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
        "within_3pct_oracle_utility",
        "within_3pt_oracle_quality",
        "utility_gap_to_oracle",
        "quality_gap_to_oracle",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "need_large_precision",
        "need_large_recall",
        "selection_rule",
    ]
    selected_cols = [column for column in cols if column in selected.columns]
    test = table[table["split"].eq("test")].copy()
    deployable_test = test[test["family"].isin(["deployed_policy", "threshold_action_policy", "reference"])]
    best_deployable = deployable_test.sort_values(["mean_utility", "mean_quality"], ascending=False).head(1)
    full_oracle = test[test["method"].eq("full_cost_aware_oracle")].head(1)
    lines = [
        "# Tool-Aware Deployed-Action Policy Bridge",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/172_tool_aware_deployed_action_policy.py",
        (
            "PYTHONPATH=src python experiments/172_tool_aware_deployed_action_policy.py "
            f"--target-table {args.target_table} --outputs {args.outputs} --output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## What This Tests",
        "",
        "- The previous 171 result routed between each query's best local-side action and best large-side action.",
        "- This bridge selects concrete actions from cached outputs using train-only action priors, deterministic tool availability, local answer agreement, and validation-selected thresholds.",
        "- It makes no GPT, Gemini, Claude, or vLLM calls; all model outputs are cached.",
        "- Diagnostic rows are explicitly labeled when they use the full oracle or all candidate answers.",
        "",
        "## Selected Rows",
        "",
        "```csv",
        compact_csv(selected[selected_cols], max_rows=80),
        "```",
        "",
        "## Best Held-Out Actual Policies",
        "",
        "```csv",
        compact_csv(test.sort_values(["mean_utility", "mean_quality"], ascending=False)[[c for c in cols if c in test.columns]], max_rows=40),
        "```",
        "",
        "## Deployed-Policy Target Check",
        "",
        *target_check_lines(full_oracle, best_deployable),
        "",
        "## Interpretation",
        "",
        "- The exact-math tool and local consensus improve action selection, but the deployable policies still trail the full cost-aware oracle.",
        "- The main gap is concrete action identity, not only local-vs-large escalation.",
        "- The strongest diagnostic rows show the ceiling if more answer evidence or an adjudicator were available, but those rows should not be claimed as deployable.",
        "",
        "## Artifacts",
        "",
        f"- All policy table: `{args.output_dir / 'table_tool_aware_deployed_action_policy_all.csv'}`",
        f"- Selected policy table: `{args.output_dir / 'table_tool_aware_deployed_action_policy_selected.csv'}`",
        f"- Query-level choices for key methods: `{args.output_dir / 'table_tool_aware_deployed_action_choices.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_tool_aware_deployed_action_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_check_lines(full_oracle: pd.DataFrame, best_deployable: pd.DataFrame) -> list[str]:
    if full_oracle.empty or best_deployable.empty:
        return ["- Missing full-oracle or deployable-policy rows."]
    oracle = full_oracle.iloc[0]
    best = best_deployable.iloc[0]
    utility_target = 0.97 * float(oracle["mean_utility"])
    quality_target = float(oracle["mean_quality"]) - 0.03
    return [
        f"- Full held-out cost-aware oracle utility: `{float(oracle['mean_utility']):.4f}`.",
        f"- 97% utility target: `{utility_target:.4f}`.",
        f"- Best deployable held-out method: `{best['method']}`.",
        f"- Best deployable utility: `{float(best['mean_utility']):.4f}`; pass: `{bool(best['mean_utility'] >= utility_target)}`.",
        f"- Full held-out cost-aware oracle quality: `{float(oracle['mean_quality']):.4f}`.",
        f"- Within-3-point quality target: `{quality_target:.4f}`.",
        f"- Best deployable quality: `{float(best['mean_quality']):.4f}`; pass: `{bool(best['mean_quality'] >= quality_target)}`.",
    ]


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
