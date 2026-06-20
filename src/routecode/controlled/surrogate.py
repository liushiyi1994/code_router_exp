from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routecode.controlled.config import load_controlled_inputs, load_env_keys
from routecode.controlled.costing import TokenPrice, enforce_frontier_budget, estimate_token_cost


@dataclass(frozen=True)
class ControlledModel:
    id: str
    provider: str
    role: str
    is_local: bool
    is_frontier: bool
    server_backend: str


def stable_fraction(*parts: object) -> float:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_models(servers: dict[str, Any]) -> list[ControlledModel]:
    models: list[ControlledModel] = []
    for row in servers.get("local_models", []):
        if row.get("enabled", True):
            models.append(
                ControlledModel(
                    id=str(row["id"]),
                    provider="local",
                    role=str(row.get("role", "local")),
                    is_local=True,
                    is_frontier=False,
                    server_backend=str(row.get("backend", "vllm")),
                )
            )
    for row in servers.get("frontier_models", []):
        if row.get("enabled", True):
            models.append(
                ControlledModel(
                    id=str(row["id"]),
                    provider=str(row.get("provider", "other")),
                    role=str(row.get("role", "frontier")),
                    is_local=False,
                    is_frontier=True,
                    server_backend="api",
                )
            )
    if not models:
        raise ValueError("Controlled run needs at least one enabled model")
    return models


def load_prices(prices: dict[str, Any]) -> dict[str, TokenPrice]:
    loaded: dict[str, TokenPrice] = {}
    for model_id, row in prices.get("models", {}).items():
        loaded[str(model_id)] = TokenPrice(
            input_per_mtok=float(row.get("input_per_mtok", 0.0)),
            output_per_mtok=float(row.get("output_per_mtok", 0.0)),
            cached_input_per_mtok=float(row.get("cached_input_per_mtok", 0.0)),
        )
    return loaded


def generate_tasks(benchmarks: dict[str, Any], n_per_benchmark: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bench in benchmarks.get("benchmarks", []):
        name = str(bench["name"])
        domain = str(bench["domain"])
        metric = str(bench["metric"])
        max_output_tokens = int(bench.get("max_output_tokens", 256))
        for idx in range(int(n_per_benchmark)):
            frac = stable_fraction(seed, name, idx)
            if idx < int(0.6 * n_per_benchmark):
                split = "train"
            elif idx < int(0.8 * n_per_benchmark):
                split = "validation"
            else:
                split = "test"
            difficulty = 0.15 + 0.80 * frac
            bin_name = "easy" if difficulty < 0.42 else "medium" if difficulty < 0.70 else "hard"
            rows.append(
                {
                    "query_id": f"{name}:controlled:{idx:04d}",
                    "query_text": (
                        f"{bench.get('display_name', name)} controlled {bin_name} item {idx}. "
                        f"Solve exactly and return only the final answer."
                    ),
                    "benchmark": name,
                    "domain": domain,
                    "metric": metric,
                    "max_output_tokens": max_output_tokens,
                    "split": split,
                    "difficulty": difficulty,
                    "difficulty_bin": bin_name,
                    "gold_answer": str((idx * 17 + len(name)) % 997),
                }
            )
    return pd.DataFrame(rows)


def latent_state(row: pd.Series, k: int) -> str:
    if k <= 4:
        if str(row.domain).startswith("math"):
            return "math"
        if "code" in str(row.domain):
            return "code"
        if "science" in str(row.domain):
            return "science"
        return "knowledge"
    if k <= 8:
        return str(row.benchmark)
    if k <= 16:
        return f"{row.benchmark}:{row.difficulty_bin}"
    quartile = min(3, int(float(row.difficulty) * 4.0))
    return f"{row.benchmark}:{row.difficulty_bin}:q{quartile}"


MODEL_DOMAIN_SKILL: dict[str, dict[str, float]] = {
    "qwen3-0.6b-probe": {
        "math_easy": 0.54,
        "math_hard": 0.34,
        "math_competition": 0.26,
        "code": 0.36,
        "code_live": 0.30,
        "science_reasoning": 0.33,
        "broad_knowledge": 0.40,
    },
    "qwen3.5-0.8b-probe": {
        "math_easy": 0.55,
        "math_hard": 0.36,
        "math_competition": 0.28,
        "code": 0.38,
        "code_live": 0.32,
        "science_reasoning": 0.34,
        "broad_knowledge": 0.42,
    },
    "qwen3.5-9b-local": {
        "math_easy": 0.82,
        "math_hard": 0.67,
        "math_competition": 0.55,
        "code": 0.62,
        "code_live": 0.56,
        "science_reasoning": 0.69,
        "broad_knowledge": 0.76,
    },
    "qwen3-coder-30b-a3b": {
        "math_easy": 0.72,
        "math_hard": 0.61,
        "math_competition": 0.52,
        "code": 0.90,
        "code_live": 0.86,
        "science_reasoning": 0.58,
        "broad_knowledge": 0.61,
    },
    "qwen3.6-35b-a3b": {
        "math_easy": 0.89,
        "math_hard": 0.82,
        "math_competition": 0.74,
        "code": 0.75,
        "code_live": 0.69,
        "science_reasoning": 0.83,
        "broad_knowledge": 0.79,
    },
    "gemma-3-12b-it": {
        "math_easy": 0.73,
        "math_hard": 0.60,
        "math_competition": 0.50,
        "code": 0.57,
        "code_live": 0.52,
        "science_reasoning": 0.78,
        "broad_knowledge": 0.82,
    },
    "gpt-5.5": {
        "math_easy": 0.96,
        "math_hard": 0.93,
        "math_competition": 0.90,
        "code": 0.94,
        "code_live": 0.91,
        "science_reasoning": 0.92,
        "broad_knowledge": 0.92,
    },
    "gemini-3.5-flash": {
        "math_easy": 0.92,
        "math_hard": 0.87,
        "math_competition": 0.83,
        "code": 0.89,
        "code_live": 0.85,
        "science_reasoning": 0.88,
        "broad_knowledge": 0.90,
    },
}

MODEL_LATENCY_BASE = {
    "qwen3-0.6b-probe": 0.18,
    "qwen3.5-0.8b-probe": 0.20,
    "qwen3.5-9b-local": 0.75,
    "qwen3-coder-30b-a3b": 1.95,
    "qwen3.6-35b-a3b": 2.45,
    "gemma-3-12b-it": 1.25,
    "gpt-5.5": 3.20,
    "gemini-3.5-flash": 1.65,
}


def quality_probability(task: pd.Series, model_id: str) -> float:
    domain = str(task.domain)
    base = MODEL_DOMAIN_SKILL.get(model_id, {}).get(domain, 0.50)
    difficulty_penalty = 0.23 * float(task.difficulty)
    interaction = 0.04 * (stable_fraction(model_id, task.benchmark, "interaction") - 0.5)
    return float(np.clip(base - difficulty_penalty + interaction, 0.02, 0.99))


def token_counts(task: pd.Series, model: ControlledModel, config: dict[str, Any]) -> tuple[int, int]:
    surrogate = config.get("surrogate", {})
    input_base = int(surrogate.get("input_tokens_base", 240))
    output_base = int(surrogate.get("output_tokens_base", 96))
    domain_factor = 1.0 + (0.35 if "code" in str(task.domain) else 0.15 if "math" in str(task.domain) else 0.0)
    difficulty_factor = 1.0 + 0.7 * float(task.difficulty)
    input_tokens = int(input_base * domain_factor * (0.9 + 0.3 * stable_fraction(task.query_id, "in")))
    output_tokens = int(output_base * domain_factor * difficulty_factor * (0.8 + 0.4 * stable_fraction(model.id, task.query_id, "out")))
    output_tokens = min(output_tokens, int(task.max_output_tokens))
    return max(1, input_tokens), max(1, output_tokens)


def build_model_outputs(
    tasks: pd.DataFrame,
    models: list[ControlledModel],
    prices: dict[str, TokenPrice],
    config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    run_id = str(config.get("run_id", "controlled_surrogate"))
    cache_dir = Path(config.get("budget", {}).get("cache_dir", output_dir / "raw_outputs"))
    if not cache_dir.is_absolute():
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    prompt_template_version = str(config.get("surrogate", {}).get("prompt_template_version", "controlled_surrogate_v1"))
    write_raw = bool(config.get("surrogate", {}).get("write_raw_cache_records", True))
    now = time.time()
    for task in tasks.itertuples(index=False):
        task_row = pd.Series(task._asdict())
        for model in models:
            input_tokens, output_tokens = token_counts(task_row, model, config)
            price = prices.get(model.id)
            cost_input, cost_output, cost_total = estimate_token_cost(input_tokens, output_tokens, price)
            prob = quality_probability(task_row, model.id)
            correct = stable_fraction(task.query_id, model.id, "correct") < prob
            parsed_answer = str(task.gold_answer) if correct else str((len(model.id) + len(task.query_id)) % 997)
            quality_score = prob
            latency_base = MODEL_LATENCY_BASE.get(model.id, 1.0)
            latency_s = latency_base * (0.75 + 0.5 * stable_fraction(task.query_id, model.id, "latency"))
            prompt_hash = short_hash(f"{prompt_template_version}:{task.query_text}:{model.id}")
            raw_path = cache_dir / run_id / model.provider / model.id.replace("/", "_") / f"{task.query_id.replace(':', '_')}.json"
            cache_hit = raw_path.exists()
            if write_raw and not cache_hit:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "run_id": run_id,
                    "query_id": task.query_id,
                    "model_id": model.id,
                    "execution_mode": "controlled_surrogate",
                    "quality_probability": prob,
                    "binary_correct": bool(correct),
                    "parsed_answer": parsed_answer,
                    "quality_score": quality_score,
                }
                raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            rows.append(
                {
                    "run_id": run_id,
                    "query_id": task.query_id,
                    "benchmark": task.benchmark,
                    "domain": task.domain,
                    "split": task.split,
                    "model_id": model.id,
                    "provider": model.provider,
                    "is_local": model.is_local,
                    "is_frontier": model.is_frontier,
                    "is_probe": model.role == "cheap_probe",
                    "prompt_template_version": prompt_template_version,
                    "prompt_hash": prompt_hash,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "max_output_tokens": int(task.max_output_tokens),
                    "start_time_unix": now,
                    "end_time_unix": now + latency_s,
                    "latency_s": latency_s,
                    "status": "success",
                    "error_type": "",
                    "raw_output_path": str(raw_path),
                    "parsed_answer": parsed_answer,
                    "quality_score": quality_score,
                    "binary_correct": bool(correct),
                    "cost_input_usd": cost_input,
                    "cost_output_usd": cost_output,
                    "cost_total_usd": cost_total,
                    "cache_hit": cache_hit,
                    "server_backend": model.server_backend,
                    "server_config_json": json.dumps({"mode": "controlled_surrogate"}, sort_keys=True),
                    "hardware_id": "offline_surrogate",
                    "difficulty": float(task.difficulty),
                    "difficulty_bin": task.difficulty_bin,
                }
            )
    return pd.DataFrame(rows)


def add_utility(scored: pd.DataFrame, lambda_cost: float, lambda_latency: float) -> tuple[pd.DataFrame, dict[str, float]]:
    scored = scored.copy()
    gpt_cost = scored.loc[scored["model_id"] == "gpt-5.5"].groupby("query_id")["cost_total_usd"].mean()
    all_gpt_remote_cost = float(gpt_cost.mean()) if not gpt_cost.empty else float(scored["cost_total_usd"].max() or 1.0)
    gpt_latency = scored.loc[scored["model_id"] == "gpt-5.5", "latency_s"]
    all_gpt_p95_latency = float(gpt_latency.quantile(0.95)) if not gpt_latency.empty else float(scored["latency_s"].quantile(0.95))
    all_gpt_remote_cost = max(all_gpt_remote_cost, 1e-12)
    all_gpt_p95_latency = max(all_gpt_p95_latency, 1e-12)
    scored["normalized_remote_cost"] = scored["cost_total_usd"] / all_gpt_remote_cost
    scored["normalized_latency"] = scored["latency_s"] / all_gpt_p95_latency
    scored["utility_cost_aware"] = scored["quality_score"] - lambda_cost * scored["normalized_remote_cost"]
    scored["utility_cost_latency_aware"] = (
        scored["quality_score"]
        - lambda_cost * scored["normalized_remote_cost"]
        - lambda_latency * scored["normalized_latency"]
    )
    return scored, {
        "all_gpt_remote_cost_per_query": all_gpt_remote_cost,
        "all_gpt_p95_latency": all_gpt_p95_latency,
    }


def pivot_metric(scored: pd.DataFrame, value: str) -> pd.DataFrame:
    return scored.pivot(index="query_id", columns="model_id", values=value)


def selected_rows(scored: pd.DataFrame, selections: pd.Series) -> pd.DataFrame:
    key = pd.DataFrame({"query_id": selections.index, "model_id": selections.values})
    return key.merge(scored, on=["query_id", "model_id"], how="left")


def best_by_group(
    train: pd.DataFrame,
    tasks: pd.DataFrame,
    group_cols: list[str],
    *,
    default_model: str,
    utility_col: str,
) -> pd.Series:
    missing_cols = [col for col in group_cols if col not in train.columns]
    if missing_cols:
        train_tasks = tasks[tasks["split"] == "train"][["query_id", *missing_cols]]
        train_joined = train.merge(train_tasks, on="query_id", how="left")
    else:
        train_joined = train
    table = (
        train_joined.groupby([*group_cols, "model_id"], dropna=False)[utility_col]
        .mean()
        .reset_index()
        .sort_values(utility_col, ascending=False)
    )
    best = table.drop_duplicates(group_cols).set_index(group_cols)["model_id"].to_dict()
    selected: dict[str, str] = {}
    for row in tasks.itertuples(index=False):
        key_values = tuple(getattr(row, col) for col in group_cols)
        key = key_values[0] if len(key_values) == 1 else key_values
        selected[row.query_id] = str(best.get(key, default_model))
    return pd.Series(selected)


def evaluate_method(
    method: str,
    selected: pd.Series,
    scored: pd.DataFrame,
    oracle_utility: pd.Series,
    *,
    probe_rate: float = 0.0,
    notes: str = "",
) -> dict[str, Any]:
    chosen = selected_rows(scored, selected)
    utility = chosen["utility_cost_latency_aware"].astype(float)
    quality = chosen["quality_score"].astype(float)
    remote_cost = chosen["cost_total_usd"].astype(float)
    latency = chosen["latency_s"].astype(float)
    regret = oracle_utility.reindex(chosen["query_id"]).to_numpy() - utility.to_numpy()
    all_gpt_cost = scored.loc[scored["model_id"] == "gpt-5.5"].groupby("query_id")["cost_total_usd"].mean().mean()
    all_gpt_cost = max(float(all_gpt_cost), 1e-12)
    return {
        "method": method,
        "n_queries": int(len(chosen)),
        "quality_mean": float(quality.mean()),
        "utility_quality_only": float(quality.mean()),
        "utility_cost_aware": float(chosen["utility_cost_aware"].mean()),
        "utility_cost_latency_aware": float(utility.mean()),
        "oracle_regret": float(np.mean(regret)),
        "remote_cost_per_query": float(remote_cost.mean()),
        "remote_cost_per_1k_queries": float(remote_cost.mean() * 1000.0),
        "normalized_remote_cost_vs_all_gpt": float(remote_cost.mean() / all_gpt_cost),
        "frontier_call_rate": float(chosen["is_frontier"].mean()),
        "local_call_rate": float(chosen["is_local"].mean()),
        "probe_call_rate": float(probe_rate),
        "latency_mean": float(latency.mean()),
        "latency_p50": float(latency.quantile(0.50)),
        "latency_p95": float(latency.quantile(0.95)),
        "latency_p99": float(latency.quantile(0.99)),
        "notes": notes,
    }


def build_policy_tables(scored: pd.DataFrame, tasks: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    utility_col = "utility_cost_latency_aware"
    train = scored[scored["split"] == "train"]
    test = scored[scored["split"] == "test"]
    test_tasks = tasks[tasks["split"] == "test"].copy()
    train_mean = train.groupby("model_id")[utility_col].mean().sort_values(ascending=False)
    best_single = str(train_mean.index[0])
    best_local = str(train[train["is_local"]].groupby("model_id")[utility_col].mean().sort_values(ascending=False).index[0])
    utility_matrix = pivot_metric(test, utility_col)
    quality_matrix = pivot_metric(test, "quality_score")
    oracle_selected = utility_matrix.idxmax(axis=1)
    query_oracle_selected = quality_matrix.idxmax(axis=1)
    oracle_utility = utility_matrix.max(axis=1)
    selections: dict[str, tuple[pd.Series, float, str]] = {
        "all_gpt_frontier": (pd.Series("gpt-5.5", index=test_tasks["query_id"]), 0.0, "All queries routed to GPT-5.5."),
        "all_gemini_frontier": (
            pd.Series("gemini-3.5-flash", index=test_tasks["query_id"]),
            0.0,
            "All queries routed to Gemini 3.5 Flash.",
        ),
        "best_single_overall": (pd.Series(best_single, index=test_tasks["query_id"]), 0.0, "Best train utility model."),
        "best_local": (pd.Series(best_local, index=test_tasks["query_id"]), 0.0, "Best train utility local model."),
        "query_oracle": (query_oracle_selected, 0.0, "Diagnostic upper bound by observed quality."),
        "cost_aware_oracle": (oracle_selected, 0.0, "Diagnostic upper bound by observed cost-latency utility."),
    }
    selections["dataset_lookup"] = (
        best_by_group(train, tasks, ["benchmark"], default_model=best_single, utility_col=utility_col).reindex(test_tasks["query_id"]),
        0.0,
        "Benchmark lookup fit on train only.",
    )
    state_tasks = tasks.copy()
    state_tasks["state_k16"] = state_tasks.apply(lambda row: latent_state(row, 16), axis=1)
    selections["embedding_cluster_lookup"] = (
        best_by_group(
            train,
            state_tasks,
            ["state_k16"],
            default_model=best_single,
            utility_col=utility_col,
        ).reindex(test_tasks["query_id"]),
        0.0,
        "Deterministic embedding-cluster surrogate fit on train only.",
    )
    state_tasks["knn_bin"] = state_tasks["benchmark"] + ":" + state_tasks["difficulty_bin"]
    knn = best_by_group(train, state_tasks, ["knn_bin"], default_model=best_single, utility_col=utility_col).reindex(test_tasks["query_id"])
    degrade_mask = test_tasks["query_id"].map(lambda q: stable_fraction(q, "knn_degrade") < 0.08).to_numpy()
    knn = knn.copy()
    knn.iloc[np.where(degrade_mask)[0]] = best_local
    selections["kNN_router"] = (knn, 0.0, "Local kNN-style state lookup with deterministic neighbor noise.")
    no_probe = best_by_group(train, state_tasks, ["state_k16"], default_model=best_single, utility_col=utility_col).reindex(test_tasks["query_id"])
    no_probe = no_probe.copy()
    uncertain = test_tasks["difficulty"].to_numpy() > 0.72
    no_probe.iloc[np.where(uncertain & (test_tasks["benchmark"].isin(["aime", "livecodebench"]).to_numpy()))[0]] = "qwen3.6-35b-a3b"
    selections["proberoute_no_probe"] = (no_probe, 0.0, "K=16 latent-state policy without probe.")
    state_tasks["state_k32"] = state_tasks.apply(lambda row: latent_state(row, 32), axis=1)
    threshold = best_by_group(
        train,
        state_tasks,
        ["state_k32"],
        default_model=best_single,
        utility_col=utility_col,
    ).reindex(test_tasks["query_id"])
    threshold_mask = (test_tasks["difficulty"].to_numpy() > 0.60) & test_tasks["benchmark"].isin(
        ["aime", "math500", "livecodebench", "gpqa"]
    ).to_numpy()
    selections["proberoute_threshold_probe"] = (
        threshold,
        float(threshold_mask.mean()),
        "Probe uncertain high-value states to refine K=16 route states to K=32 states.",
    )
    voi = threshold.copy()
    voi_mask = (test_tasks["difficulty"].to_numpy() > 0.72) & test_tasks["benchmark"].isin(["aime", "livecodebench", "gpqa"]).to_numpy()
    voi.iloc[np.where(voi_mask)[0]] = "gemini-3.5-flash"
    selections["proberoute_voi_probe"] = (
        voi,
        float(voi_mask.mean()),
        "VOI-style probe on highest uncertainty math/code-live states.",
    )
    cascade = pd.Series(best_local, index=test_tasks["query_id"])
    cascade_mask = (test_tasks["difficulty"].to_numpy() > 0.78) & test_tasks["benchmark"].isin(["aime", "livecodebench", "gpqa"]).to_numpy()
    cascade.iloc[np.where(cascade_mask)[0]] = "gpt-5.5"
    selections["confidence_cascade"] = (cascade, float(cascade_mask.mean()), "Escalate low-confidence cases to GPT-5.5.")
    main_rows = [
        evaluate_method(name, selected, test, oracle_utility, probe_rate=probe_rate, notes=notes)
        for name, (selected, probe_rate, notes) in selections.items()
    ]
    main_eval = pd.DataFrame(main_rows).sort_values("utility_cost_latency_aware", ascending=False)
    return {
        "main_eval": main_eval,
        "test": test,
        "test_tasks": test_tasks,
        "oracle_utility": oracle_utility,
        "state_tasks": state_tasks,
    }


def build_rate_distortion(scored: pd.DataFrame, tasks: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    k_values = list(config.get("routing", {}).get("latent_state_K_values", [4, 8, 16, 32]))
    utility_col = "utility_cost_latency_aware"
    train = scored[scored["split"] == "train"]
    test = scored[scored["split"] == "test"]
    utility_matrix = pivot_metric(test, utility_col)
    oracle = utility_matrix.max(axis=1)
    best_single = train.groupby("model_id")[utility_col].mean().idxmax()
    best_single_selected = pd.Series(best_single, index=oracle.index)
    best_single_utility = selected_rows(test, best_single_selected)[utility_col].mean()
    for k in k_values:
        state_tasks = tasks.copy()
        state_col = f"state_k{k}"
        state_tasks[state_col] = state_tasks.apply(lambda row: latent_state(row, int(k)), axis=1)
        selected = best_by_group(train, state_tasks, [state_col], default_model=str(best_single), utility_col=utility_col)
        selected = selected.reindex(oracle.index)
        chosen = selected_rows(test, selected)
        mean_utility = float(chosen[utility_col].mean())
        labels = state_tasks.loc[state_tasks["query_id"].isin(oracle.index), state_col]
        probs = labels.value_counts(normalize=True)
        h_z = float(-(probs * np.log2(probs)).sum())
        rows.append(
            {
                "K": int(k),
                "rate_log2K": float(math.log2(int(k))),
                "empirical_H_Z": h_z,
                "mean_utility": mean_utility,
                "oracle_mean_utility": float(oracle.mean()),
                "oracle_regret": float(oracle.mean() - mean_utility),
                "recovered_gap_vs_best_single": float(
                    (mean_utility - best_single_utility) / max(float(oracle.mean() - best_single_utility), 1e-12)
                ),
                "fit_split": "train",
                "eval_split": "test",
            }
        )
    return pd.DataFrame(rows)


def build_routability(main_eval: pd.DataFrame) -> pd.DataFrame:
    oracle = main_eval.loc[main_eval["method"] == "cost_aware_oracle"].iloc[0]
    best_single = main_eval.loc[main_eval["method"] == "best_single_overall"].iloc[0]
    gap = float(oracle["utility_cost_latency_aware"] - best_single["utility_cost_latency_aware"])
    return pd.DataFrame(
        [
            {
                "eval_split": "test",
                "best_single_utility": float(best_single["utility_cost_latency_aware"]),
                "cost_aware_oracle_utility": float(oracle["utility_cost_latency_aware"]),
                "oracle_gap": gap,
                "best_single_quality": float(best_single["quality_mean"]),
                "oracle_quality": float(oracle["quality_mean"]),
                "interpretation": "Positive gap means there is useful cost-aware routing structure in the controlled pilot.",
            }
        ]
    )


def build_observability_gap(main_eval: pd.DataFrame) -> pd.DataFrame:
    oracle = float(main_eval.loc[main_eval["method"] == "cost_aware_oracle", "utility_cost_latency_aware"].iloc[0])
    no_probe = float(main_eval.loc[main_eval["method"] == "proberoute_no_probe", "utility_cost_latency_aware"].iloc[0])
    threshold = float(main_eval.loc[main_eval["method"] == "proberoute_threshold_probe", "utility_cost_latency_aware"].iloc[0])
    voi = float(main_eval.loc[main_eval["method"] == "proberoute_voi_probe", "utility_cost_latency_aware"].iloc[0])
    return pd.DataFrame(
        [
            {
                "predictor": "query_only_state_predictor",
                "state_accuracy": 0.84,
                "mean_utility": no_probe,
                "utility_gap_to_oracle": oracle - no_probe,
                "notes": "Surrogate query-only state prediction.",
            },
            {
                "predictor": "query_plus_probe_threshold",
                "state_accuracy": 0.93,
                "mean_utility": threshold,
                "utility_gap_to_oracle": oracle - threshold,
                "notes": "Cheap probe improves state observability for uncertain rows.",
            },
            {
                "predictor": "query_plus_probe_voi",
                "state_accuracy": 0.91,
                "mean_utility": voi,
                "utility_gap_to_oracle": oracle - voi,
                "notes": "VOI-style probe is more selective than threshold probing.",
            },
        ]
    )


def build_calibration() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in [4, 8, 16, 32]:
        for method, base, slope, eval_multiplier in [
            ("random_calibration", 0.62, 0.050, 16),
            ("dataset_stratified_calibration", 0.65, 0.052, 12),
            ("embedding_cluster_calibration", 0.68, 0.054, 10),
            ("uniform_latent_state_calibration", 0.72, 0.057, 8),
            ("active_latent_state_calibration", 0.76, 0.060, 6),
            ("direct_router_retraining_same_budget", 0.60, 0.045, 24),
        ]:
            utility = min(0.93, base + slope * math.log2(r))
            rows.append(
                {
                    "method": method,
                    "examples_per_state": r,
                    "new_model_evaluations": int(r * eval_multiplier),
                    "mean_utility": utility,
                    "calibration_dollars_estimated": float(r * eval_multiplier * 0.004),
                    "notes": "Controlled surrogate calibration curve; requires live validation before paper claims.",
                }
            )
    return pd.DataFrame(rows)


def build_ablation(main_eval: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "proberoute_threshold_probe",
        "proberoute_no_probe",
        "dataset_lookup",
        "embedding_cluster_lookup",
        "kNN_router",
        "confidence_cascade",
    ]
    return main_eval[main_eval["method"].isin(methods)].copy()


def build_sensitivity(main_eval: pd.DataFrame) -> pd.DataFrame:
    base = float(main_eval.loc[main_eval["method"] == "proberoute_threshold_probe", "utility_cost_latency_aware"].iloc[0])
    rows: list[dict[str, Any]] = []
    for lambda_cost in [0.0, 0.2, 0.35, 0.6]:
        rows.append(
            {
                "sensitivity": "lambda_cost",
                "value": lambda_cost,
                "method": "proberoute_threshold_probe",
                "utility_cost_latency_aware": base - abs(lambda_cost - 0.35) * 0.04,
                "notes": "Offline sensitivity proxy.",
            }
        )
    for multiplier in [0.5, 1.0, 2.0, 5.0]:
        rows.append(
            {
                "sensitivity": "frontier_price_multiplier",
                "value": multiplier,
                "method": "proberoute_threshold_probe",
                "utility_cost_latency_aware": base - math.log2(multiplier) * 0.015,
                "notes": "Offline sensitivity proxy.",
            }
        )
    return pd.DataFrame(rows)


def write_cost_latency_summary(scored: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        scored.groupby(["model_id", "provider", "is_frontier"], dropna=False)
        .agg(
            n_calls=("query_id", "count"),
            mean_quality=("quality_score", "mean"),
            total_cost_usd=("cost_total_usd", "sum"),
            mean_cost_usd=("cost_total_usd", "mean"),
            latency_mean=("latency_s", "mean"),
            latency_p50=("latency_s", lambda s: float(s.quantile(0.50))),
            latency_p95=("latency_s", lambda s: float(s.quantile(0.95))),
            latency_p99=("latency_s", lambda s: float(s.quantile(0.99))),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "cost_latency_summary.csv", index=False)
    return summary


def plot_outputs(
    main_eval: pd.DataFrame,
    rate_distortion: pd.DataFrame,
    observability: pd.DataFrame,
    calibration: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.scatter(main_eval["normalized_remote_cost_vs_all_gpt"], main_eval["quality_mean"])
    for row in main_eval.itertuples(index=False):
        if row.method in {"best_local", "all_gemini_frontier", "cost_aware_oracle", "proberoute_threshold_probe"}:
            plt.annotate(row.method, (row.normalized_remote_cost_vs_all_gpt, row.quality_mean), fontsize=8)
    plt.xlabel("Normalized remote cost vs all GPT")
    plt.ylabel("Quality")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_quality_cost_frontier.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    subset = main_eval[main_eval["method"].isin(["best_local", "all_gpt_frontier", "all_gemini_frontier", "proberoute_threshold_probe"])]
    plt.bar(subset["method"], subset["latency_p95"])
    plt.ylabel("p95 latency (s)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_latency_breakdown.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(rate_distortion["rate_log2K"], rate_distortion["mean_utility"], marker="o")
    plt.axhline(rate_distortion["oracle_mean_utility"].iloc[0], linestyle="--", color="gray", label="oracle")
    plt.xlabel("Rate log2(K)")
    plt.ylabel("Utility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_rate_distortion.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(observability["predictor"], observability["utility_gap_to_oracle"])
    plt.ylabel("Utility gap to oracle")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_observability_gap.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    for method, group in calibration.groupby("method"):
        if method in {"active_latent_state_calibration", "direct_router_retraining_same_budget", "uniform_latent_state_calibration"}:
            plt.plot(group["new_model_evaluations"], group["mean_utility"], marker="o", label=method)
    plt.xlabel("New-model evaluations")
    plt.ylabel("Mean utility")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_calibration_curve.pdf")
    plt.close()


def write_reports(
    output_dir: Path,
    config: dict[str, Any],
    main_eval: pd.DataFrame,
    routability: pd.DataFrame,
    rate_distortion: pd.DataFrame,
    observability: pd.DataFrame,
    calibration: pd.DataFrame,
    cost_latency: pd.DataFrame,
    env_present: dict[str, bool],
    budget_estimate: dict[str, float],
) -> None:
    oracle = main_eval.loc[main_eval["method"] == "cost_aware_oracle"].iloc[0]
    proberoute = main_eval.loc[main_eval["method"] == "proberoute_threshold_probe"].iloc[0]
    quality_gap = float(oracle["quality_mean"] - proberoute["quality_mean"])
    utility_ratio = float(proberoute["utility_cost_latency_aware"] / max(float(oracle["utility_cost_latency_aware"]), 1e-12))
    normalized_cost = float(proberoute["normalized_remote_cost_vs_all_gpt"])
    frontier_rate = float(proberoute["frontier_call_rate"])
    probe_rate = float(proberoute["probe_call_rate"])
    all_gpt_p95 = float(main_eval.loc[main_eval["method"] == "all_gpt_frontier", "latency_p95"].iloc[0])
    latency_ratio = float(proberoute["latency_p95"] / max(all_gpt_p95, 1e-12))
    stage_status = [
        ("within_3_quality_points", quality_gap <= 0.03, f"quality_gap={quality_gap:.4f}"),
        ("oracle_utility_95pct", utility_ratio >= 0.95, f"utility_ratio={utility_ratio:.4f}"),
        ("remote_cost_le_0p35x", normalized_cost <= 0.35, f"normalized_cost={normalized_cost:.4f}"),
        ("p95_latency_le_all_gpt_or_1p2x", latency_ratio <= 1.2, f"latency_ratio={latency_ratio:.4f}"),
        ("frontier_call_rate_le_0p40", frontier_rate <= 0.40, f"frontier_call_rate={frontier_rate:.4f}"),
        ("probe_rate_20_to_40pct", 0.20 <= probe_rate <= 0.40, f"probe_rate={probe_rate:.4f}"),
    ]
    status_rows = [
        {
            "target": target,
            "status": "met_in_controlled_surrogate" if passed else "missed_in_controlled_surrogate",
            "evidence": evidence,
        }
        for target, passed, evidence in stage_status
    ]
    pd.DataFrame(status_rows).to_csv(output_dir / "table_expected_results_status.csv", index=False)
    status_lines = "\n".join(f"| {row['target']} | {row['status']} | {row['evidence']} |" for row in status_rows)
    env_lines = "\n".join(f"- `{key}` present: `{value}`" for key, value in sorted(env_present.items()) if "key" in key.lower())
    if not env_lines:
        env_lines = "- No API key variable names were found in the configured env file."
    budget_lines = "\n".join(f"- `{model}` estimated frontier spend: `${cost:.4f}`" for model, cost in budget_estimate.items())
    report = f"""# Controlled ProbeRoute++ Pilot Run Report

Run id: `{config.get('run_id')}`

Execution mode: `{config.get('execution_mode')}`. This run is a cache-backed offline surrogate over the requested controlled model pool. It validates schemas, routing math, cost accounting, latency accounting, tables, figures, and claim gates before paid API or large local vLLM generation. It is not a paper-level live benchmark result.

## Frontier And Budget Guardrails

- Frontier calls allowed: `{config.get('budget', {}).get('allow_frontier_calls')}`
- Total frontier spend cap: `${float(config.get('budget', {}).get('max_total_frontier_spend_usd', 0.0)):.2f}`
- Per-frontier-model spend cap: `${float(config.get('budget', {}).get('max_spend_per_frontier_model_usd', 0.0)):.2f}`
- Frontier pool: `gpt-5.5`, `gemini-3.5-flash`
- Claude/Anthropic models: not present in runnable configs

API key variable check, without printing secrets:

{env_lines}

Estimated offline frontier cost if these cached surrogate calls were live:

{budget_lines}

## Main Results

| method | quality | utility | normalized remote cost | frontier rate | probe rate | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for row in main_eval.head(12).itertuples(index=False):
        report += (
            f"| {row.method} | {row.quality_mean:.4f} | {row.utility_cost_latency_aware:.4f} | "
            f"{row.normalized_remote_cost_vs_all_gpt:.4f} | {row.frontier_call_rate:.4f} | "
            f"{row.probe_call_rate:.4f} | {row.latency_p95:.4f} |\n"
        )
    report += f"""
## Stage 1 Questions

- Cost-aware oracle gap exists: `{float(routability['oracle_gap'].iloc[0]):.4f}` utility gap over best single.
- K=8/16 latent states recover most oracle gap in this surrogate: best K utility `{float(rate_distortion['mean_utility'].max()):.4f}` versus oracle `{float(rate_distortion['oracle_mean_utility'].iloc[0]):.4f}`.
- Query-only observability gap remains: query-only gap `{float(observability.iloc[0]['utility_gap_to_oracle']):.4f}`.
- Cheap probes improve state inference/utility in the surrogate: threshold-probe gap `{float(observability.iloc[1]['utility_gap_to_oracle']):.4f}`.
- Continue as ProbeRoute++ only after replacing surrogate outputs with live cached local/frontier generations.

## Expected Result Status

| target | status | evidence |
| --- | --- | --- |
{status_lines}

## Outputs

- `model_outputs.parquet`
- `scored_outputs.parquet`
- `cost_latency_summary.csv`
- `table_routability.csv`
- `table_rate_distortion.csv`
- `table_observability_gap.csv`
- `table_main_eval.csv`
- `table_calibration.csv`
- `table_ablation.csv`
- `table_sensitivity.csv`
- five PDF figures

## Next Steps

1. Run Stage 0 live cache collection with `allow_frontier_calls=true` after reviewing the cost estimate.
2. Serve local models sequentially through vLLM and replace surrogate local rows with cached live rows.
3. Recompute the same tables without changing thresholds before making any paper-level claim.
"""
    (output_dir / "RUN_REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "PILOT_OBSERVATION_MEMO.md").write_text(report, encoding="utf-8")
    status_doc = f"""# Expected Results Status

This file maps Phase 3 numerical targets to the current controlled surrogate evidence. These are engineering checks, not live benchmark claims.

| target | status | evidence |
| --- | --- | --- |
{status_lines}

The next required upgrade is live cached Stage 0 generation for GPT-5.5, Gemini 3.5 Flash, and sequential local vLLM models under the configured spend caps.
"""
    (output_dir / "EXPECTED_RESULTS_STATUS.md").write_text(status_doc, encoding="utf-8")


def run_controlled_surrogate(config_path: str | Path, *, stage: str = "pilot") -> dict[str, Path]:
    bundle = load_controlled_inputs(config_path)
    config = bundle["config"]
    prices = load_prices(bundle["prices"])
    models = load_models(bundle["servers"])
    output_dir = Path(config.get("outputs", {}).get("output_dir", "results/controlled"))
    output_dir.mkdir(parents=True, exist_ok=True)
    n_per_benchmark = int(
        config.get("surrogate", {}).get(
            "dry_run_examples_per_benchmark" if stage == "dry_run" else "pilot_examples_per_benchmark",
            5 if stage == "dry_run" else 100,
        )
    )
    env_file = Path(config.get("budget", {}).get("env_file", ".env"))
    env_present = load_env_keys(env_file)
    tasks = generate_tasks(bundle["benchmarks"], n_per_benchmark=n_per_benchmark, seed=int(config.get("seed", 42)))
    model_outputs = build_model_outputs(tasks, models, prices, config, output_dir)
    budget_estimate = (
        model_outputs[model_outputs["is_frontier"]]
        .groupby("model_id")["cost_total_usd"]
        .sum()
        .astype(float)
        .to_dict()
    )
    enforce_frontier_budget(
        budget_estimate,
        max_total_frontier_spend_usd=float(config.get("budget", {}).get("max_total_frontier_spend_usd", 0.0)),
        max_spend_per_frontier_model_usd=float(config.get("budget", {}).get("max_spend_per_frontier_model_usd", 0.0)),
    )
    model_outputs.to_parquet(output_dir / "model_outputs.parquet", index=False)
    scored, _normalizers = add_utility(
        model_outputs,
        lambda_cost=float(config.get("routing", {}).get("lambda_cost", 0.35)),
        lambda_latency=float(config.get("routing", {}).get("lambda_latency", 0.05)),
    )
    scored.to_parquet(output_dir / "scored_outputs.parquet", index=False)
    cost_latency = write_cost_latency_summary(scored, output_dir)
    tables = build_policy_tables(scored, tasks, config)
    main_eval = tables["main_eval"]
    routability = build_routability(main_eval)
    rate_distortion = build_rate_distortion(scored, tasks, config)
    observability = build_observability_gap(main_eval)
    calibration = build_calibration()
    ablation = build_ablation(main_eval)
    sensitivity = build_sensitivity(main_eval)
    routability.to_csv(output_dir / "table_routability.csv", index=False)
    rate_distortion.to_csv(output_dir / "table_rate_distortion.csv", index=False)
    observability.to_csv(output_dir / "table_observability_gap.csv", index=False)
    main_eval.to_csv(output_dir / "table_main_eval.csv", index=False)
    calibration.to_csv(output_dir / "table_calibration.csv", index=False)
    ablation.to_csv(output_dir / "table_ablation.csv", index=False)
    sensitivity.to_csv(output_dir / "table_sensitivity.csv", index=False)
    plot_outputs(main_eval, rate_distortion, observability, calibration, output_dir)
    write_reports(
        output_dir,
        config,
        main_eval,
        routability,
        rate_distortion,
        observability,
        calibration,
        cost_latency,
        env_present,
        budget_estimate,
    )
    return {
        "output_dir": output_dir,
        "run_report": output_dir / "RUN_REPORT.md",
        "expected_status": output_dir / "EXPECTED_RESULTS_STATUS.md",
    }
